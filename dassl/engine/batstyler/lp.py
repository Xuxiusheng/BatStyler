import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerBase, SimpleTrainer
from dassl.metrics import compute_accuracy
from torch import nn
from .clip import clip, load_clip
import os
import random
from dassl.optim import build_optimizer, build_lr_scheduler
from collections import OrderedDict
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
import datetime
import time
import torch.nn.functional as F
from dassl.data import DataManager
from dassl.evaluation import build_evaluator
from tqdm import tqdm
import numpy as np
from .generator import StyleGenerator

class MLPNet(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg
        self.clip_model = load_clip(cfg)
        self.s = cfg.TRAINER.BATSTYLER.S
        self.m = cfg.TRAINER.BATSTYLER.M
        self.eps = cfg.TRAINER.BATSTYLER.EPS
        self.input_dim = self.clip_model.visual.output_dim
        self.fc = nn.Linear(self.input_dim, len(classnames), bias=False)
        nn.init.xavier_normal_(self.fc.weight)

    def arcface(self, input, labels):
        for param in self.fc.parameters():
            param.data = F.normalize(param.data, p=2, dim=1)

        x = F.normalize(input, p=2, dim=1)
        wf = self.fc(x)
        numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

    def forward(self, input):
        return self.fc(input)

    def encode_image(self, image):
        image_features = self.clip_model.encode_image(image)
        return image_features


@TRAINER_REGISTRY.register()
class LinearTrainer(SimpleTrainer):
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.cfg = cfg
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.build_model()
        self.build_data_loader()

        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = ([0, 0, 0, 0], 0)

        self.test_interval = cfg.TRAINER.BATSTYLER.INTERVAL
    
    def build_model(self):
        cfg = self.cfg
        with open(cfg.TRAINER.BATSTYLER.CLASSDIR, 'r') as f:
            lines = f.readlines()
        # self.classnames = [" ".join(line.strip().lower().split("_")) for line in lines]
        self.classnames = [" ".join(line.strip().split("_")) for line in lines]
        self.loss_fn = nn.NLLLoss()
        self.model = MLPNet(cfg, self.classnames).cuda()
        for name, param in self.model.named_parameters():
            if "clip" in name:
                param.requires_grad_(False)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

    def build_data_loader(self):
        cfg = self.cfg
        self.generator = StyleGenerator(cfg, self.classnames, self.model.clip_model)
        train_data = self.generator.traindata()
        dm = DataManager(cfg, train_data)
        
        self.train_loader_x = dm.train_loader_x
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm

    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch()
            if self.epoch % self.test_interval == 0:
                self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        end = time.time()
        s1 = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            if meet_freq:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))
        
        n_iter = self.epoch * self.num_batches + self.batch_idx
        for name, meter in losses.meters.items():
            self.write_scalar("train/" + name, meter.avg, n_iter)
        self.write_scalar("train/lr", self.get_current_lr(), n_iter)
        e1 = time.time()
        print(e1 - s1)

        end = time.time()
        
    def forward_backward(self, batch):
        text_feature, label = self.parse_batch_train(batch)
        loss = self.model.arcface(text_feature, label)
        output = self.model(text_feature)
        self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary
        
    def parse_batch_train(self, batch):
        text_feature, label = batch["text_feature"].cuda(), batch["label"].cuda()
        return text_feature, label
    
    def after_epoch(self):
        result, is_best = self.test(split='test')
        if is_best:
            self.save_model(
                self.epoch,
                self.output_dir,
                val_result=result,
                model_name="model-best.pth"
            )
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
    
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.cuda()
        label = label.cuda()
        return input, label
    
    def model_inference(self, input):
        image_features = self.model.encode_image(input)
        image_features = F.normalize(image_features, p=2, dim=1)
        return self.model(image_features)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        result = []
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        true_label = []
        pred_label = []
        s1 = time.time()
        for data_loader_domain in data_loader:
            print(f"Evaluate on the *{split}* set")
            for batch_idx, batch in enumerate(tqdm(data_loader_domain)):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process(output, label)
                true_label.extend(label.cpu().numpy().tolist())
                pred_label.extend(output.argmax(dim=1).cpu().numpy().tolist())
            e1 = time.time()
            print(e1 - s1)
            results = self.evaluator.evaluate()

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)
            result.append(list(results.values())[0])
            self.evaluator.reset()

        mean_acc = np.mean(result)
        print("-" * 10)
        print(f"Mean-Acc:{mean_acc}")
        print("*" * 10)
        is_best = False
        if self.best_result[-1] < mean_acc:
            self.best_result = (result, mean_acc)
            is_best = True
        return result, is_best