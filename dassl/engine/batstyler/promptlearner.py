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
import math


class ETFHead(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg
        self.clip_model = load_clip(cfg)
        self.n_styles = cfg.TRAINER.BATSTYLER.N_STYLES
        self.ctx_dim = self.clip_model.ln_final.weight.shape[0]
        self.classnames = classnames
        self.n_cls = len(classnames)
        self.generate_orthogonal(self.clip_model.visual.output_dim)
        ctx_vectors = torch.empty(self.n_styles, 1, self.ctx_dim, dtype=torch.float32)
        ctx_vectors.requires_grad_(True)
        ctx_vectors = nn.init.normal_(ctx_vectors, std=0.02)
        self.style_embedding = nn.Parameter(ctx_vectors)

        self.text = "a X style of a "

    def generate_orthogonal(self, in_features):
        rand_mat = np.random.random(size=(in_features, self.n_styles))
        orth_vec, _ = np.linalg.qr(rand_mat)
        orth_vec = torch.tensor(orth_vec).float()
        assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(self.n_styles), atol=1.e-7), \
            "The max irregular value is : {}".format(torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(self.n_styles))))
        
        i_nc_nc = torch.eye(self.n_styles)
        one_nc_nc = torch.mul(torch.ones(self.n_styles, self.n_styles), (1 / self.n_styles))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc), math.sqrt(self.n_styles / (self.n_styles - 1)))
        self.etf_vec = etf_vec.cuda()

        with torch.no_grad():
            self.tokenized_c = clip.tokenize(self.classnames).cuda()
            self.c_output = self.clip_model.encode_text(self.tokenized_c)
            self.c_norm = self.c_output / self.c_output.norm(dim=1, keepdim=True)

    def loss(self, x, label):
        output = torch.exp(torch.mm(x, self.etf_vec))
        label_col = torch.diag(output)
        diversity_loss = -torch.log(label_col / output.sum(dim=1)).mean()
        content_consistency_loss = torch.tensor(0.0).cuda()

        for i in label:
            sc_template = []
            for cls_idx in range(self.n_cls):
                sc_template.append(self.text + " " + self.classnames[cls_idx])
            with torch.no_grad():
                tokenized_sc = clip.tokenize(sc_template).cuda()
                sc_embedding = self.clip_model.token_embedding(tokenized_sc)
            sc_prefix = sc_embedding[:, :2, :]
            sc_suffix = sc_embedding[:, 3:, :]

            sc_prompt = torch.cat(
                [
                    sc_prefix, 
                    self.style_embedding[i].repeat(sc_prefix.shape[0], 1, 1), 
                    sc_suffix
                ], dim=1
            )

            sc_outputs = self.clip_model.forward_text(sc_prompt, tokenized_sc)
        
            sc_norm = sc_outputs / sc_outputs.norm(dim=1, keepdim=True)
            assert self.c_norm.shape == sc_norm.shape

            zimm = F.cosine_similarity(sc_norm.unsqueeze(1), self.c_norm.unsqueeze(0), dim=2)
            exp_zimm = torch.exp(zimm)
            per_zimm = exp_zimm / exp_zimm.sum(dim=1, keepdim=True)

            content_consistency_loss += -torch.log(per_zimm.diag()).mean()

        content_consistency_loss = content_consistency_loss / x.shape[0]

        return diversity_loss, content_consistency_loss


    def forward(self, batch, batch_num):
        start = batch * batch_num
        style_embedding = self.style_embedding[start:start+batch_num, :, :]
        n_style = style_embedding.shape[0]
        with torch.no_grad():
            tokenize = clip.tokenize(self.text).cuda()
            embedding = self.clip_model.token_embedding(tokenize)
            prefix = embedding[:, :2, :]
            suffix = embedding[:, 3:, :]

        prompt = torch.cat(
            [
                prefix.repeat(n_style, 1, 1), 
                style_embedding, 
                suffix.repeat(n_style, 1, 1)
            ], dim=1
        )

        output = self.clip_model.forward_text(prompt, tokenize.repeat(n_style, 1))
        output = F.normalize(output, p=2, dim=1)

        return output, torch.arange(start, start + n_style).long().cuda()

        

@TRAINER_REGISTRY.register()
class PromptLearnerTrainer(SimpleTrainer):
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.cfg = cfg
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.weight_path = os.path.join(self.output_dir, "style.pth")
        self.n_style = cfg.TRAINER.BATSTYLER.N_STYLES
        self.style_batch = cfg.TRAINER.BATSTYLER.STYLE_BATCH
        assert self.n_style % self.style_batch ==0, "N_style % style_batch must 0"

        self.build_model()
        
    def build_model(self):
        cfg = self.cfg
        with open(cfg.TRAINER.BATSTYLER.COARSESEMANTICSET, 'r') as f:
            lines = f.readlines()
        self.classnames = [" ".join(line.strip().split('_')) for line in lines]
        self.model = ETFHead(cfg, self.classnames).cuda()
        for name, param in self.model.named_parameters():
            if "clip" in name:
                param.requires_grad_(False)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM, epochs=self.max_epoch * self.n_style)
        self.register_model("model", self.model, self.optim)

    def train(self):
        start_time = time.time()
        num_batches = self.n_style // self.style_batch
        self.model.train()
        for epoch in range(self.max_epoch):
            for i in range(num_batches):
                self.optim.zero_grad()
                output, label = self.model(i, self.style_batch)
                style_diversity_loss, content_consistency_loss = self.model.loss(output, label)
                loss = style_diversity_loss + content_consistency_loss
                loss.backward()
                self.optim.step()

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    current_lr = self.optim.param_groups[0]["lr"]
                    info = []
                    info += [f"epoch [{epoch + 1}/{self.max_epoch}]"]
                    info += [f"batch [{i + 1}/{num_batches}]"]
                    info += [f"loss {loss.item()}"]
                    info += [f"lr {current_lr:.4e}"]
                    print(" ".join(info))
            self.sched.step()

        self.model.eval()
        torch.save(self.model.style_embedding, self.weight_path)
        print("-" * 20)
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        print("********finished********")