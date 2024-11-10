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
import g4f
from g4f.client import Client


@TRAINER_REGISTRY.register()
class KMeansPlusTrainer(SimpleTrainer):
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.clip_model = load_clip(cfg)

    def train(self):
        start_time = time.time()
        self.encode()
        self.clusterByK()
        print("-" * 20)
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        print("********finished********")

    def clusterByK(self):
        max_sc, num_cluster, labels = self.cluster()
        with open(os.path.join(self.output_dir, "coarse_grained_semantic.txt"), 'w') as f:
            f.write(str(num_cluster) + '\n')
            labels = labels.tolist()
            for i in labels:
                f.write(str(i) + '\n')

    def encode(self):
        cfg = self.cfg
        with open(cfg.TRAINER.BATSTYLER.CLASSDIR, 'r') as fr:
            lines = fr.readlines()
        classnames = [line.strip() for line in lines]
        self.classnames = classnames
        with torch.no_grad():
            text_features = self.clip_model.encode_text(clip.tokenize(classnames).cuda())
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        self.text_features = text_features
        

    def cluster(self, max_iter=1000):

        n_samples, n_features = self.text_features.shape
        max_sc = 0
        clusters = 0
        labels_k = None
        
        for i in range(1, 5):
            centroids = self.text_features[torch.randperm(self.text_features.shape[0])[:4], :]
            for _ in range(max_iter):
                similarity = F.cosine_similarity(self.text_features.unsqueeze(1), centroids.unsqueeze(0), dim=2)
                labels = similarity.argmax(axis=1)

                new_centroids = torch.zeros_like(centroids)

                for j in range(i):
                    cluster_points = self.text_features[labels.cpu() == j]
                    if cluster_points.shape[0] > 0:
                        new_centroid = cluster_points.mean(dim=0)
                        new_centroids[j] = new_centroid

                centroids = new_centroids

            similarity = F.cosine_similarity(self.text_features.unsqueeze(1), centroids.unsqueeze(0), dim=2)
            labels = similarity.argmax(axis=1)

            sc = self.silhouette_coefficient_cosine(self.text_features, labels)
            if sc > max_sc:
                max_sc = sc
                clusters = i
                labels_k = labels
        
        return max_sc, clusters, labels_k

    def silhouette_coefficient_cosine(self, X, labels):
        n_samples = X.shape[0]
        silhouette_avg = 0.0

        n_samples, n_features = X.shape
        silhouette_scores = torch.zeros(n_samples)
        for i in range(n_samples):
            same_cluster = (labels == labels[i])
            if same_cluster.sum() > 1:
                a = 1 - F.cosine_similarity(X[i:i+1, :].unsqueeze(1), X[same_cluster].unsqueeze(0), dim=2).mean()
            else:
                a = 0

            b_min = float('inf')
            for label in torch.unique(labels):
                if label != labels[i]:
                    different_cluster = (labels == label)
                    if different_cluster.sum() > 0:
                        b = 1 - F.cosine_similarity(X[i:i+1, :].unsqueeze(1), X[different_cluster].unsqueeze(0), dim=2).mean()
                        b_min = min(b_min, b)          
            s = (b_min - a) / max(a, b_min)
            silhouette_scores[i] = s

        return silhouette_scores.mean().item()
    

@TRAINER_REGISTRY.register()
class LLMTrainer(SimpleTrainer):
    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self):
        start_time = time.time()
        self.query()
        print("-" * 20)
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        print("********finished********")

    
    def query(self):
        cfg = self.cfg
        with open(cfg.TRAINER.BATSTYLER.CLASSDIR, 'r') as fr:
            lines = fr.readlines()
        classnames = [line.strip() for line in lines]
        self.classnames = np.array(classnames)

        with open(os.path.join(self.output_dir, "coarse_grained_semantic.txt"), 'r') as f:
            lines = f.readlines()
        
        num_cluster = int(lines[0].strip())
        labels = np.array([int(line.strip()) for line in lines[1:]])

        labelset = np.unique(labels)

        new_name = cfg.TRAINER.BATSTYLER.CLASSDIR.split('/')[-1][:-4] + '_new.txt'
        fw = open("./batstyler/semantics/" + new_name, 'w')
        
        for l in labelset:
            classes = self.classnames[labels == l]
            
            client = Client()

            response = client.chat.completions.create(
                model="gpt-4-32k",
                messages=[
                    {
                        "role": "user",
                        "content": "Please tell me " + str(classes) + " have in common with only three words. If not, it can be nothing. The format of output is divided by ' '  "
                    }
                ]
            )

            semantic = response.choices[0].message.content.split(' ')
            for s in semantic:
                fw.write(s + '\n')
        fw.close()
