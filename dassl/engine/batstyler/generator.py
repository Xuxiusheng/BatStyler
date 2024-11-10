from torch import nn
import os
import torch
from .clip import clip
import torch.nn.functional as F

class StyleGenerator(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        self.template = "a X style of a "

        self.styles = torch.load(os.path.join(cfg.OUTPUT_DIR.replace('linear', 'learner'), "style.pth")).cuda()
        self.styles.requires_grad_(False)

    def get_text_feature(self, classname, idx):
        text = self.template + classname
        with torch.no_grad():
            tokenize = clip.tokenize(text).cuda()
            embedding = self.clip_model.token_embedding(tokenize)

            prefix = embedding[:, :2, :]
            suffix = embedding[:, 3:, :]
            prompt = torch.cat(
                [
                    prefix, 
                    self.styles[idx:idx+1, :, :], 
                    suffix
                ], dim=1
            )
            output = self.clip_model.forward_text(prompt, tokenize)
            output = F.normalize(output, p=2, dim=1)
        return output.squeeze(0).cpu()

    def traindata(self):
        cfg = self.cfg
        train_data = {
            "classnames": self.classnames, 
            "generator": self, 
            "n_cls": len(self.classnames), 
            "n_styles": self.styles.shape[0], 
        }
        return train_data