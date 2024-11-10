from .clip import *
import torch

def load_clip(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    assert backbone_name in clip._MODELS, f"{backbone_name} not find in {clip.available_models()}"
    model_path = clip._download(clip._MODELS[backbone_name], root=cfg.TRAINER.BATSTYLER.WEIGHT_PATH)
    try:
        model = torch.jit.load(model_path, map_location="cuda").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cuda")

    model = clip.build_model(state_dict or model.state_dict()).cuda()
    return model