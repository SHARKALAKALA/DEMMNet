import torch

from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, adjust_lr
from .ranger.ranger import Ranger
from .ranger.ranger913A import RangerVA
from .ranger.rangerqh import RangerQH

import torch.nn as nn


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'city', 'irseg']
    if cfg['dataset'] == 'nyuv2':
        from .datasets.nyuv2 import NYUv2
        return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd':
        from .datasets.sunrgbd import SUNRGBD
        return SUNRGBD(cfg, mode='train'), SUNRGBD(cfg, mode='test')
    if cfg['dataset'] == 'city':
        from .datasets.city import CITY
        return CITY(cfg, mode='train'), CITY(cfg, mode='val'), CITY(cfg, mode='test')
    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

def get_model(cfg):
    if cfg['model_name'] == 'seg_light':
        from .models.seg_light import seg_seg
        return seg_seg(num_classes=cfg['n_classes'])
