# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
# from core.dn_crvd_dataset import DNCRVDDataset

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
MODELS = Registry('model', parent=MMCV_MODELS)
BACKBONES = MODELS
COMPONENTS = MODELS
LOSSES = MODELS
