'''msccnet_resnet50os8_FF++_all'''
import os
from .base_mscc import *


# modify dataset config ['NeuralTextures','Deepfakes','Face2Face','FaceShifter','FaceSwap']
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'Pixel_FF++',
    'rootdir': '/home/ctmiao/data/preprocessed/',
    'compression': 'C40',
    'fake_type': ['NeuralTextures','Deepfakes','Face2Face','FaceShifter','FaceSwap'],
})
DATASET_CFG['train']['aug_opts'] = [
    ('Resize', {'output_size': (520, 520), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
]
DATASET_CFG['val']['aug_opts'] = [
    ('Resize', {'output_size': (512, 512), 'keep_ratio': False, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (512, 512), 'keep_ratio': False, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
DATALOADER_CFG.update({
    'train': {
        'batch_size': 64, 'num_workers': 8, 'shuffle': True, 'pin_memory': True, 'drop_last': True,
    },
    'val': {
        'batch_size': 1, 'num_workers': 8, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    },
    'test': {
        'batch_size': 1, 'num_workers': 8, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    }
})
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update({
    'lr': 0.009
})
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 30
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
LOSSES_CFG.update({
    'pix_cls_map': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'pix_cls_mask': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'img_cls': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
})
LOSSES_CFG.pop('pix_aux')
LOSSES_CFG.pop('pix_cls')
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'type': 'msccnet',
    'num_classes': 2,
    'backbone': {
        'type': 'resnet50',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 8,
        'use_stem': True,
        'selected_indices': (0, 1, 2, 3),
    },
    'head': {
        'in_channels': 2048,
        'feats_channels': 512,
        'out_channels': 512,
        'dropout': 0.1,
        'spectral_height': 64,
        'spectral_width': 64,
        'spectral_k': 2,
    },
    'auxiliary': None,
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()

# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG.update({
    'output_dir':'/home/ctmiao/output/models/MFCC/',
    'work_dir': 'msccnet-k2_resnet50os8_ffc40_all',
    'logfilepath': 'msccnet_resnet50os8',
    'log_interval_iterations': 1000,
    'eval_interval_epochs': 30,
    'save_interval_epochs': 1,
    'resultsavepath': 'msccnet_resnet50os8_results.pkl',
})