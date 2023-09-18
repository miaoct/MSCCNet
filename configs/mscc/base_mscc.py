'''base config for fcn'''
# config for dataset
DATASET_CFG = {
    'type': '',
    'rootdir': '',
    'train': {
        'set': 'train',
        'aug_opts': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ],
    },
    'val': {
        'set': 'val',
        'aug_opts': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    },
    'test': {
        'set': 'test',
        'aug_opts': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    }
}
# config for dataloader
DATALOADER_CFG = {
    'train': {
        'batch_size': 16, 'num_workers': 16, 'shuffle': True, 'pin_memory': True, 'drop_last': True,
    },
    'val': {
        'batch_size': 1, 'num_workers': 1, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    },
    'test': {
        'batch_size': 1, 'num_workers': 1, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    }
}
# config for optimizer
OPTIMIZER_CFG = {
    'type': 'sgd',
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'params_rules': {},
}
# config for scheduler
SCHEDULER_CFG = {
    'type': 'poly',
    'max_epochs': 0,
    'power': 0.9,
}
# config for losses
LOSSES_CFG = {
    'pix_aux': {
        'celoss': {'scale_factor': 0.4, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'pix_cls': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
    'img_cls': {
        'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'}
    },
}
# config for segmentor
SEGMENTOR_CFG = {
    'type': 'fcn',
    'num_classes': 2,
    'benchmark': True,
    'align_corners': False,
    'backend': 'nccl',
    'norm_cfg': {'type': 'syncbatchnorm'},
    'act_cfg': {'type': 'relu', 'inplace': True},
    'backbone': {
        'type': 'resnet101',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 8,
        'use_stem': True,
        'selected_indices': (2, 3),
    },
    'classifier': {
        'last_inchannels': 2048,
    },
    'head': {
        'in_channels': 2048,
        'feats_channels': 512,
        'out_channels': 512,
        'dropout': 0.1,
        'spectral_height': 64,
        'spectral_width': 64,
        'spectral_k': 8,
    },
    'auxiliary': {
        'in_channels': 1024,
        'out_channels': 512,
        'dropout': 0.1,
    }
}
# config for inference
INFERENCE_CFG = {
    'mode': 'whole',
    'opts': {}, 
    'tricks': {
        'multiscale': [1],
        'flip': False,
        'use_probs_before_resize': False
    },
}
# config for common
COMMON_CFG = {
    'output_dir':'',
    'work_dir': 'ckpts',
    'logfilepath': '',
    'log_interval_iterations': 50,
    'eval_interval_epochs': 10,
    'save_interval_epochs': 1,
    'resultsavepath': '',
}