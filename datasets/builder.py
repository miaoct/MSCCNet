
from .ff import FFDataset


'''BuildDataset'''
def BuildDataset(mode, logger_handle, dataset_cfg):
    # supported datasets
    supported_datasets = {
        'Pixel_FF++': FFDataset,
    }
    # parse
    cfg = dataset_cfg[mode.lower()].copy()
    if 'train' in dataset_cfg: dataset_cfg.pop('train')
    if 'val' in dataset_cfg: dataset_cfg.pop('val')
    if 'test' in dataset_cfg: dataset_cfg.pop('test')
    dataset_cfg.update(cfg)
    assert dataset_cfg['type'] in supported_datasets, 'unsupport dataset type %s' % dataset_cfg['type']
    # return
    dataset = supported_datasets[dataset_cfg['type']](mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
    return dataset