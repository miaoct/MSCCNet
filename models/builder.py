
import copy
from .msccnet import MSCCNet

'''BuildModel'''
def BuildModel(segmentor_cfg, mode):
    supported_models = {
        'msccnet': MSCCNet,
    }
    selected_segmentor = supported_models[segmentor_cfg['type']]
    segmentor_cfg = copy.deepcopy(segmentor_cfg)
    segmentor_cfg.pop('type')
    return selected_segmentor(segmentor_cfg, mode=mode)