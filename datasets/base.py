
import cv2
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from .transforms import Resize, RandomCrop, RandomFlip, PhotoMetricDistortion, RandomRotation, Padding, ToTensor, Normalize, Compose


'''BaseDataset'''
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, logger_handle, dataset_cfg):
        assert mode in ['TRAIN', 'VAL', 'TEST']
        self.mode = mode
        self.logger_handle = logger_handle
        self.dataset_cfg = dataset_cfg
        self.transforms = Compose(self.constructtransforms(self.dataset_cfg['aug_opts']))
    '''pull item'''
    def __getitem__(self, index):
        raise NotImplementedError('not be implemented')
    '''length'''
    def __len__(self):
        raise NotImplementedError('not be implemented')
    '''sync transform'''
    def synctransform(self, sample):
        assert hasattr(self, 'transforms') and self.transforms, 'undefined transforms'
        sample = self.transforms(sample)
        return sample
    '''read sample'''
    def read(self, imagepath, annpath, with_ann=True):
        assert self.mode in ['TRAIN', 'VAL', 'TEST']
        # read image
        image = cv2.imread(imagepath)
        # read annotation
        if annpath.endswith('.png') or annpath.endswith('.jpg') or annpath.endswith('.tif') or annpath.endswith('.bmp') or annpath.endswith('.JPG'):
            segmentation = cv2.imread(annpath, cv2.IMREAD_GRAYSCALE) if with_ann else np.zeros((image.shape[0], image.shape[1]))
        else:
            raise NotImplementedError('Unsupport data type of %s' % annpath.split('.')[-1])
        # return sample
        sample = {
            'image': image, 
            'segmentation': segmentation.copy(), 
            'width': segmentation.shape[1], 
            'height': segmentation.shape[0]
        }
        if self.mode == 'TEST' or self.mode == 'VAL': sample.update({'groundtruth': segmentation.copy()})
        return sample
    '''construct the transforms'''
    def constructtransforms(self, aug_opts):
        # obtain the transforms
        transforms = []
        supported_transforms = {
            'Resize': Resize,
            'RandomCrop': RandomCrop,
            'RandomFlip': RandomFlip,
            'PhotoMetricDistortion': PhotoMetricDistortion,
            'RandomRotation': RandomRotation,
            'Padding': Padding,
            'ToTensor': ToTensor,
            'Normalize': Normalize
        }
        for aug_opt in aug_opts:
            key, value = aug_opt
            assert key in supported_transforms, 'unsupport transform %s' % key
            transforms.append(supported_transforms[key](**value))
        # return the transforms
        return transforms
    