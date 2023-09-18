'''
Function:
    Load the pixel-level DEFACTO dataset
Author:
    Changtao Miao
    refer to Zhenchao Jin 'https://github.com/SegmentationBLWX/sssegmentation'
'''
import os
import glob
from .base import BaseDataset


'''DEFACTODataset'''
class DEFACTODataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(DEFACTODataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        data_name = dataset_cfg['type']
        image_dir = os.path.join(rootdir, data_name)
        if self.mode.lower() == 'test':
            _split_f = os.path.join(image_dir, 'test_vis.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        # obatin images
        self.images = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                line = line.rstrip()
                words = line.split()
                filepath = os.path.join(image_dir, words[0])

                assert os.path.isfile(filepath)
                self.images.append((filepath, int(words[1])))
        # # print
        # print('Loading {} images in the folder {}'.format(self.__len__(), self.image_dir)) 
    '''pull item'''
    def __getitem__(self, index):
        imagepath, label = self.images[index]
        # seek the corresponding pixe-level annotation path
        # "/home/ctmiao/data/preprocessed/Pixel_Cross-test/DEFACTO/Mask/fake/0.jpg_IN_1.jpg"
        # "/home/ctmiao/data/preprocessed/Pixel_Cross-test/DEFACTO/Image/fake/0.jpg_IN_1.jpg.tif"
        if label == 0:
            maskpath = imagepath.replace('Image', 'Mask')
        else:
            maskpath = imagepath.replace('Image', 'Mask')[:-4]
        assert os.path.isfile(maskpath)
        # read files
        sample = self.read(imagepath, maskpath, self.dataset_cfg.get('with_ann', True))
        # add image-level annotation
        sample.update({'imagelabel': label})
        sample.update({'id': '_'.join(imagepath.split('/')[-2:])})
        # data aug and transform
        if self.mode == 'TRAIN':
            sample['segmentation'][sample['segmentation'] == 255] = 1.
            sample = self.synctransform(sample)
        else:
            sample['groundtruth'][sample['groundtruth'] == 255] = 1.
            sample = self.synctransform(sample)
        return sample
    '''length'''
    def __len__(self):
        return len(self.images)