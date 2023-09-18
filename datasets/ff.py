'''
Function:
    Load the pixel-level FF++ dataset
Author:
    Changtao Miao
    refer to Zhenchao Jin 'https://github.com/SegmentationBLWX/sssegmentation'
'''
import os
import glob
from .base import BaseDataset


'''FFDataset'''
class FFDataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(FFDataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        data_name = dataset_cfg['type']
        self.compression = dataset_cfg['compression']
        fake_type = dataset_cfg['fake_type']
        self.image_dir = os.path.join(rootdir, data_name, self.compression)
        if self.mode.lower() == 'train':
            _split_f = os.path.join(rootdir, data_name, 'train.txt')
        elif self.mode.lower() == 'val':
            _split_f = os.path.join(rootdir, data_name, 'valid.txt')
        elif self.mode.lower() == 'test':
            _split_f = os.path.join(rootdir, data_name, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        # obatin images
        self.images = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                fake_name = line.rstrip()
                
                for src in fake_type:
                    filelist = glob.glob(os.path.join(self.image_dir, src, fake_name, '*.png'))
                    # if len(filelist) < 20:
                    #     print(f'{src},{fake_name},{len(filelist)}')

                    for fake_path in filelist:
                        assert os.path.isfile(fake_path)
                        self.images.append((fake_path, int(1)))

                        real_name = fake_name.split('_')[0]
                        real_id = os.path.basename(fake_path)
                        real_path = os.path.join(self.image_dir, 'youtube', real_name, real_id)
                        assert os.path.isfile(real_path)
                        self.images.append((real_path, int(0)))
        # # print
        # print('Loading {} images in the folder {}'.format(self.__len__(), self.image_dir)) 
    '''pull item'''
    def __getitem__(self, index):
        imagepath, label = self.images[index]
        # seek the corresponding pixe-level annotation path
        maskpath = imagepath.replace(self.compression, 'Mask')
        assert os.path.isfile(maskpath)
        # read files
        sample = self.read(imagepath, maskpath, self.dataset_cfg.get('with_ann', True))
        # add image-level annotation
        sample.update({'imagelabel': label})
        sample.update({'id': '_'.join(imagepath.split('/')[-3:])})
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