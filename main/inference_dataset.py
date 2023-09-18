'''
Function:
    Visualize the segmentation results by using our segmentors
Author:
    Changtao Miao
    refer to Zhenchao Jin 'https://github.com/SegmentationBLWX/sssegmentation'
'''
import sys
sys.path.append('')
import os
import cv2
import copy
import torch
import warnings
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from configs import BuildConfig
from datasets import BuildDataset
from models import BuildModel
from utils import ( BuildDistributedDataloader, Logger, checkdir, loadcheckpoints, BuildPalette)
warnings.filterwarnings('ignore')


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed testing', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=4, type=int)
    parser.add_argument('--outputfile', dest='outputfile', help='name to save output image(s)', type=str, default='')
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--modeltype', dest='modeltype', help='name to save output image(s)', type=str, default='')
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to resume from', type=str, required=True)
    args = parser.parse_args()
    return args


'''Demo'''
class Demo():
    def __init__(self):
        self.cmd_args = parseArgs()
        self.cfg, self.cfg_file_path = BuildConfig(self.cmd_args.cfgfilepath)
        assert self.cmd_args.outputfile, 'imagepath or imagedir should be specified'
        assert torch.cuda.is_available(), 'cuda is not available'
        # init distributed training
        self.cfg.DATALOADER_CFG['test']['batch_size'] = self.cmd_args.nproc_per_node
        dist.init_process_group(backend=self.cfg.SEGMENTOR_CFG.get('backend', 'nccl'))
    '''start'''
    def start(self):
        cmd_args, cfg, cfg_file_path = self.cmd_args, self.cfg, self.cfg_file_path
        # initialize logger_handle
        logger_handle = Logger(cfg.COMMON_CFG['logfilepath'])
        # number of gpus, for distribued testing, only support a process for a GPU
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node != self.cmd_args.nproc_per_node:
            if (self.cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
                logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default')
        ngpus_per_node = self.cmd_args.nproc_per_node
        # check work dir
        # checkdir(cfg.COMMON_CFG['work_dir'])
        # cuda detect
        use_cuda = torch.cuda.is_available()
        # build segmentor
        cfg.SEGMENTOR_CFG['backbone']['pretrained'] = False
        segmentor = BuildModel(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='TEST')
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        # build dataset and dataloder
        dataset_cfg = copy.deepcopy(cfg.DATASET_CFG)
        # dataset_cfg.update({'type': 'Pixel_FF++_test',}) 
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.DATALOADER_CFG)
        batch_size, num_workers = dataloader_cfg['test']['batch_size'], dataloader_cfg['test']['num_workers']
        batch_size_per_node = batch_size // ngpus_per_node
        num_workers_per_node = num_workers // ngpus_per_node
        dataloader_cfg['test'].update({'batch_size': batch_size_per_node, 'num_workers': num_workers_per_node})
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['test'])
        # build palette
        palette = BuildPalette(dataset_type=dataset_cfg['type'], num_classes=cfg.SEGMENTOR_CFG['num_classes'], logger_handle=logger_handle)
        # load checkpoints
        cmd_args.local_rank = 0
        checkpoints = loadcheckpoints(cmd_args.checkpointspath, logger_handle=logger_handle, cmd_args=cmd_args)
        try:
            segmentor.load_state_dict(checkpoints['model'])
        except Exception as e:
            logger_handle.warning(str(e) + '\n' + 'Try to load checkpoints by using strict=False')
            segmentor.load_state_dict(checkpoints['model'], strict=False)
        # set eval
        segmentor.eval()
        # start to test
        inference_cfg = copy.deepcopy(cfg.INFERENCE_CFG)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        # if not cmd_args.imagedir:
        #     imagepaths = [cmd_args.imagepath]
        # else:
        #     imagenames = os.listdir(cmd_args.imagedir)
        #     imagepaths = [os.path.join(cmd_args.imagedir, name) for name in imagenames]
        # pbar = tqdm(range(len(imagepaths)))
        dataloader.sampler.set_epoch(0)
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, samples in pbar:
            pbar.set_description('Processing %s/%s in rank %s' % (batch_idx+1, len(dataloader), cmd_args.local_rank))
            imageids, image, widths, heights = samples['id'], samples['image'], samples['width'], samples['height']
            infer_tricks = inference_cfg['tricks']
            cascade_cfg = infer_tricks.get('cascade', {'key_for_pre_output': 'memory_gather_logits', 'times': 1, 'forward_default_args': None})
            # sample = dataset.read(imagepath, 'none.png', False)
            # image = sample['image']
            # sample = dataset.synctransform(sample, 'all')
            # image_tensor = sample['image'].unsqueeze(0).type(FloatTensor)
            for idx in range(cascade_cfg['times']):
                forward_args = None
                if idx > 0: 
                    output_list = [
                        F.interpolate(outputs, size=output_list[-1].shape[2:], mode='bilinear', align_corners=segmentor.align_corners) for outputs in output_list
                    ]
                    forward_args = {cascade_cfg['key_for_pre_output']: sum(output_list) / len(output_list)}
                    if cascade_cfg['forward_default_args'] is not None: 
                        forward_args.update(cascade_cfg['forward_default_args'])
                output_list = self.auginference(
                    segmentor=segmentor,
                    images=image,
                    inference_cfg=inference_cfg,
                    num_classes=cfg.SEGMENTOR_CFG['num_classes'],
                    FloatTensor=FloatTensor,
                    align_corners=segmentor.align_corners,
                    forward_args=forward_args,
                )
            output_list = [
                F.interpolate(output, size=(heights, widths), mode='bilinear', align_corners=segmentor.align_corners) for output in output_list
            ]
            output = sum(output_list) / len(output_list)
            pred = (torch.argmax(output[0], dim=0)).cpu().numpy().astype(np.int32)
            mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for clsid, color in enumerate(palette):
                mask[pred == clsid, :] = np.array(color)[::-1]
            
            # image = image * 0.5 + mask * 0.5
            mask = mask.astype(np.uint8)
            if cmd_args.outputfile:
                # FF++
                # output_name = imageids[0].split('.')[0] + '_' + cmd_args.modeltype
                # cv2.imwrite(os.path.join(cmd_args.outputfile, output_name + '.png'), mask)
                # DEFATO
                output_name = imageids[0] + '_' + cmd_args.modeltype
                cv2.imwrite(os.path.join(cmd_args.outputfile, output_name + '.png'), mask)
                # image_ = image.cpu().numpy().astype(np.int32)
                # cv2.imwrite(os.path.join(cmd_args.outputfile, imageids[0] + '_' + '.png'), image_)
            else:
                # cv2.imwrite(os.path.join(cfg.COMMON_CFG['work_dir'], imagepath.split('/')[-1].split('.')[0] + '.png'), image)
                print('select output path')
    '''inference with augmentations'''
    def auginference(self, segmentor, images, inference_cfg, num_classes, FloatTensor, align_corners, forward_args=None):
        infer_tricks, outputs_list = inference_cfg['tricks'], []
        for scale_factor in infer_tricks['multiscale']:
            images_scale = F.interpolate(images, scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)
            outputs = self.inference(
                segmentor=segmentor, 
                images=images_scale.type(FloatTensor), 
                inference_cfg=inference_cfg, 
                num_classes=num_classes, 
                forward_args=forward_args,
            ).cpu()
            outputs_list.append(outputs)
            if infer_tricks['flip']:
                images_flip = torch.from_numpy(np.flip(images_scale.cpu().numpy(), axis=3).copy())
                outputs_flip = self.inference(
                    segmentor=segmentor, 
                    images=images_flip.type(FloatTensor), 
                    inference_cfg=inference_cfg, 
                    num_classes=num_classes, 
                    forward_args=forward_args,
                )
                fix_ann_pairs = inference_cfg.get('fix_ann_pairs', None)
                if fix_ann_pairs is None:
                    for aug_opt in self.cfg.DATASET_CFG['train']['aug_opts']:
                        if 'RandomFlip' in aug_opt: 
                            fix_ann_pairs = aug_opt[-1].get('fix_ann_pairs', None)
                if fix_ann_pairs is not None:
                    outputs_flip_clone = outputs_flip.data.clone()
                    for (pair_a, pair_b) in fix_ann_pairs:
                        outputs_flip[:, pair_a, :, :] = outputs_flip_clone[:, pair_b, :, :]
                        outputs_flip[:, pair_b, :, :] = outputs_flip_clone[:, pair_a, :, :]
                outputs_flip = torch.from_numpy(np.flip(outputs_flip.cpu().numpy(), axis=3).copy()).type_as(outputs)
                outputs_list.append(outputs_flip)
        return outputs_list
    '''inference'''
    def inference(self, segmentor, images, inference_cfg, num_classes, forward_args=None):
        assert inference_cfg['mode'] in ['whole', 'slide']
        use_probs_before_resize = inference_cfg['tricks']['use_probs_before_resize']
        if inference_cfg['mode'] == 'whole':
            if forward_args is None:
                _, outputs = segmentor(images)
            else:
                _, outputs = segmentor(images, **forward_args)
            if use_probs_before_resize: 
                outputs = F.softmax(outputs, dim=1)
        else:
            align_corners = segmentor.align_corners
            opts = inference_cfg['opts']
            stride_h, stride_w = opts['stride']
            cropsize_h, cropsize_w = opts['cropsize']
            batch_size, _, image_h, image_w = images.size()
            num_grids_h = max(image_h - cropsize_h + stride_h - 1, 0) // stride_h + 1
            num_grids_w = max(image_w - cropsize_w + stride_w - 1, 0) // stride_w + 1
            outputs = images.new_zeros((batch_size, num_classes, image_h, image_w))
            count_mat = images.new_zeros((batch_size, 1, image_h, image_w))
            for h_idx in range(num_grids_h):
                for w_idx in range(num_grids_w):
                    x1, y1 = w_idx * stride_w, h_idx * stride_h
                    x2, y2 = min(x1 + cropsize_w, image_w), min(y1 + cropsize_h, image_h)
                    x1, y1 = max(x2 - cropsize_w, 0), max(y2 - cropsize_h, 0)
                    crop_images = images[:, :, y1:y2, x1:x2]
                    if forward_args is None:
                        outputs_crop = segmentor(crop_images)
                    else:
                        outputs_crop = segmentor(crop_images, **forward_args)
                    outputs_crop = F.interpolate(outputs_crop, size=crop_images.size()[2:], mode='bilinear', align_corners=align_corners)
                    if use_probs_before_resize: 
                        outputs_crop = F.softmax(outputs_crop, dim=1)
                    outputs += F.pad(outputs_crop, (int(x1), int(outputs.shape[3] - x2), int(y1), int(outputs.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            outputs = outputs / count_mat
        return outputs


'''debug'''
if __name__ == '__main__':
    with torch.no_grad():
        client = Demo()
        client.start()