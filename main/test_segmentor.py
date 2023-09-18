'''
Function:
    Test the deepfake localizator
Author:
    Changtao Miao
    refer to Zhenchao Jin 'https://github.com/SegmentationBLWX/sssegmentation'
'''
import sys
sys.path.append('')
import os
import copy
import time
import torch
import pickle
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from configs import BuildConfig
from datasets import BuildDataset
from models import BuildModel
from utils import ( BuildDistributedDataloader, BuildDistributedModel, Logger, checkdir, 
                    setRandomSeed, PixMetric, ImgMetric, loadcheckpoints)

warnings.filterwarnings('ignore')


'''Tester'''
class Tester():
    def __init__(self, cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path):
        # set attribute
        self.cfg = cfg
        self.ngpus_per_node = ngpus_per_node
        self.logger_handle = logger_handle
        self.cmd_args = cmd_args
        self.cfg_file_path = cfg_file_path
        assert torch.cuda.is_available(), 'cuda is not available'
        # init distributed training
        self.cfg.DATALOADER_CFG['test']['batch_size'] = self.cmd_args.nproc_per_node
        dist.init_process_group(backend=self.cfg.SEGMENTOR_CFG.get('backend', 'nccl'))
        # output path
        self.output_path = os.path.join(cfg.COMMON_CFG['output_dir'], cfg.COMMON_CFG['work_dir'])
    '''start tester'''
    def start(self, all_preds, all_gts, all_imgpreds, all_imggts):
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        rank_id = cmd_args.local_rank
        # build dataset and dataloader
        dataset = BuildDataset(mode='TEST', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.DATALOADER_CFG)
        batch_size, num_workers = dataloader_cfg['test']['batch_size'], dataloader_cfg['test']['num_workers']
        batch_size_per_node = batch_size // ngpus_per_node
        num_workers_per_node = num_workers // ngpus_per_node
        dataloader_cfg['test'].update({'batch_size': batch_size_per_node, 'num_workers': num_workers_per_node})
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['test'])
        # build segmentor
        cfg.SEGMENTOR_CFG['backbone']['pretrained'] = False
        segmentor = BuildModel(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='TEST')
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        # load checkpoints
        if cmd_args.evalmode == 'online':
            checkpointspath = os.path.join(self.output_path, 'epoch_best.pth')
        elif cmd_args.evalmode == 'offline':
            checkpointspath = cmd_args.checkpointspath
        else:
            logger_handle.info('testing mode error! plaese select online or offline')
        checkpoints = loadcheckpoints(checkpointspath, logger_handle=logger_handle, cmd_args=cmd_args)
        try:
            segmentor.load_state_dict(checkpoints['model'])
        except Exception as e:
            logger_handle.warning(str(e) + '\n' + 'Try to load checkpoints by using strict=False')
            segmentor.load_state_dict(checkpoints['model'], strict=False)
        # parallel
        segmentor = BuildDistributedModel(segmentor, {'device_ids': [cmd_args.local_rank]})
        # print information
        if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
            logger_handle.info(f'Config file path: {cfg_file_path}')
            logger_handle.info(f'DATASET_CFG: \n{cfg.DATASET_CFG}')
            logger_handle.info(f'DATALOADER_CFG: \n{cfg.DATALOADER_CFG}')
            logger_handle.info(f'OPTIMIZER_CFG: \n{cfg.OPTIMIZER_CFG}')
            logger_handle.info(f'SCHEDULER_CFG: \n{cfg.SCHEDULER_CFG}')
            logger_handle.info(f'LOSSES_CFG: \n{cfg.LOSSES_CFG}')
            logger_handle.info(f'SEGMENTOR_CFG: \n{cfg.SEGMENTOR_CFG}')
            logger_handle.info(f'INFERENCE_CFG: \n{cfg.INFERENCE_CFG}')
            logger_handle.info(f'COMMON_CFG: \n{cfg.COMMON_CFG}')
            logger_handle.info(f'Resume from: {checkpointspath}')
        # set eval
        segmentor.eval()
        # start to test
        FloatTensor = torch.cuda.FloatTensor
        inference_cfg = copy.deepcopy(cfg.INFERENCE_CFG)
        with torch.no_grad():
            dataloader.sampler.set_epoch(0)
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, samples in pbar:
                pbar.set_description('Processing %s/%s in rank %s' % (batch_idx+1, len(dataloader), rank_id))
                imageids, images, widths, heights, gts, imggts = samples['id'], samples['image'], samples['width'], samples['height'], samples['groundtruth'], samples['imagelabel']
                infer_tricks, align_corners = inference_cfg['tricks'], segmentor.module.align_corners
                cascade_cfg = infer_tricks.get('cascade', {'key_for_pre_output': 'memory_gather_logits', 'times': 1, 'forward_default_args': None})
                for idx in range(cascade_cfg['times']):
                    forward_args = None
                    if idx > 0: 
                        maskoutputs_list = [
                            F.interpolate(maskoutputs, size=maskoutputs_list[-1].shape[2:], mode='bilinear', align_corners=align_corners) for maskoutputs in maskoutputs_list
                        ]
                        forward_args = {cascade_cfg['key_for_pre_output']: sum(maskoutputs_list) / len(maskoutputs_list)}
                        if cascade_cfg['forward_default_args'] is not None: 
                            forward_args.update(cascade_cfg['forward_default_args'])
                    imgpreds_list, maskoutputs_list = self.auginference(
                                                    segmentor=segmentor,
                                                    images=images,
                                                    inference_cfg=inference_cfg,
                                                    num_classes=dataset.num_classes,
                                                    FloatTensor=FloatTensor,
                                                    align_corners=align_corners,
                                                    forward_args=forward_args,
                    )
                for idx in range(len(maskoutputs_list[0])):
                    maskoutput = [
                        F.interpolate(maskoutputs[idx: idx+1], size=(heights[idx], widths[idx]), mode='bilinear', align_corners=align_corners) for maskoutputs in maskoutputs_list
                    ]
                    maskoutput = sum(maskoutput) / len(maskoutput)
                    # maskpred = (torch.argmax(maskoutput[0], dim=0)).cpu().numpy().astype(np.int64)
                    maskpred = (torch.argmax(maskoutput, 1)).squeeze(0).cpu().numpy().astype(np.int64)
                    all_preds.append([imageids[idx], maskpred])
                    gt = gts[idx].cpu().numpy().astype(np.int64)
                    gt[gt >= dataset.num_classes] = -1
                    all_gts.append(gt)
                    # image-level
                    imgpred = [
                        F.softmax(imgpreds[idx:idx+1], dim=1).data.cpu().numpy()[:, 1].astype(np.float64) for imgpreds in imgpreds_list
                    ]
                    imgpred = sum(imgpred) / len(imgpred)
                    all_imgpreds.append([imageids[idx], imgpred])
                    imggt = imggts[idx].cpu().numpy().astype(np.float64)
                    all_imggts.append(imggt)
    '''inference with augmentations'''
    def auginference(self, segmentor, images, inference_cfg, num_classes, FloatTensor, align_corners, forward_args=None):
        infer_tricks, maskoutputs_list, imgpreds_list = inference_cfg['tricks'], [], []
        for scale_factor in infer_tricks['multiscale']:
            images_scale = F.interpolate(images, scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)
            imgpreds, maskoutputs = self.inference(
                segmentor=segmentor, 
                images=images_scale.type(FloatTensor), 
                inference_cfg=inference_cfg, 
                num_classes=num_classes, 
                forward_args=forward_args,
            )
            maskoutputs_list.append(maskoutputs.cpu())
            imgpreds_list.append(imgpreds.cpu())

            if infer_tricks['flip']:
                images_flip = torch.from_numpy(np.flip(images_scale.cpu().numpy(), axis=3).copy())
                imgpreds_flip, maskoutputs_flip = self.inference(
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
                    maskoutputs_flip_clone = maskoutputs_flip.data.clone()
                    for (pair_a, pair_b) in fix_ann_pairs:
                        maskoutputs_flip[:, pair_a, :, :] = maskoutputs_flip_clone[:, pair_b, :, :]
                        maskoutputs_flip[:, pair_b, :, :] = maskoutputs_flip_clone[:, pair_a, :, :]
                outputs_flip = torch.from_numpy(np.flip(maskoutputs_flip.cpu().numpy(), axis=3).copy()).type_as(maskoutputs)
                maskoutputs_list.append(outputs_flip)
                imgpreds_list.append(imgpreds_flip)
        return imgpreds_list, maskoutputs_list
    '''inference'''
    def inference(self, segmentor, images, inference_cfg, num_classes, forward_args=None):
        assert inference_cfg['mode'] in ['whole', 'slide']
        use_probs_before_resize = inference_cfg['tricks']['use_probs_before_resize']
        if inference_cfg['mode'] == 'whole':
            if forward_args is None:
                imgpreds, maskoutputs = segmentor(images)
            else:
                imgpreds, maskoutputs = segmentor(images, **forward_args)
            if use_probs_before_resize:
                maskoutputs = F.softmax(maskoutputs, dim=1)
        else:
            align_corners = segmentor.module.align_corners
            FloatTensor = torch.cuda.FloatTensor
            opts = inference_cfg['opts']
            stride_h, stride_w = opts['stride']
            cropsize_h, cropsize_w = opts['cropsize']
            batch_size, _, image_h, image_w = images.size()
            num_grids_h = max(image_h - cropsize_h + stride_h - 1, 0) // stride_h + 1
            num_grids_w = max(image_w - cropsize_w + stride_w - 1, 0) // stride_w + 1
            maskoutputs = images.new_zeros((batch_size, num_classes, image_h, image_w))
            count_mat = images.new_zeros((batch_size, 1, image_h, image_w))
            imgpreds = torch.zeros((batch_size, num_classes)).type(FloatTensor)
            count = torch.zeros((batch_size, 1)).type(FloatTensor)
            for h_idx in range(num_grids_h):
                for w_idx in range(num_grids_w):
                    x1, y1 = w_idx * stride_w, h_idx * stride_h
                    x2, y2 = min(x1 + cropsize_w, image_w), min(y1 + cropsize_h, image_h)
                    x1, y1 = max(x2 - cropsize_w, 0), max(y2 - cropsize_h, 0)
                    crop_images = images[:, :, y1:y2, x1:x2]
                    if forward_args is None:
                        imgpreds_crop, maskoutputs_crop = segmentor(crop_images)
                    else:
                        imgpreds_crop, maskoutputs_crop = segmentor(crop_images, **forward_args)
                    maskoutputs_crop = F.interpolate(maskoutputs_crop, size=crop_images.size()[2:], mode='bilinear', align_corners=align_corners)
                    if use_probs_before_resize: 
                        maskoutputs_crop = F.softmax(maskoutputs_crop, dim=1)
                    maskoutputs += F.pad(maskoutputs_crop, (int(x1), int(maskoutputs.shape[3] - x2), int(y1), int(maskoutputs.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
                    imgpreds += imgpreds_crop
                    count[:, :] += 1
            assert (count_mat == 0).sum() == 0 or (count == 0).sum() == 0
            maskoutputs = maskoutputs / count_mat
            imgpreds = imgpreds / count
        return imgpreds, maskoutputs


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed testing', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--evalmode', dest='evalmode', help='evaluate mode, support online and offline', default='online', type=str)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to resume from', type=str, required=False)
    parser.add_argument('--random_seed', dest='random_seed', help='set random seed', default=866, type=int)
    args = parser.parse_args()
    return args

'''main'''
def main():
    # parse arguments
    args = parseArgs()
    # seed
    setRandomSeed(args.random_seed)
    # build config
    cfg, cfg_file_path = BuildConfig(args.cfgfilepath)
    # check work dir
    checkdir(os.path.join(cfg.COMMON_CFG['output_dir'], cfg.COMMON_CFG['work_dir']))
    # initialize logger_handle
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_test.log".format(time_str, cfg.COMMON_CFG['logfilepath'])
    logger_handle = Logger(os.path.join(cfg.COMMON_CFG['output_dir'], cfg.COMMON_CFG['work_dir'], log_name))
    # number of gpus, for distribued testing, only support a process for a GPU
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node != args.nproc_per_node:
        if (args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
            logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default')
        ngpus_per_node = args.nproc_per_node
    # instanced Tester
    all_preds, all_gts, all_imgpreds, all_imggts = [], [], [], []
    client = Tester(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)
    client.start(all_preds, all_gts, all_imgpreds, all_imggts)
    img_metric = ImgMetric(num_class=cfg.SEGMENTOR_CFG['num_classes'])
    pix_metric = PixMetric(num_class=cfg.SEGMENTOR_CFG['num_classes'])
    # save results and evaluate
    rank_id = args.local_rank
    output_path = os.path.join(cfg.COMMON_CFG['output_dir'], cfg.COMMON_CFG['work_dir'])
    filename = cfg.COMMON_CFG['resultsavepath'].split('.')[0] + f'_{rank_id}'+ '_test.' + cfg.COMMON_CFG['resultsavepath'].split('.')[-1]
    with open(os.path.join(output_path, filename), 'wb') as fp:
        pickle.dump([all_preds, all_gts, all_imgpreds, all_imggts], fp)
    rank = torch.tensor([rank_id], device='cuda')
    rank_list = [rank.clone() for _ in range(ngpus_per_node)]
    dist.all_gather(rank_list, rank)
    logger_handle.info('Rank %s finished' % int(rank.item()))
    if rank_id == 0:
        all_preds_gather, all_gts_gather, all_imgpreds_gather, all_imggts_gather = [], [], [], []
        for rank in rank_list:
            rank = str(int(rank.item()))
            filename = cfg.COMMON_CFG['resultsavepath'].split('.')[0] + f'_{rank}'+ '_test.' + cfg.COMMON_CFG['resultsavepath'].split('.')[-1]
            fp = open(os.path.join(output_path, filename), 'rb')
            all_preds, all_gts, all_imgpreds, all_imggts = pickle.load(fp)
            all_preds_gather += all_preds
            all_gts_gather += all_gts
            all_imgpreds_gather += all_imgpreds
            all_imggts_gather += all_imggts
        all_preds_filtered, all_gts_filtered, all_imgpreds_filtered, all_imggts_filtered, all_ids = [], [], [], [], []
        for idx, pred in enumerate(all_preds_gather):
            if pred[0] in all_ids: 
                continue
            all_ids.append(pred[0])
            all_preds_filtered.append(pred[1])
            all_gts_filtered.append(all_gts_gather[idx])
            all_imgpreds_filtered.append(all_imgpreds_gather[idx][1])
            all_imggts_filtered.append(all_imggts_gather[idx])
        # calculate the metric
        logger_handle.info('All Finished, all_preds: %s' % (len(all_preds_filtered)))
        img_metric.update(all_imgpreds_filtered, all_imggts_filtered)
        img_acc, img_auc, img_f1, img_eer = img_metric.total_score()
        logger_handle.info("img_acc: {:.4f} || img_auc: {:.4f} || img_f1: {:.4f} || img_eer: {:.4f}".format(
                            img_acc, img_auc, img_f1, img_eer))
        pix_metric.update(all_preds_filtered, all_gts_filtered)
        pix_acc, pix_mIoU, pix_f1, pix_mcc = pix_metric.total_score()
        logger_handle.info("pix_acc: {:.4f} || pix_f1: {:.4f} || pix_mIoU: {:.4f} || pix_MCC: {:.4f}".format(
                            pix_acc, pix_f1, pix_mIoU, pix_mcc))
        # remove .pkl file
        for rank in rank_list:
            rank = str(int(rank.item()))
            filename = cfg.COMMON_CFG['resultsavepath'].split('.')[0] + f'_{rank}'+ '_test.'+ cfg.COMMON_CFG['resultsavepath'].split('.')[-1]
            if os.path.exists(os.path.join(output_path, filename)):
                os.remove(os.path.join(output_path, filename))
                print('Remove result.pkl')


'''debug'''
if __name__ == '__main__':
    with torch.no_grad():
        main()