'''
Function:
    Train the deepfake localizator
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
from utils import (BuildOptimizer, BuildScheduler, BuildDistributedDataloader, BuildDistributedModel,
                    Logger, checkdir, setRandomSeed, PixMetric, ImgMetric, loadcheckpoints, savecheckpoints)

warnings.filterwarnings('ignore')


'''Trainer'''
class Trainer():
    def __init__(self, cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path):
        # set attribute
        self.cfg = cfg
        self.ngpus_per_node = ngpus_per_node
        self.logger_handle = logger_handle
        self.cmd_args = cmd_args
        self.cfg_file_path = cfg_file_path
        assert torch.cuda.is_available(), 'cuda is not available'
        # init distributed training
        dist.init_process_group(backend=self.cfg.SEGMENTOR_CFG.get('backend', 'nccl'))
        # init metric
        self.img_metric = ImgMetric(num_class=cfg.SEGMENTOR_CFG['num_classes'])
        self.pix_metric = PixMetric(num_class=cfg.SEGMENTOR_CFG['num_classes'])
        # output path
        self.output_path = os.path.join(cfg.COMMON_CFG['output_dir'], cfg.COMMON_CFG['work_dir'])
    '''start trainer'''
    def start(self):
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        # build dataset and dataloader
        dataset = BuildDataset(mode='TRAIN', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.DATALOADER_CFG)
        batch_size, num_workers = dataloader_cfg['train']['batch_size'], dataloader_cfg['train']['num_workers']
        batch_size_per_node = batch_size // ngpus_per_node
        num_workers_per_node = num_workers // ngpus_per_node
        dataloader_cfg['train'].update({'batch_size': batch_size_per_node, 'num_workers': num_workers_per_node})
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['train'])
        # build segmentor
        segmentor = BuildModel(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), mode='TRAIN')
        torch.cuda.set_device(cmd_args.local_rank)
        segmentor.cuda(cmd_args.local_rank)
        torch.backends.cudnn.benchmark = cfg.SEGMENTOR_CFG['benchmark']
        # build optimizer
        optimizer_cfg = copy.deepcopy(cfg.OPTIMIZER_CFG)
        optimizer = BuildOptimizer(segmentor, optimizer_cfg)
        # build scheduler
        scheduler_cfg = copy.deepcopy(cfg.SCHEDULER_CFG)
        scheduler_cfg.update({
            'lr': cfg.OPTIMIZER_CFG['lr'],
            'iters_per_epoch': len(dataloader),
            'params_rules': cfg.OPTIMIZER_CFG['params_rules'],
        })
        scheduler = BuildScheduler(optimizer=optimizer, scheduler_cfg=scheduler_cfg)
        start_epoch, end_epoch = 1, scheduler_cfg['max_epochs']
        # load checkpoints
        if cmd_args.checkpointspath and os.path.exists(cmd_args.checkpointspath):
            checkpoints = loadcheckpoints(cmd_args.checkpointspath, logger_handle=logger_handle, cmd_args=cmd_args)
            try:
                segmentor.load_state_dict(checkpoints['model'])
            except Exception as e:
                logger_handle.warning(str(e) + '\n' + 'Try to load checkpoints by using strict=False')
                segmentor.load_state_dict(checkpoints['model'], strict=False)
            if 'optimizer' in checkpoints: 
                optimizer.load_state_dict(checkpoints['optimizer'])
            if 'cur_epoch' in checkpoints: 
                start_epoch = checkpoints['cur_epoch'] + 1
                scheduler.setstate({'cur_epoch': checkpoints['cur_epoch'], 'cur_iter': checkpoints['cur_iter']})
                assert checkpoints['cur_iter'] == len(dataloader) * checkpoints['cur_epoch']
        else:
            cmd_args.checkpointspath = ''
        # parallel segmentor
        build_dist_model_cfg = self.cfg.SEGMENTOR_CFG.get('build_dist_model_cfg', {})
        build_dist_model_cfg.update({'device_ids': [cmd_args.local_rank]})
        segmentor = BuildDistributedModel(segmentor, build_dist_model_cfg)
        # print config
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
            logger_handle.info(f'Resume from: {cmd_args.checkpointspath}')
            logger_handle.info(f'Load {len(dataset)} data samples from training set')
        # start to train the segmentor
        FloatTensor, losses_log_dict_memory = torch.cuda.FloatTensor, {}
        for epoch in range(start_epoch, end_epoch+1):
            # --set train
            segmentor.train()
            dataloader.sampler.set_epoch(epoch)
            # --train epoch
            for batch_idx, samples in enumerate(dataloader):
                learning_rate = scheduler.updatelr()
                images, targets = samples['image'].type(FloatTensor), {'segmentation': samples['segmentation'].type(FloatTensor), 'label': samples['imagelabel'].type(FloatTensor)}
                optimizer.zero_grad()
                # calculate the losses
                loss, losses_log_dict = segmentor(images, targets, cfg.LOSSES_CFG)
                # loss, losses_log_dict = self.calculatelosses(predictions=outputs_dict, targets=targets, losses_cfg=cfg.LOSSES_CFG)
                for key, value in losses_log_dict.items():
                    if key in losses_log_dict_memory: 
                        losses_log_dict_memory[key].append(value)
                    else: 
                        losses_log_dict_memory[key] = [value]
                loss.backward()
                scheduler.step()
                if (cmd_args.local_rank == 0) and (scheduler.cur_iter % cfg.COMMON_CFG['log_interval_iterations'] == 0):
                    loss_log = ''
                    for key, value in losses_log_dict_memory.items():
                        loss_log += '%s %.4f, ' % (key, sum(value) / len(value))
                    losses_log_dict_memory = dict()
                    logger_handle.info(
                        f'[Epoch]: {epoch}/{end_epoch}, [Batch]: {batch_idx+1}/{len(dataloader)}, [Segmentor]: {cfg.SEGMENTOR_CFG["type"]}-{cfg.SEGMENTOR_CFG["backbone"]["type"]}, '
                        f'[DATASET]: {cfg.DATASET_CFG["type"]}, [LEARNING_RATE]: {learning_rate}\n\t[LOSS]: {loss_log}'
                    )
            scheduler.cur_epoch = epoch
            # --save checkpoints
            if epoch % cfg.COMMON_CFG['save_interval_epochs'] == 0:
                state_dict = scheduler.state()
                state_dict['model'] = segmentor.module.state_dict()
                savepath = os.path.join(self.output_path, 'epoch_%s.pth' % epoch)
                if cmd_args.local_rank == 0:
                    savecheckpoints(state_dict, savepath, logger_handle, cmd_args=cmd_args)
            # remove before checkpoints
            if os.path.exists(os.path.join(self.output_path, 'epoch_%s.pth' % (epoch-2))) and cmd_args.local_rank == 0:
                    os.remove(os.path.join(self.output_path, 'epoch_%s.pth' % (epoch-2)))
                    print('Remove epoch_%s.pth' % (epoch-2))
            # --eval checkpoints
            if (epoch % cfg.COMMON_CFG['eval_interval_epochs'] == 0) or (epoch == end_epoch):
                self.evaluate(segmentor, scheduler)
    '''evaluate'''
    def evaluate(self, segmentor, scheduler):
        cfg, ngpus_per_node, cmd_args, logger_handle = self.cfg, self.ngpus_per_node, self.cmd_args, self.logger_handle
        rank_id = cmd_args.local_rank
        # build dataset and dataloader
        dataset = BuildDataset(mode='VAL', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        dataloader_cfg = copy.deepcopy(cfg.DATALOADER_CFG)
        batch_size, num_workers = ngpus_per_node, dataloader_cfg['val']['num_workers']
        batch_size_per_node = batch_size // ngpus_per_node
        num_workers_per_node = num_workers // ngpus_per_node
        dataloader_cfg['val'].update({'batch_size': batch_size_per_node, 'num_workers': num_workers_per_node})
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['val'])
        # start to eval
        self.img_metric.reset()
        self.pix_metric.reset()
        segmentor.eval()
        segmentor.module.mode = 'VAL'
        inference_cfg, all_preds, all_gts, all_imgpreds, all_imggts = cfg.INFERENCE_CFG, [], [], [], []
        align_corners = segmentor.module.align_corners
        FloatTensor = torch.cuda.FloatTensor
        use_probs_before_resize = inference_cfg['tricks']['use_probs_before_resize']
        assert inference_cfg['mode'] in ['whole', 'slide']
        with torch.no_grad():
            dataloader.sampler.set_epoch(0)
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, samples in pbar:
                pbar.set_description('Processing %s/%s in rank %s' % (batch_idx+1, len(dataloader), rank_id))
                imageids, images, widths, heights, gts, imggts = samples['id'], samples['image'].type(FloatTensor), samples['width'], samples['height'], samples['groundtruth'], samples['imagelabel']
                if inference_cfg['mode'] == 'whole':
                    imgpreds, maskoutputs = segmentor(images)
                    if use_probs_before_resize:
                        maskoutputs = F.softmax(maskoutputs, dim=1)
                else:
                    opts = inference_cfg['opts']
                    stride_h, stride_w = opts['stride']
                    cropsize_h, cropsize_w = opts['cropsize']
                    batch_size, _, image_h, image_w = images.size()
                    num_grids_h = max(image_h - cropsize_h + stride_h - 1, 0) // stride_h + 1
                    num_grids_w = max(image_w - cropsize_w + stride_w - 1, 0) // stride_w + 1
                    maskoutputs = images.new_zeros((batch_size, cfg.SEGMENTOR_CFG['num_classes'], image_h, image_w))
                    count_mat = images.new_zeros((batch_size, 1, image_h, image_w))
                    imgpreds = torch.zeros((batch_size, cfg.SEGMENTOR_CFG['num_classes'])).type(FloatTensor)
                    count = torch.zeros((batch_size, 1)).type(FloatTensor)
                    for h_idx in range(num_grids_h):
                        for w_idx in range(num_grids_w):
                            x1, y1 = w_idx * stride_w, h_idx * stride_h
                            x2, y2 = min(x1 + cropsize_w, image_w), min(y1 + cropsize_h, image_h)
                            x1, y1 = max(x2 - cropsize_w, 0), max(y2 - cropsize_h, 0)
                            crop_images = images[:, :, y1:y2, x1:x2]
                            imgpreds_crop, maskoutputs_crop = segmentor(crop_images)
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
                for idx in range(len(maskoutputs)):
                    # pixel-level
                    maskoutput = F.interpolate(maskoutputs[idx: idx+1], size=(heights[idx], widths[idx]), mode='bilinear', align_corners=align_corners)
                    # maskpred = (torch.argmax(maskoutput[0], dim=0)).cpu().numpy().astype(np.int64)
                    maskpred = (torch.argmax(maskoutput, 1)).squeeze(0).cpu().numpy().astype(np.int64)
                    all_preds.append([imageids[idx], maskpred])
                    gt = gts[idx].cpu().numpy().astype(np.int64)
                    gt[gt >= dataset.num_classes] = -1
                    all_gts.append(gt)
                    # image-level
                    imgpred = F.softmax(imgpreds[idx:idx+1], dim=1).data.cpu().numpy()[:, 1].astype(np.float64)
                    all_imgpreds.append([imageids[idx], imgpred])
                    imggt = imggts[idx].cpu().numpy().astype(np.float64)
                    all_imggts.append(imggt)

        # collect eval results
        filename = cfg.COMMON_CFG['resultsavepath'].split('.')[0] + f'_{rank_id}.' + cfg.COMMON_CFG['resultsavepath'].split('.')[-1]
        with open(os.path.join(self.output_path, filename), 'wb') as fp:
            pickle.dump([all_preds, all_gts, all_imgpreds, all_imggts], fp)
        rank = torch.tensor([rank_id], device='cuda')
        rank_list = [rank.clone() for _ in range(ngpus_per_node)]
        dist.all_gather(rank_list, rank)
        logger_handle.info('Rank %s finished' % int(rank.item()))
        if rank_id == 0:
            all_preds_gather, all_gts_gather, all_imgpreds_gather, all_imggts_gather = [], [], [], []
            for rank in rank_list:
                rank = str(int(rank.item()))
                filename = cfg.COMMON_CFG['resultsavepath'].split('.')[0] + f'_{rank}.' + cfg.COMMON_CFG['resultsavepath'].split('.')[-1]
                fp = open(os.path.join(self.output_path, filename), 'rb')
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
            self.img_metric.update(all_imgpreds_filtered, all_imggts_filtered)
            img_acc, img_auc, img_f1, img_eer = self.img_metric.total_score()
            logger_handle.info("img_acc: {:.4f} || img_auc: {:.4f} || img_f1: {:.4f} || img_eer: {:.4f}".format(
                                img_acc, img_auc, img_f1, img_eer))
            self.pix_metric.update(all_preds_filtered, all_gts_filtered)
            pix_acc, pix_mIoU, pix_f1, pix_mcc =self.pix_metric.total_score()
            logger_handle.info("pix_acc: {:.4f} || pix_f1: {:.4f} || pix_mIoU: {:.4f} || pix_MCC: {:.4f}".format(
                                pix_acc, pix_f1, pix_mIoU, pix_mcc))
            # remove .pkl file
            for rank in rank_list:
                rank = str(int(rank.item()))
                filename = cfg.COMMON_CFG['resultsavepath'].split('.')[0] + f'_{rank}.' + cfg.COMMON_CFG['resultsavepath'].split('.')[-1]
                if os.path.exists(os.path.join(self.output_path, filename)):
                    os.remove(os.path.join(self.output_path, filename))
                    print('Remove result.pkl')
            # --save best checkpoints
            state_dict = scheduler.state()
            state_dict['model'] = segmentor.module.state_dict()
            savepath = os.path.join(self.output_path, 'epoch_best.pth')
            savecheckpoints(state_dict, savepath, logger_handle, cmd_args=cmd_args)
    
        segmentor.train()
        segmentor.module.mode = 'TRAIN'


'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed training', default=0, type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8, type=int)
    parser.add_argument('--cfgfilepath', dest='cfgfilepath', help='config file path you want to use', type=str, required=True)
    parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to resume from', default='', type=str)
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
    log_name = "{}_{}.log".format(time_str, cfg.COMMON_CFG['logfilepath'])
    logger_handle = Logger(os.path.join(cfg.COMMON_CFG['output_dir'], cfg.COMMON_CFG['work_dir'], log_name))
    # number of gpus, for distribued training, only support a process for a GPU
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node != args.nproc_per_node:
        if args.local_rank == 0: 
            logger_handle.warning('ngpus_per_node is not equal to nproc_per_node, force ngpus_per_node = nproc_per_node by default')
        ngpus_per_node = args.nproc_per_node
    # instanced Trainer
    client = Trainer(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)
    client.start()


'''debug'''
if __name__ == '__main__':
    main()