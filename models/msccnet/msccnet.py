'''
Function:
    Implementation of Multi-level Features Fusion and Multi-Spectral Class Center Network
Author:
    Changtao Miao
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from ..backbones import BuildActivation, BuildNormalization, constructnormcfg

from .mscc import MSCC

'''MSCCNet'''
class MSCCNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(MSCCNet, self).__init__(cfg, mode)
        norm_cfg, act_cfg, head_cfg = self.norm_cfg, self.act_cfg, cfg['head']

        # build bottleneck
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels']*2, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels']*4, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build mfc module
        self.mscc = MSCC(
            num_classes=cfg['num_classes'], 
            feats_channels=head_cfg['feats_channels'], 
            out_channels=head_cfg['out_channels'],
            dropout=head_cfg['dropout'],
            height=head_cfg['spectral_height'],
            width=head_cfg['spectral_width'],
            k=head_cfg['spectral_k'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        # build fusion bottleneck
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels']*3, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(head_cfg['out_channels'], head_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['out_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )

        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net','bottleneck1','bottleneck2','bottleneck3', 'mscc', 'bottleneck_fusion', 'decoder', 'auxiliary_decoder']
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to classifier
        predictions_img = self.classifer(backbone_outputs[-1])
        # multi-level features fusion
        x1 = backbone_outputs[1]
        x1 = self.bottleneck1(x1)
        x2 = backbone_outputs[2]
        x2 = self.bottleneck2(x2)
        x3 = backbone_outputs[3]
        x3 = self.bottleneck3(x3)
        x123 = self.bottleneck_fusion(torch.cat((x1, x2, x3), dim=1))
        # feed to decoder
        x123, fres_map_output = self.mscc(x123)
        predictions_mask = self.decoder(x123)
        # forward according to the mode
        if self.mode == 'TRAIN':
            outputs_dict = self.forwardtrain(
                predictions_image = predictions_img,
                predictions_mask=predictions_mask,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
                compute_loss=False,
            )
            preds_mask = outputs_dict.pop('pix_cls')
            preds_map = F.interpolate(fres_map_output, size=img_size, mode='bilinear', align_corners=self.align_corners)
            outputs_dict.update({'pix_cls_mask': preds_mask, 'pix_cls_map': preds_map})
            loss, losses_log_dict = self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
            return loss, losses_log_dict
        return predictions_img, predictions_mask
