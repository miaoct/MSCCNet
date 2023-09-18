'''
Function:
    Implementation of Multi-Spectral Class Center
Author:
    Changtao Miao
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones import BuildActivation, BuildNormalization, constructnormcfg
from .multi_spectral import MultiSpectralDCTLayer


class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)
    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


'''Multi-Spectral Class Center'''
class MSCC(nn.Module):
    def __init__(self, num_classes, feats_channels, out_channels, dropout,
                height=64, width=64, k=8, norm_cfg=None, act_cfg=None):
        super(MSCC, self).__init__()
        # set attributes
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.out_channels = out_channels
        self.num = k * k

        self.frequency = MultiSpectralDCTLayer(height, width, feats_channels)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )

        self.decoder_map = nn.Sequential(
            nn.Conv2d(feats_channels // self.num, feats_channels // self.num, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=feats_channels // self.num, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(dropout),
            nn.Conv2d(feats_channels // self.num, num_classes, kernel_size=1, stride=1, padding=0),
        )

        self.fuse = nn.Conv2d(self.num, 1, kernel_size=1)
        self.gcn = GCN(self.num, feats_channels // self.num)
       
    '''forward'''
    def forward(self, x):
        residual = x # [B, C, H, W]
        B, C, H, W = x.size()
        
        # compute the high-low frequency prediction map
        fres = self.frequency(x) # [B, C, H, W]

        # split features
        x = channel_split(x, num=self.num) # [B, n, H, W, c]   C=n*c
        fres = channel_split(fres, num=self.num) # [B, n, H, W, c]
        
        n = x.shape[1]
        c = x.shape[-1]

        # compute prediction map
        fres_ = fres.view(B*n, H, W, c).permute(0,3,1,2).contiguous() # [B*n, c, H, W]
        fres_map = self.decoder_map(fres_) # [B*n, K, H, W]
        
        # output map
        fres_map_output = self.fuse(fres_map.view(B, n, self.num_classes, H*W)) # [B, 1, K, H*W]
        fres_map_output = fres_map_output.squeeze(1).view(B, self.num_classes, H, W)

        # similarity
        x = x.view(B, n, H*W, c) # [B, n, H*W, c]
        fres = fres.view(B, n, H*W, c) # [B, n, H*W, c]
        fres_map = fres_map.view(B, n, self.num_classes, H*W) # [B, n, K, H*W]
        local_fre = torch.matmul(fres_map, fres) # [B, n, K, c]

        local_fre = self.gcn(local_fre) # [B, n, K, c]
    
        weight_map = torch.matmul(x, local_fre.transpose(2, 3)) # [B, n, H*W, K]
        weight_map = F.softmax(weight_map, dim=-1)
        out = torch.matmul(weight_map, local_fre) # [B, n, H*W, c]

        out = out.view(B, n, H, W, c)
        out = channel_recover(out, num=self.num)
        out = self.bottleneck(torch.cat([residual, out], dim=1))
        
        return out, fres_map_output


def channel_split(input, num=2):
    """
    b c h w -> b num h w c/num
    """
    B, C, H, W = input.size()
    c = C // num
    out = input.view(B, num, c, H, W)
    out = out.permute(0,1,3,4,2).contiguous() # [B, num, H, W, c]
    return out

def channel_recover(input, num=2):
    """
    b num h w c/num -> b c h w
    """
    B, N, H, W, c = input.size()
    C = num * c
    out = input.permute(0,1,4,2,3).contiguous() # [B, num, c, H, W]
    out = out.view(B, C, H, W) # [B, C, H, W]
    return out
    
