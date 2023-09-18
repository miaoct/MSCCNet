'''
Function:
    Implementation of Multi-Spectral Class Center
Author:
    Changtao Miao
    refer to 'Fcanet: Frequency channel attention networks' ICCV 2021
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
# from einops import rearrange

def get_freq_indices(k=8):
    uv_xy = []
    assert k % 2 == 0, "k must be even number" 

    for x in range(k):
        for y in range(k):
            uv_xy.append((x,y))

    return uv_xy


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, channel, k=8):
        super(MultiSpectralDCTLayer, self).__init__()
        
        uv_xy = get_freq_indices(k=k)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, uv_xy, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        return x

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) # freq=u_x,v_y   pos=t_x,t_y    POS=tile_size_x(H),tile_size_y(W)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, uv_xy, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(uv_xy)

        for i, (u_x, v_y) in enumerate(uv_xy):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter