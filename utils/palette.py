'''
Function:
    Color map for visualizing the segmentation mask
Author:
    Zhenchao Jin
'''
import os
import numpy as np


'''FF++'''
ff_palette = [(0, 0, 0), (255, 255, 255)]
assert len(ff_palette) == 2

'''DEFACTO'''
defacto_palette = [(0, 0, 0), (255, 255, 255)]
assert len(defacto_palette) == 2


'''FFIW-10K'''
ffiw_palette = [(0, 0, 0), (255, 0, 0)]
assert len(ffiw_palette) == 2


'''collect the suppoted palettes'''
supported_palettes = {
    'Pixel_FF++_test': ff_palette,
    'DEFACTO': defacto_palette,
    'supervisely': ffiw_palette,
}


'''GeneratePalette'''
def GeneratePalette(num_classes):
    palette = [0] * (num_classes * 3)
    for j in range(0, num_classes):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    
    palette = np.array(palette).reshape(-1, 3)
    palette = palette.tolist()
    return palette


'''BuildPalette'''
def BuildPalette(dataset_type, num_classes=2, logger_handle=None):
    if dataset_type not in supported_palettes:
        logger_handle.warning('Unsupport dataset type %s, try to generate the palette according to the number of classes' % dataset_type)
        return GeneratePalette(num_classes)
    return supported_palettes[dataset_type]