import numpy as np
import pickle
from scipy.ndimage.interpolation import map_coordinates


def correct_illumination(im, file_path, rescale=True):
    illumination_correction = np.load(file_path, allow_pickle=True)
    _dtype = im.dtype
    _min,_max = np.iinfo(_dtype).min, np.iinfo(_dtype).max
    # apply corr
    im = im.astype(np.float32) / illumination_correction[np.newaxis,:]
    if rescale: 
        im = (im - np.min(im)) / (np.max(im) - np.min(im)) * _max + _min
    im = np.clip(im, a_min=_min, a_max=_max)
    return im.astype(_dtype)

def correct_hotpixels(im, hot_pix_th=0.50, hot_th=4, 
                      interpolation_style='nearest'):
    '''Function to remove hot pixels by interpolation in each single layer, from Pu's ImageAnalysis 3'''
    dtype=im.dtype
    # create convolution matrix, ignore boundaries for now
    _conv = (np.roll(im,1,1)+np.roll(im,-1,1)+np.roll(im,1,2)+np.roll(im,-1,2))/4
    # hot pixels must be have signals higher than average of neighboring pixels by hot_th in more than hot_pix_th*total z-stacks
    _hotmat = im > hot_th * _conv
    _hotmat2D = np.sum(_hotmat,0)
    _hotpix_cand = np.where(_hotmat2D > hot_pix_th*np.shape(im)[0])
    # if no hot pixel detected, directly exit
    if len(_hotpix_cand[0]) == 0:
        return im
    # create new image to interpolate the hot pixels with average of neighboring pixels
    _nim = im.copy()
    if interpolation_style == 'nearest':
        for _x, _y in zip(_hotpix_cand[0],_hotpix_cand[1]):
            if _x > 0 and  _y > 0 and _x < im.shape[1]-1 and  _y < im.shape[2]-1:
                _nim[:,_x,_y] = (_nim[:,_x+1,_y]+_nim[:,_x-1,_y]+_nim[:,_x,_y+1]+_nim[:,_x,_y-1])/4
    return _nim.astype(dtype)

def correct_chromatic_aberration(im, chromatic_file_path):
    # load chromatic npy file
    chromatic_npy = np.load(chromatic_file_path, allow_pickle=True)
    _coords = np.meshgrid(np.arange(im.shape[0]), 
                                np.arange(im.shape[1]), 
                                np.arange(im.shape[2]), 
                                )
    # transpose is necessary  
    _coords = np.stack(_coords).transpose((0, 2, 1, 3))
    # warp coordinates
    _coords = _coords + chromatic_npy
    del chromatic_npy
    # map coordinates
    warped_im = map_coordinates(im, 
                                _coords.reshape(_coords.shape[0], -1),
                                mode='nearest').astype(im.dtype)
    warped_im = warped_im.reshape(im.shape)
    del _coords
    return warped_im

def correct_bleedthrough(ims, bleedthrough_file, rescale=True):
    """correct bleedthrough: ims is a list of images. Each image in the list is an image of the corresponding color in the corrected channel"""
    # define im size and number of channels
    im_size = ims[0].shape
    _dtype = ims[0].dtype
    num_im = len(ims)
    # load bleedthrough correction
    bleedthrough_correction = np.load(bleedthrough_file, allow_pickle=True)
    bleedthrough_correction = bleedthrough_correction.reshape(num_im, num_im, im_size[-2], im_size[-1])
    # correct bleedthrough
    corrected_ims = []
    for i in range(num_im):
        _min,_max = np.iinfo(_dtype).min, np.iinfo(_dtype).max
        # init image
        _im = np.zeros(im_size)
        for j in range(num_im):
            _im += ims[j] * bleedthrough_correction[i, j]
        # rescale
        if rescale: # (np.max(_im) > _max or np.min(_im) < _min)
            _im = (_im - np.min(_im)) / (np.max(_im) - np.min(_im)) * _max + _min
        _im = np.clip(_im, a_min=_min, a_max=_max).astype(_dtype)
        corrected_ims.append(_im.copy())
        # release RAM
        del _im
    return corrected_ims