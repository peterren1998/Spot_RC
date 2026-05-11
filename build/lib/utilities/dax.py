import numpy as np
from .correction import correct_illumination, correct_hotpixels, correct_chromatic_aberration, correct_bleedthrough
from .alignment import correct_image3D_by_microscope_param

class Dax_Processor():
    """Class to process dax images"""

    def __init__(self,
                 imageFileName,
                 channels,
                 imageSize,
                 correction_dict,
                 microscope_params = None,
                 verbose=False
                 ):
        """
        imageFileName: dax file name
        channels: color channels to load (in that order)
        imageSize: sizes [z, x, y] 
        correction_dict: the correction dictionary is constructed as follows:
            Keys: correction type
            The corresponding values are:
                'illumination': {channel: file_path_to_npy_file}
                'bleedthrough': {path: path to npy file; channel: list of channels}
                'hotpixel': True or False
                'chromatic': {channel: path_tochromatic_npy_file}
        """
        self.dax_filename = imageFileName
        if isinstance(channels, list):
            self.channels = [str(ch) for ch in channels]
        elif isinstance(channels, str) or isinstance(channels, int):
            self.channels = [str(channels)]
        else:
            raise ValueError('Need to input channels')
        self.image_size = imageSize
        self.correction_dict = correction_dict
        self.microscope_params = microscope_params
        self.verbose = verbose
    
    def load_image(self):
        ### load image
        raw_image = np.fromfile(self.dax_filename, dtype='uint16', count = -1)
        # calculate the number of frames
        num_frames = self.image_size[0] * len(self.channels)
        # reshape the image files
        raw_image = np.reshape(raw_image, [num_frames, self.image_size[1], self.image_size[2]])
        ### split image
        _ch_inds = [self.channels.index(_ch) for _ch in self.channels]
        num_colors = len(self.channels)
        _ch_starts = [(_i) %num_colors for _i in _ch_inds]
        splitted_ims = [raw_image[_s:_s+self.image_size[0]*num_colors:num_colors].copy() for _s in _ch_starts]
        ### save attributes
        for _ch, _im in zip(self.channels, splitted_ims):
            setattr(self, f"im_{_ch}", _im)
        return
    
    def correct_image(self, sel_channels=None):
        # correct images based on the correction dictionary
        if sel_channels is None:
            # correct all channels
            sel_channels = self.channels
        elif isinstance(sel_channels, list):
            sel_channels = [str(ch) for ch in sel_channels]
        elif isinstance(sel_channels, str) or isinstance(sel_channels, int):
            sel_channels = [str(sel_channels)]
        else:
            raise ValueError('Value is wrong for selected channels')
    
        # correct hot pixel
        if 'hotpixel' in self.correction_dict.keys():
            for ch in sel_channels:
                im = getattr(self, f'im_{ch}')
                corrected_im = correct_hotpixels(im)
                setattr(self, f'im_{ch}', corrected_im)
                if self.verbose:
                    print(f'-----Finished hot pixel correction for channel {ch}')

        # correct illumination
        if 'illumination' in self.correction_dict.keys():
            illumination_dict = self.correction_dict['illumination']
            for ch in sel_channels:
                if ch in illumination_dict.keys():
                    im = getattr(self, f'im_{ch}')
                    corrected_im = correct_illumination(im, illumination_dict[ch])
                    setattr(self, f'im_{ch}', corrected_im)
                    if self.verbose:
                        print(f'-----Finished illumination correction for channel {ch}')
        
        # bleed through correction
        if 'bleedthrough' in self.correction_dict.keys():
            corrected_channels = self.correction_dict['bleedthrough']['channel']
            bleedthrough_correction_file = self.correction_dict['bleedthrough']['path']
            if all(element in sel_channels for element in corrected_channels):
                # generate a list of images to correct
                ims = []
                for ch in corrected_channels:
                    ims.append(getattr(self, f'im_{ch}'))
                corrected_ims = correct_bleedthrough(ims, bleedthrough_correction_file)
                # release RAM
                del ims
                # update
                for i, ch in enumerate(corrected_channels):
                    setattr(self, f'im_{ch}', corrected_ims[i])
                if self.verbose:
                    print(f'-----Finished bleedthrough corrections')

        # chromatic correction
        if 'chromatic' in self.correction_dict.keys():
            chromatic_dict = self.correction_dict['chromatic']
            for ch in sel_channels:
                if ch in chromatic_dict.keys():
                    im = getattr(self, f'im_{ch}')
                    warped_im = correct_chromatic_aberration(im, chromatic_dict[ch])
                    setattr(self, f'im_{ch}', warped_im)
                    if self.verbose:
                        print(f'-----Finished chromatic aberration correction for channel {ch}')
        
        # correct by microscope parameters
        if self.microscope_params is not None:
            for ch in sel_channels:
                _correct_image = correct_image3D_by_microscope_param(getattr(self, f'im_{ch}'), self.microscope_params)
                setattr(self, f"im_{ch}", _correct_image)
                print(f'-----Finished microscope correction for channel {ch}')
        
        return