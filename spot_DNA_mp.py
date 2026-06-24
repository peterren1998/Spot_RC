import argparse
import os
import pandas as pd
import numpy as np
import pickle
import h5py

from .utilities import dax, fitting, alignment

conventional_image_prefix = 'Conv_zscan_' # then str(fov).zfill(3)+'.dax'
channels_for_FISH = ['750', '647', '561']
num_pixel_xy = 2048
distance_zxy = [250, 108, 108]

def _format_seed_stats(seed_stats):
    return (
        f"raw={seed_stats['raw_seed_candidates']}, "
        f"edge_kept={seed_stats['after_edge_filter']}, "
        f"hot_kept={seed_stats['after_hot_pixel_filter']}, "
        f"max_trunc={seed_stats['truncated_by_max']}, "
        f"final={seed_stats['final_seed_count']}, "
        f"stopped={seed_stats['stopped_reason']}, "
        f"threshold={seed_stats['threshold_used']}, "
        f"iterations={seed_stats['iterations_used']}"
    )

def load_color_info(color_info_file, round_name, channels_for_FISH=channels_for_FISH):
    color_dict = {}
    df_color = pd.read_csv(color_info_file)
    df_round = df_color[df_color['Hyb']==round_name].copy()
    df_round.reset_index(inplace=True, drop=True)
    for col in df_round.columns:
        if col in channels_for_FISH:
            if not pd.isnull(df_round.loc[0, col]):
                color_dict[col] = df_round.loc[0, col]
    return color_dict


def SpotDNA_mp(data_folders, analysis_folder, correction_file, parameter_file, fov, fiducial_channel='488', dapi_channel='405', num_z=50, ref_round='H0C1', overwrite=False):
    
    ### define image file
    _image_rounds = []
    round_to_folder = {}
    for data_folder in data_folders:
        for _round in os.listdir(data_folder):
            if _round[0]=='H':
                _image_rounds.append(_round)
                round_to_folder[_round] = data_folder
     
    # check reference round
    if ref_round not in _image_rounds:
        raise ValueError(f'Missing reference round in data folder')
    # generate image files, temporary
    image_file_name = conventional_image_prefix+str(fov).zfill(3)+'.dax'
    _image_files = []
    for _round in _image_rounds:
        _image_files.append(os.path.join(round_to_folder[_round], _round, image_file_name))
    # check image files and start processing from the reference round
    image_rounds = [ref_round]
    image_files = [os.path.join(round_to_folder[ref_round], ref_round, image_file_name)]
    for fl, _round in zip(_image_files, _image_rounds):
        if (os.path.exists(fl)) and (_round not in image_rounds):
            image_rounds.append(_round)
            image_files.append(fl)
    print(f'START analyzing fov {fov} in rounds', end=': ', flush=True)
    for _round in image_rounds:
        print(_round, end=', ', flush=True)
    print('\n', flush=True)
    # change the format into array
    image_rounds = np.array(image_rounds)
    image_files = np.array(image_files)
   
    
    # define basic image parameters
    imageSize = [num_z, num_pixel_xy, num_pixel_xy]

    ### define output file
    # identify whether Color_Usage.csv is present
    color_info_file = os.path.join(analysis_folder, 'Color_Usage.csv')
    if not os.path.exists(color_info_file):
        raise ValueError(f'{color_info_file} is missing')
    
    ### define output file
    output_file = os.path.join(analysis_folder, image_file_name.replace('.dax', '.hdf5'))
    ### TO DO: write the overwrite infomation

    # load correction dictionary
    correction_dict = pickle.load(open(correction_file, 'rb'))
    
    ### load reference image and store
    print('-Start loading reference fiducial and DAPI image', flush=True)
    fiducial_channel = 488
    # check whether the fiducial image has already exists
    ref_fiducial_image = None
    if os.path.exists(output_file) and (not overwrite):
        with h5py.File(output_file, 'r+') as hdf_file:
            if 'reference_fiducial_image' in hdf_file:
                ref_fiducial_image = hdf_file['reference_fiducial_image'][:]
                print('---Read reference fiducial image directly from hdf5', flush=True)
    # when ref fiducial image is not loaded
    if ref_fiducial_image is None:
        ref_im_file = image_files[image_rounds==ref_round][0]
        # load reference image
        ref_image_channel = channels_for_FISH + [fiducial_channel, dapi_channel]
        ref_dax = dax.Dax_Processor(ref_im_file, ref_image_channel, imageSize, correction_dict)
        ref_dax.load_image()
        ref_dax.correct_image()
        ref_fiducial_image = getattr(ref_dax, f'im_{fiducial_channel}').copy()
        # Write the reference image to the HDF5 file
        with h5py.File(output_file, 'a') as hdf_file:
            hdf_file.create_dataset('reference_fiducial_image', 
                                    data=getattr(ref_dax, f'im_{fiducial_channel}'))
            hdf_file.create_dataset('DNA_DAPI_image', 
                                    data=getattr(ref_dax, f'im_{dapi_channel}'))
        print('---Finish saving reference fiducial and DAPI image\n', flush=True)
        del ref_dax

    ### load parameters for picking
    parameters = pickle.load(open(parameter_file, 'rb'))

    ### iterate through the image files
    for image_file, round_name in zip(image_files, image_rounds):
        
        # load color usage
        color_usage = load_color_info(color_info_file, round_name)
        # stop if there is no color usage information
        if len(color_usage.keys())==0:
            raise ValueError(f'Missing color usage information for round {round_name}')
        
        ### load and correct image
        print(f'-Start analyzing images for round {round_name}', flush=True)
        if round_name == ref_round:
            # load dapi channel for reference round
            load_channels = channels_for_FISH+[fiducial_channel, dapi_channel]
        else:
            load_channels = channels_for_FISH+[fiducial_channel]
        print(f'---Load image from file {image_file}', flush=True)
        dax_cls = dax.Dax_Processor(image_file, load_channels, imageSize, correction_dict)
        dax_cls.load_image()
        dax_cls.correct_image()
        print(f'---Finish image correction for round {round_name}', flush=True)
        
        ### calculate drift
        # load and correct fiducial images
        fiducial_image = getattr(dax_cls, f'im_{fiducial_channel}')
        # calculate
        if round_name!=ref_round:
            print(f'---Calculate drift for round {round_name}', flush=True)
            drift, drift_flag = alignment.align_image(fiducial_image, ref_fiducial_image)
            print(f'---Drift for round {round_name} calculated with {drift_flag}', flush=True)
        else:
            print(f'---No drift calculation for reference round', flush=True)
            drift, drift_flag = [0,0,0], 'Reference image'

        #### start spot finding
        print(f'---Start spot finding for round {round_name}', flush=True)
        for color, bit in color_usage.items():
            ### check whether the bit information exists
            if overwrite==False:
                with h5py.File(output_file, 'r+') as hdf_file:
                    if bit in hdf_file:
                        bit_info = hdf_file[bit]
                        if ('drift' in bit_info) and ('spots' in bit_info):
                            print(f'---Spot information for bit {bit} already exists.', flush=True)
                            continue

            ### use color to find images: 
            im = getattr(dax_cls, f'im_{color}')
            
            ### generate seed
            seeds, seed_stats = fitting.get_seeds(
                im,
                max_num_seeds=parameters['max_num_seed'],
                th_seed=parameters['seed_threshold'][color],
                min_dynamic_seeds=parameters['min_num_seed'],
                return_stats=True,
            )
            print(
                f"-----{len(seeds)} seeded with th={parameters['seed_threshold'][color]} in channel {color} for round {round_name} "
                f"({_format_seed_stats(seed_stats)})",
                flush=True,
            )
            ### fitting
            fitter = fitting.iter_fit_seed_points(im, seeds.T)    
            # fit
            fitter.firstfit()
            # check
            fitter.repeatfit()
            # get spots
            fitted_spots = np.array(fitter.ps)
            if fitted_spots.ndim != 2 or fitted_spots.shape[0] == 0:
                print(
                    f"-----0 found in channel {color} in round {round_name} (fit=0, nan_removed=0, boundary_removed=0)",
                    flush=True,
                )
                continue
            total_fitted = int(fitted_spots.shape[0])
            keep_nan = np.sum(np.isnan(fitted_spots), axis=1) == 0
            nan_removed = int(total_fitted - np.count_nonzero(keep_nan))
            spots = fitted_spots[keep_nan]
            if spots.shape[0] == 0:
                print(
                    f"-----0 found in channel {color} in round {round_name} (fit={total_fitted}, nan_removed={nan_removed}, boundary_removed=0)",
                    flush=True,
                )
                continue
            # remove all boundary points
            _kept_flags = (spots[:,1:4] > np.zeros(3)).all(1)                 * (spots[:, 1:4] < np.array(np.shape(im))).all(1)
            boundary_removed = int(spots.shape[0] - np.count_nonzero(_kept_flags))
            spots = spots[np.where(_kept_flags)[0]]
            print(
                f"-----{len(spots)} found in channel {color} in round {round_name} "
                f"(fit={total_fitted}, nan_removed={nan_removed}, boundary_removed={boundary_removed})",
                flush=True,
            )
            
            ### shift the spots
            spots = alignment.shift_spots(spots, drift)

            ### store the spots
            with h5py.File(output_file, 'a') as hdf_file:
                if (bit in hdf_file) and (overwrite):
                    # If the group exists, clear it
                    hdf_file[bit].clear()
                bit_info = hdf_file.create_group(bit)
                bit_info.create_dataset('drift', data=drift)
                bit_info.create_dataset('drift_flag', data=drift_flag)
                bit_info.create_dataset('spots', data=spots)
                print(f"---Spots for bit {bit} stored in hdf5 file", flush=True)
            
        print(f'-Finish analyzing images for round {round_name}.\n', flush=True)
    
    return
