import argparse
import os
import pandas as pd
import numpy as np
import pickle
import h5py
from utilities import dax, fitting, alignment
import json

conventional_image_prefix = 'Conv_zscan_' # then str(fov).zfill(3)+'.dax'
channels_for_FISH = ['750', '647', '561']
num_pixel_xy = 2048
distance_zxy = [250, 108, 108]

def build_parser():
    parser = argparse.ArgumentParser(description='Spot finding in RC')

    # specify parent data folder
    parser.add_argument('-d', '--data-folder', dest='data_folder', type=str, required=True, 
                        help='Path to the parent data folder')
    # define output folder
    # the output folder should contain the color usage information
    parser.add_argument('-o', '--analysis-folder', dest='analysis_folder', type=str, required=True,
                        help='Path to the folder storing analyzed result')
    # DNA segment file
    parser.add_argument('-s', '--segment', dest='segment_file', type=str, required=True, help='File path to DNA segment')
    # load fov information
    parser.add_argument('--fov', type=int, required=True, help='fov number to be analyzed')
    # load reference round
    parser.add_argument('--ref', dest='ref_round', type=str, required=True, help='The folder name for the reference round')
    # number of z stacks
    parser.add_argument('-z', '--num-z', dest='num_z', type=int, default=50, help='The number of z stacks')
    # microscope dictionary
    parser.add_argument('-m', '--microscope', dest='microscope_file', type=str, help='File path to microscope dictionary')
    # overwrite
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the stored analyzed data')
    # load fiducial channel
    parser.add_argument('--fiducial-channel', dest='fiducial_channel', type=str, default='488', help='Color channel for fiducial image')
    # load DAPI channel
    parser.add_argument('--DAPI-channel', dest='dapi_channel', type=str, default='405', help='Color channel for DAPI image')
    # load whether to use conventional image file name
    parser.add_argument('--novel-name', dest='use_new_name', action='store_true', help='Use novel nomenclature sceme')
    # load the parameter dictionary file path
    parser.add_argument('-p', '--parameter', dest='parameter_file', type=str, required=True, 
                        help='Path to the pickle file containing parameters for picking')
    # load the correction dictionary file path
    parser.add_argument('-c', '--correction', dest='correction_file', type=str, required=True, 
                        help='Path to the pickle file for the correction dictionary')

    return parser

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

def SpotDNA():
    parser = build_parser()

    args, argv = parser.parse_known_args()
    
    ### define image file
    if args.use_new_name is False:
        _image_rounds = np.array([_round for _round in os.listdir(args.data_folder) if _round[0]=='H']) # TO DO add sanity check for folder validity)
        # check reference round
        if args.ref_round not in _image_rounds:
            raise ValueError(f'Missing reference round in data folder')
        # generate image files, temporary
        image_file_name = conventional_image_prefix+str(args.fov).zfill(3)+'.dax'
        _image_files = np.array([os.path.join(args.data_folder, _round_name, image_file_name) for _round_name in _image_rounds])
        # check image files and start processing from the reference round
        image_rounds = [args.ref_round]
        image_files = [os.path.join(args.data_folder, args.ref_round, image_file_name)]
        for fl, _round in zip(_image_files, _image_rounds):
            if (os.path.exists(fl)) and (_round not in image_rounds):
                image_rounds.append(_round)
                image_files.append(fl)
        print(f'START analyzing fov {args.fov} in rounds', end=': ', flush=True)
        for _round in image_rounds:
            print(_round, end=', ', flush=True)
        print('\n', flush=True)
        # change the format into array
        image_rounds = np.array(image_rounds)
        image_files = np.array(image_files)
    ### TO DO: write code for new naming scheme
    else:
        raise Exception('New naming scheme needs to be written')
    
    # define basic image parameters
    imageSize = [args.num_z, num_pixel_xy, num_pixel_xy]

    ### define output file
    # identify whether Color_Usage.csv is present
    color_info_file = os.path.join(args.analysis_folder, 'Color_Usage.csv')
    if not os.path.exists(color_info_file):
        raise ValueError(f'{color_info_file} is missing')
    
    ### define output file
    output_file = os.path.join(args.analysis_folder, image_file_name.replace('.dax', '.hdf5'))
    ### TO DO: write the overwrite infomation

    # load correction dictionary
    correction_dict = pickle.load(open(args.correction_file, 'rb'))
    
    ### load parameters for picking
    parameters = pickle.load(open(args.parameter_file, 'rb'))

    # load DNA mask and calculate the expected number of pots
    if os.path.exists(args.segment_file):
        dna_dapi_mask = np.load(args.segment_file)
        num_cells = len(np.unique(dna_dapi_mask)) - 1
        max_num_seed = int(num_cells*parameters['expected_spots_per_cell']*2)
        min_num_seed = int(num_cells*parameters['expected_spots_per_cell']*1)
        print(f'-Expected {min_num_seed} number of spots for {args.fov}')
    else:
        raise ValueError('Segmentation files does not exist!')
     

    # load microscope parameter dictionary
    if hasattr(args, 'microscope_file') and args.microscope_file is not None:
        with open(args.microscope_file, 'r') as file:
            microscope_dict = json.load(file)

    ### load reference image and store
    print('-Start loading reference fiducial and DAPI image', flush=True)
    fiducial_channel = args.fiducial_channel
    # check whether the fiducial image has already exists
    ref_fiducial_image = None
    if os.path.exists(output_file) and (not args.overwrite):
        with h5py.File(output_file, 'r+') as hdf_file:
            if 'reference_fiducial_image' in hdf_file:
                ref_fiducial_image = hdf_file['reference_fiducial_image'][:]
                print('---Read reference fiducial image directly from hdf5', flush=True)
    # when ref fiducial image is not loaded
    if ref_fiducial_image is None:
        ref_im_file = image_files[image_rounds==args.ref_round][0]
        # load reference image
        ref_image_channel = channels_for_FISH + [fiducial_channel, args.dapi_channel]
        ref_dax = dax.Dax_Processor(ref_im_file, ref_image_channel, imageSize, correction_dict, microscope_dict)
        ref_dax.load_image()
        ref_dax.correct_image()
        ref_fiducial_image = getattr(ref_dax, f'im_{fiducial_channel}').copy()
        # Write the reference image to the HDF5 file
        with h5py.File(output_file, 'a') as hdf_file:
            hdf_file.create_dataset('reference_fiducial_image', 
                                    data=getattr(ref_dax, f'im_{fiducial_channel}'))
            hdf_file.create_dataset('DNA_DAPI_image', 
                                    data=getattr(ref_dax, f'im_{args.dapi_channel}'))
        print('---Finish saving reference fiducial and DAPI image\n', flush=True)
        del ref_dax

    ### iterate through the image files
    for image_file, round_name in zip(image_files, image_rounds):
        
        # load color usage
        color_usage = load_color_info(color_info_file, round_name)
        # stop if there is no color usage information
        if len(color_usage.keys())==0:
            raise ValueError(f'Missing color usage information for round {round_name}')
        # check whether the color usage information has already exists
        total_bits  = 0
        analyzed_bits = 0
        for color, bit in color_usage.items():
            ### check whether the bit information exists
            total_bits += 1
            if args.overwrite==False:
                with h5py.File(output_file, 'r+') as hdf_file:
                    if bit in hdf_file:
                        bit_info = hdf_file[bit]
                        if ('drift' in bit_info) and ('spots' in bit_info):
                            _drift = np.array(bit_info['drift_flag'])
                            analyzed_bits += 1
        if (analyzed_bits!=0) and (analyzed_bits==total_bits):
            print(f'---Spot information for round {round_name} already exists with {_drift}.', flush=True)
            continue
        
        ### load and correct image
        print(f'-Start analyzing images for round {round_name}', flush=True)
        if round_name == args.ref_round:
            # load dapi channel for reference round
            load_channels = channels_for_FISH+[fiducial_channel, args.dapi_channel]
        else:
            load_channels = channels_for_FISH+[fiducial_channel]
        print(f'---Load image from file {image_file}', flush=True)
        dax_cls = dax.Dax_Processor(image_file, load_channels, imageSize, correction_dict, microscope_dict)
        dax_cls.load_image()
        dax_cls.correct_image()
        print(f'---Finish image correction for round {round_name}', flush=True)
        
        ### calculate drift
        # load and correct fiducial images
        fiducial_image = getattr(dax_cls, f'im_{fiducial_channel}')
        # calculate
        if round_name!=args.ref_round:
            print(f'---Calculate drift for round {round_name}', flush=True)
            drift, drift_flag = alignment.align_image(fiducial_image, ref_fiducial_image)
            print(f'---Drift for round {round_name} calculated with {drift_flag}', flush=True)
        else:
            print(f'---No drift calculation for reference round', flush=True)
            drift, drift_flag = [0,0,0], 'Reference image'
        # release RAM
        del fiducial_image
        # shift the segment
        from scipy.ndimage import shift
        shifted_segment = shift(dna_dapi_mask, -np.array(drift), mode='constant', cval=0)

        #### start spot finding
        print(f'---Start spot finding for round {round_name}', flush=True)
        for color, bit in color_usage.items():
            ### check whether the bit information exists
            if args.overwrite==False:
                with h5py.File(output_file, 'r+') as hdf_file:
                    if bit in hdf_file:
                        bit_info = hdf_file[bit]
                        if ('drift' in bit_info) and ('spots' in bit_info):
                            print(f'---Spot information for bit {bit} already exists.', flush=True)
                            continue

            ### generate seed
            seeds = fitting.get_seeds(getattr(dax_cls, f'im_{color}'), max_num_seeds=max_num_seed, 
                                      th_seed=parameters['seed_threshold'][color], 
                                      min_dynamic_seeds=min_num_seed,
                                      segment=shifted_segment)
            print(f"-----{len(seeds)} seeded with th={parameters['seed_threshold'][color]} in channel {color} for round {round_name}", flush=True)
            if len(seeds)>0:
                ### fitting
                fitter = fitting.iter_fit_seed_points(getattr(dax_cls, f'im_{color}'), seeds.T)    
                # release RAM
                del seeds
                # fit
                fitter.firstfit()
                # check
                fitter.repeatfit()
                # get spots
                spots = np.array(fitter.ps)
                spots = spots[np.sum(np.isnan(spots),axis=1)==0] # remove NaNs
                # remove all boundary points
                _kept_flags = (spots[:,1:4] > np.zeros(3)).all(1) \
                    * (spots[:, 1:4] < np.array(imageSize)).all(1)
                spots = spots[np.where(_kept_flags)[0]]
                print(f"-----{len(spots)} found in channel {color} in round {round_name}", flush=True)
                
                ### generate shift function by chromatic abberation
                if ('chromatic_constant' in correction_dict.keys()):
                    if str(color) in correction_dict['chromatic_constant'].keys():
                        chromatic_function = alignment.generate_chromatic_function(correction_dict['chromatic_constant'][str(color)])
                        # need to get the coordinates that reverse by microscope parameters
                        microscope_translated_spots = alignment.reverse_microscope_translation_spot(spots, microscope_dict)
                        new_spots = chromatic_function(microscope_translated_spots)
                        # change back by microscope parameters
                        spots = alignment.microscope_translation_spot(new_spots, microscope_dict)
                        # release RAM
                        del microscope_translated_spots
                        del new_spots
                        del chromatic_function
                
                # apply drift
                spots = alignment.shift_spots(spots, drift)
            else:
                spots = []

            ### store the spots
            with h5py.File(output_file, 'a') as hdf_file:
                if (bit in hdf_file) and (args.overwrite):
                    # If the group exists, clear it
                    hdf_file[bit].clear()
                bit_info = hdf_file.create_group(bit)
                bit_info.create_dataset('drift', data=drift)
                bit_info.create_dataset('drift_flag', data=drift_flag)
                bit_info.create_dataset('spots', data=spots)
                print(f"---Spots for bit {bit} stored in hdf5 file", flush=True)
            
            # release RAM
            del spots

        print(f'-Finish analyzing images for round {round_name}.\n', flush=True)
        # release RAM
        del dax_cls
        del shifted_segment

    return
