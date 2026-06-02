import argparse
import os
import re
import pandas as pd
import numpy as np
import pickle
import h5py
from utilities import dax, fitting, alignment
import json
from scipy.ndimage import shift

conventional_image_prefix = 'Conv_zscan_' # then str(fov).zfill(3)+'.dax'
# channels_for_FISH = ['750', '647', '561']
channels_for_FISH = ['750', '647']
default_signal_drift_channels = ['750', '647', '561']
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
    # drift method
    parser.add_argument('--drift-method', dest='drift_method', choices=['fiducial', 'signal'],
                        default='fiducial',
                        help='Use fiducial image correlation or adjacent-round signal-spot alignment')
    parser.add_argument('--fish-channels', dest='fish_channels', type=str, default=None,
                        help='Comma-separated signal channels to load and fit. Defaults to 750,647 for fiducial drift and 750,647,561 for signal drift')
    parser.add_argument('--dax-channels', dest='dax_channels', type=str, default=None,
                        help='Comma-separated full acquired DAX channel order for non-reference rounds')
    parser.add_argument('--ref-dax-channels', dest='ref_dax_channels', type=str, default=None,
                        help='Comma-separated full acquired DAX channel order for the reference round')
    parser.add_argument('--signal-drift-channels', dest='signal_drift_channels', type=str, default=None,
                        help='Comma-separated fitted signal channels to pool for signal drift. Defaults to all FISH channels')
    parser.add_argument('--signal-drift-search-radius', dest='signal_drift_search_radius', type=str,
                        default='4,30,30',
                        help='Anisotropic z,x,y search radius for signal-spot candidate matches')
    parser.add_argument('--signal-drift-bin-size', dest='signal_drift_bin_size', type=str,
                        default='1,2,2',
                        help='Anisotropic z,x,y displacement bin size for signal drift clustering')
    parser.add_argument('--signal-drift-residual-radius', dest='signal_drift_residual_radius', type=str,
                        default='1,3,3',
                        help='Anisotropic z,x,y residual radius for signal drift inliers')
    parser.add_argument('--signal-drift-min-matches', dest='signal_drift_min_matches', type=int,
                        default=20,
                        help='Minimum inlier candidate matches required to accept signal drift')
    # load whether to use conventional image file name
    parser.add_argument('--novel-name', dest='use_new_name', action='store_true', help='Use novel nomenclature sceme')
    # load the parameter dictionary file path
    parser.add_argument('-p', '--parameter', dest='parameter_file', type=str, required=True, 
                        help='Path to the pickle file containing parameters for picking')
    # load the correction dictionary file path
    parser.add_argument('-c', '--correction', dest='correction_file', type=str, required=True, 
                        help='Path to the pickle file for the correction dictionary')

    return parser


def _parse_channel_list(channel_text, default=None):
    if channel_text is None:
        return None if default is None else list(default)
    channels = [str(_ch).strip() for _ch in channel_text.split(',')]
    channels = [_ch for _ch in channels if len(_ch) > 0]
    if len(channels) == 0:
        return None if default is None else list(default)
    return channels


def _validate_required_channels(load_channels, required_channels, context):
    missing_channels = [ch for ch in required_channels if ch not in load_channels]
    if len(missing_channels) > 0:
        raise ValueError(
            f'{context} channel list is missing required channels {missing_channels}. '
            f'Full channel list: {load_channels}'
        )


def _parse_zxy_triplet(value, name):
    values = [float(_v.strip()) for _v in str(value).split(',') if len(_v.strip()) > 0]
    if len(values) == 1:
        return np.repeat(values[0], 3)
    if len(values) != 3:
        raise ValueError(f'{name} should be one value or three comma-separated z,x,y values')
    return np.array(values, dtype=float)


def _unique_channels(channels):
    unique = []
    for ch in channels:
        ch = str(ch)
        if ch not in unique:
            unique.append(ch)
    return unique


def _natural_sort_key(value):
    return [int(_part) if _part.isdigit() else _part for _part in re.split(r'(\d+)', str(value))]


def _empty_spots():
    return np.empty((0, 11), dtype=float)


def _pool_spot_coords(spots_by_channel, channels=None):
    if channels is None:
        channels = list(spots_by_channel.keys())
    pooled = []
    for ch in channels:
        if ch not in spots_by_channel:
            continue
        spots = np.array(spots_by_channel[ch])
        if spots.ndim == 2 and spots.shape[0] > 0 and spots.shape[1] == 11:
            pooled.append(spots[:, 1:4])
    if len(pooled) == 0:
        return np.empty((0, 3), dtype=float)
    return np.concatenate(pooled, axis=0)


def _write_dataset(group, name, data):
    if name in group:
        del group[name]
    if data is None:
        data = ''
    if isinstance(data, str):
        group.create_dataset(name, data=data, dtype=h5py.string_dtype(encoding='utf-8'))
    else:
        group.create_dataset(name, data=data)


def _write_spot_group(output_file, bit, spots, drift, drift_flag, overwrite=False,
                      drift_method='fiducial', parent_round=None, drift_qc=None):
    with h5py.File(output_file, 'a') as hdf_file:
        if bit in hdf_file:
            if overwrite:
                del hdf_file[bit]
            else:
                raise ValueError(f'Spot information for bit {bit} already exists. Use --overwrite to replace it.')
        bit_info = hdf_file.create_group(bit)
        _write_dataset(bit_info, 'drift', np.array(drift, dtype=float))
        _write_dataset(bit_info, 'drift_flag', drift_flag)
        _write_dataset(bit_info, 'drift_method', drift_method)
        _write_dataset(bit_info, 'drift_parent_round', parent_round)
        _write_dataset(bit_info, 'spots', np.array(spots))
        if drift_qc is not None:
            qc_group = bit_info.create_group('drift_qc')
            for key, value in drift_qc.items():
                _write_dataset(qc_group, key, value)


def _count_analyzed_bits(output_file, color_usage):
    total_bits = len(color_usage)
    analyzed_bits = 0
    last_drift_flag = None
    if not os.path.exists(output_file):
        return total_bits, analyzed_bits, last_drift_flag
    with h5py.File(output_file, 'r') as hdf_file:
        for bit in color_usage.values():
            if bit in hdf_file:
                bit_info = hdf_file[bit]
                if ('drift' in bit_info) and ('spots' in bit_info):
                    if 'drift_flag' in bit_info:
                        last_drift_flag = np.array(bit_info['drift_flag'])
                    analyzed_bits += 1
    return total_bits, analyzed_bits, last_drift_flag


def _load_existing_aligned_pool(output_file, color_usage, drift_channels):
    spots_by_channel = {}
    drift = None
    with h5py.File(output_file, 'r') as hdf_file:
        for color, bit in color_usage.items():
            if bit not in hdf_file:
                continue
            bit_info = hdf_file[bit]
            if 'spots' in bit_info:
                spots_by_channel[color] = np.array(bit_info['spots'])
            if drift is None and 'drift' in bit_info:
                drift = np.array(bit_info['drift'])
    if drift is None:
        drift = np.zeros(3)
    return _pool_spot_coords(spots_by_channel, drift_channels), drift


def _existing_drift_methods(output_file, color_usage):
    methods = set()
    if not os.path.exists(output_file):
        return methods
    with h5py.File(output_file, 'r') as hdf_file:
        for bit in color_usage.values():
            if bit not in hdf_file:
                continue
            bit_info = hdf_file[bit]
            if 'drift_method' not in bit_info:
                methods.add('unknown')
                continue
            method = np.array(bit_info['drift_method']).item()
            if isinstance(method, bytes):
                method = method.decode('utf-8')
            methods.add(str(method))
    return methods


def _save_reference_dapi(output_file, dapi_image, overwrite=False):
    with h5py.File(output_file, 'a') as hdf_file:
        if 'DNA_DAPI_image' in hdf_file:
            if overwrite:
                del hdf_file['DNA_DAPI_image']
            else:
                return
        hdf_file.create_dataset('DNA_DAPI_image', data=dapi_image)


def _candidate_image_file_names(fov):
    image_file_names = []
    for width in (3, 2, 1):
        image_file_name = conventional_image_prefix + str(fov).zfill(width) + '.dax'
        if image_file_name not in image_file_names:
            image_file_names.append(image_file_name)
    return image_file_names


def _build_round_table(data_folder, ref_round, image_file_names, signal_drift=False):
    if isinstance(image_file_names, str):
        image_file_names = [image_file_names]

    all_rounds = [_round for _round in os.listdir(data_folder) if _round[0] == 'H']
    if ref_round not in all_rounds:
        ref_round_path = os.path.join(data_folder, ref_round)
        raise ValueError(
            f'Missing reference round {ref_round} in data folder {data_folder}. '
            f'Tried round path: {ref_round_path}'
        )

    attempted_ref_files = [
        os.path.join(data_folder, ref_round, _image_file_name)
        for _image_file_name in image_file_names
    ]
    image_file_name = None
    for _image_file_name, _image_file in zip(image_file_names, attempted_ref_files):
        if os.path.exists(_image_file):
            image_file_name = _image_file_name
            break

    if image_file_name is None:
        raise ValueError(
            f'Missing reference image file for round {ref_round}. '
            f'Tried: {"; ".join(attempted_ref_files)}'
        )

    available_rounds = [
        _round for _round in all_rounds
        if os.path.exists(os.path.join(data_folder, _round, image_file_name))
    ]

    parent_round = {ref_round: None}
    if signal_drift:
        sorted_rounds = sorted(available_rounds, key=_natural_sort_key)
        ref_ind = sorted_rounds.index(ref_round)
        image_rounds = [ref_round] + sorted_rounds[ref_ind+1:] + list(reversed(sorted_rounds[:ref_ind]))
        for _ind in range(ref_ind + 1, len(sorted_rounds)):
            parent_round[sorted_rounds[_ind]] = sorted_rounds[_ind - 1]
        for _ind in range(ref_ind - 1, -1, -1):
            parent_round[sorted_rounds[_ind]] = sorted_rounds[_ind + 1]
    else:
        image_rounds = [ref_round]
        for _round in all_rounds:
            if (_round in available_rounds) and (_round not in image_rounds):
                image_rounds.append(_round)

    image_files = [os.path.join(data_folder, _round, image_file_name) for _round in image_rounds]
    return np.array(image_rounds), np.array(image_files), parent_round, image_file_name


def _nearest_aligned_parent(round_name, parent_rounds, aligned_spot_pools):
    parent_round = parent_rounds[round_name]
    while parent_round is not None and parent_round not in aligned_spot_pools:
        parent_round = parent_rounds[parent_round]
    return parent_round


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


def _correct_spot_chromatic(spots, color, correction_dict, microscope_dict):
    if len(spots) == 0:
        return spots
    if 'chromatic_constant' not in correction_dict.keys():
        return spots
    if str(color) not in correction_dict['chromatic_constant'].keys():
        return spots

    chromatic_function = alignment.generate_chromatic_function(correction_dict['chromatic_constant'][str(color)])
    if microscope_dict is not None:
        microscope_translated_spots = alignment.reverse_microscope_translation_spot(spots, microscope_dict)
        new_spots = chromatic_function(microscope_translated_spots)
        spots = alignment.microscope_translation_spot(new_spots, microscope_dict)
    else:
        spots = chromatic_function(spots)
    return spots


def _fit_spots_for_color(dax_cls, color, imageSize, parameters, max_num_seed,
                         min_num_seed, shifted_segment, correction_dict, microscope_dict,
                         round_name):
    dynamic_niters = parameters.get('dynamic niters', 10)
    seeds = fitting.get_seeds(getattr(dax_cls, f'im_{color}'), max_num_seeds=max_num_seed,
                              th_seed=parameters['seed_threshold'][color],
                              min_dynamic_seeds=min_num_seed,
                              dynamic_niters=dynamic_niters,
                              segment=shifted_segment,
                              minimum_threshold=parameters['min_threshold'])
    print(f"-----{len(seeds)} seeded with th={parameters['seed_threshold'][color]} in channel {color} for round {round_name}", flush=True)
    if len(seeds) == 0:
        return _empty_spots()

    fitter = fitting.iter_fit_seed_points(getattr(dax_cls, f'im_{color}'), seeds.T)
    fitter.firstfit()
    fitter.repeatfit()
    spots = np.array(fitter.ps)
    if spots.ndim != 2 or spots.shape[0] == 0:
        return _empty_spots()

    spots = spots[np.sum(np.isnan(spots), axis=1) == 0] # remove NaNs
    if spots.shape[0] == 0:
        return _empty_spots()

    # remove all boundary points
    _kept_flags = (spots[:, 1:4] > np.zeros(3)).all(1) \
        * (spots[:, 1:4] < np.array(imageSize)).all(1)
    spots = spots[np.where(_kept_flags)[0]]
    print(f"-----{len(spots)} found in channel {color} in round {round_name}", flush=True)
    spots = _correct_spot_chromatic(spots, color, correction_dict, microscope_dict)
    return spots


def SpotDNA():
    parser = build_parser()

    args, argv = parser.parse_known_args()
    signal_drift = args.drift_method == 'signal'
    fish_channels = _parse_channel_list(args.fish_channels)
    if fish_channels is None:
        fish_channels = list(default_signal_drift_channels if signal_drift else channels_for_FISH)
    dax_channels = _parse_channel_list(args.dax_channels)
    ref_dax_channels = _parse_channel_list(args.ref_dax_channels)
    if signal_drift and dax_channels is None:
        dax_channels = _unique_channels(fish_channels + [args.fiducial_channel])
    if signal_drift and ref_dax_channels is None:
        ref_dax_channels = _unique_channels(dax_channels + [args.dapi_channel])
    signal_drift_channels = _parse_channel_list(args.signal_drift_channels, default=fish_channels)
    signal_search_radius = _parse_zxy_triplet(args.signal_drift_search_radius, '--signal-drift-search-radius')
    signal_bin_size = _parse_zxy_triplet(args.signal_drift_bin_size, '--signal-drift-bin-size')
    signal_residual_radius = _parse_zxy_triplet(args.signal_drift_residual_radius, '--signal-drift-residual-radius')
    
    ### define image file
    if args.use_new_name is False:
        # generate image files, temporary
        image_file_names = _candidate_image_file_names(args.fov)
        image_rounds, image_files, signal_parent_round, image_file_name = _build_round_table(
            args.data_folder, args.ref_round, image_file_names, signal_drift=signal_drift)
        print(f'-Using image file name {image_file_name}', flush=True)
        print(f'START analyzing fov {args.fov} in rounds', end=': ', flush=True)
        for _round in image_rounds:
            print(_round, end=', ', flush=True)
        print('\n', flush=True)
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

    # load DNA mask and calculate the expected number of spots
    if os.path.exists(args.segment_file):
        dna_dapi_mask = np.load(args.segment_file)
        num_cells = len(np.unique(dna_dapi_mask)) - 1
        max_num_seed = int(num_cells*parameters['expected_spots_per_cell']*2)
        min_num_seed = int(num_cells*parameters['expected_spots_per_cell']*1)
        print(f'-Expected {min_num_seed} number of spots for {args.fov}')
    else:
        raise ValueError('Segmentation files does not exist!')
     

    # load microscope parameter dictionary
    microscope_dict = None
    if hasattr(args, 'microscope_file') and args.microscope_file is not None:
        with open(args.microscope_file, 'r') as file:
            microscope_dict = json.load(file)

    ref_fiducial_image = None
    fiducial_channel = args.fiducial_channel
    if not signal_drift:
        ### load reference image and store
        print('-Start loading reference fiducial and DAPI image', flush=True)
        # check whether the fiducial image has already exists
        if os.path.exists(output_file) and (not args.overwrite):
            with h5py.File(output_file, 'r+') as hdf_file:
                if 'reference_fiducial_image' in hdf_file:
                    ref_fiducial_image = hdf_file['reference_fiducial_image'][:]
                    print('---Read reference fiducial image directly from hdf5', flush=True)
        # when ref fiducial image is not loaded
        if ref_fiducial_image is None:
            ref_im_file = image_files[image_rounds==args.ref_round][0]
            # load reference image
            ref_image_channel = _unique_channels(fish_channels + [fiducial_channel, args.dapi_channel])
            ref_dax = dax.Dax_Processor(ref_im_file, ref_image_channel, imageSize, correction_dict, microscope_dict)
            ref_dax.load_image()
            ref_dax.correct_image()
            ref_fiducial_image = getattr(ref_dax, f'im_{fiducial_channel}').copy()
            # Write the reference image to the HDF5 file
            with h5py.File(output_file, 'a') as hdf_file:
                if args.overwrite and 'reference_fiducial_image' in hdf_file:
                    del hdf_file['reference_fiducial_image']
                if args.overwrite and 'DNA_DAPI_image' in hdf_file:
                    del hdf_file['DNA_DAPI_image']
                if 'reference_fiducial_image' not in hdf_file:
                    hdf_file.create_dataset('reference_fiducial_image',
                                            data=getattr(ref_dax, f'im_{fiducial_channel}'))
                if 'DNA_DAPI_image' not in hdf_file:
                    hdf_file.create_dataset('DNA_DAPI_image',
                                            data=getattr(ref_dax, f'im_{args.dapi_channel}'))
            print('---Finish saving reference fiducial and DAPI image\n', flush=True)
            del ref_dax
    else:
        print('-Using adjacent-round signal spots for drift; reference fiducial image is not loaded', flush=True)

    aligned_spot_pools = {}
    drift_by_round = {}

    ### iterate through the image files
    for image_file, round_name in zip(image_files, image_rounds):

        # load color usage
        color_usage = load_color_info(color_info_file, round_name, fish_channels)
        if len(color_usage.keys()) == 0:
            if signal_drift and round_name != args.ref_round:
                print(
                    f'---No requested FISH channels in Color_Usage for round {round_name}; '
                    'skip spot fitting and signal drift for this round.',
                    flush=True,
                )
                continue
            raise ValueError(f'Missing color usage information for round {round_name}')
        # check whether the color usage information has already exists
        total_bits, analyzed_bits, _drift = _count_analyzed_bits(output_file, color_usage)
        if signal_drift and (not args.overwrite) and (0 < analyzed_bits < total_bits):
            raise ValueError(
                f'Partial spot information already exists for round {round_name}. '
                'Signal-drift mode needs internally consistent per-round spots; rerun with --overwrite.'
            )
        if (analyzed_bits != 0) and (analyzed_bits == total_bits):
            print(f'---Spot information for round {round_name} already exists with {_drift}.', flush=True)
            if signal_drift:
                existing_methods = _existing_drift_methods(output_file, color_usage)
                if existing_methods != {'signal'}:
                    raise ValueError(
                        f'Existing spot information for round {round_name} was not written by signal-drift mode '
                        f'({existing_methods}). Rerun with --overwrite or use a fresh analysis folder.'
                    )
                aligned_pool, drift = _load_existing_aligned_pool(output_file, color_usage, signal_drift_channels)
                aligned_spot_pools[round_name] = aligned_pool
                drift_by_round[round_name] = drift
            continue

        ### load and correct image
        print(f'-Start analyzing images for round {round_name}', flush=True)
        if signal_drift:
            if round_name == args.ref_round:
                load_channels = list(ref_dax_channels)
                _validate_required_channels(load_channels, fish_channels + [args.dapi_channel], 'Reference DAX')
            else:
                load_channels = list(dax_channels)
                _validate_required_channels(load_channels, fish_channels, 'DAX')
        else:
            if round_name == args.ref_round:
                # load dapi channel for reference round
                load_channels = _unique_channels(fish_channels + [fiducial_channel, args.dapi_channel])
            else:
                load_channels = _unique_channels(fish_channels + [fiducial_channel])
        print(f"---Load image from file {image_file} with channels {','.join(load_channels)}", flush=True)
        dax_cls = dax.Dax_Processor(image_file, load_channels, imageSize, correction_dict, microscope_dict)
        dax_cls.load_image()
        dax_cls.correct_image()
        print(f'---Finish image correction for round {round_name}', flush=True)

        if signal_drift and round_name == args.ref_round:
            _save_reference_dapi(output_file, getattr(dax_cls, f'im_{args.dapi_channel}'), args.overwrite)

        if not signal_drift:
            ### calculate drift from fiducial images
            fiducial_image = getattr(dax_cls, f'im_{fiducial_channel}')
            if round_name != args.ref_round:
                print(f'---Calculate drift for round {round_name}', flush=True)
                drift, drift_flag = alignment.align_image(fiducial_image, ref_fiducial_image)
                print(f'---Drift for round {round_name} calculated with {drift_flag}', flush=True)
            else:
                print(f'---No drift calculation for reference round', flush=True)
                drift, drift_flag = [0, 0, 0], 'Reference image'
            del fiducial_image
            shifted_segment = shift(dna_dapi_mask, -np.array(drift), mode='constant', cval=0)

            #### start spot finding
            print(f'---Start spot finding for round {round_name}', flush=True)
            for color, bit in color_usage.items():
                ### check whether the bit information exists
                if args.overwrite == False and os.path.exists(output_file):
                    with h5py.File(output_file, 'r') as hdf_file:
                        if bit in hdf_file:
                            bit_info = hdf_file[bit]
                            if ('drift' in bit_info) and ('spots' in bit_info):
                                print(f'---Spot information for bit {bit} already exists.', flush=True)
                                continue

                spots = _fit_spots_for_color(dax_cls, color, imageSize, parameters, max_num_seed,
                                             min_num_seed, shifted_segment, correction_dict,
                                             microscope_dict, round_name)
                spots = alignment.shift_spots(spots, drift)

                _write_spot_group(output_file, bit, spots, drift, drift_flag,
                                  overwrite=args.overwrite, drift_method='fiducial')
                print(f"---Spots for bit {bit} stored in hdf5 file", flush=True)
                del spots

            print(f'-Finish analyzing images for round {round_name}.\n', flush=True)
            del dax_cls
            del shifted_segment
            continue

        parent_round = _nearest_aligned_parent(round_name, signal_parent_round, aligned_spot_pools)
        if parent_round is None:
            seed_drift = np.zeros(3)
        else:
            seed_drift = drift_by_round.get(parent_round, np.zeros(3))
        shifted_segment = shift(dna_dapi_mask, -np.array(seed_drift), mode='constant', cval=0)

        #### start spot finding before signal drift calculation
        print(f'---Start pre-drift spot finding for round {round_name}', flush=True)
        spots_by_channel = {}
        for color, bit in color_usage.items():
            spots_by_channel[color] = _fit_spots_for_color(dax_cls, color, imageSize, parameters,
                                                           max_num_seed, min_num_seed,
                                                           shifted_segment, correction_dict,
                                                           microscope_dict, round_name)

        raw_signal_pool = _pool_spot_coords(spots_by_channel, signal_drift_channels)
        if parent_round is None:
            print(f'---No signal drift calculation for reference round', flush=True)
            drift, drift_flag = np.zeros(3), 'Reference image'
            drift_qc = {
                'num_source_spots': int(len(raw_signal_pool)),
                'num_reference_spots': int(len(raw_signal_pool)),
                'num_candidate_pairs': 0,
                'num_inliers': int(len(raw_signal_pool)),
                'inlier_fraction': 1.0 if len(raw_signal_pool) > 0 else 0.0,
                'median_residual': 0.0,
                'max_candidate_pairs_exceeded': False,
            }
        else:
            parent_pool = aligned_spot_pools.get(parent_round)
            print(f'---Calculate signal drift for round {round_name} from adjacent round {parent_round}', flush=True)
            drift, drift_flag, drift_qc = alignment.align_spots_by_displacement(
                raw_signal_pool,
                parent_pool,
                search_radius=signal_search_radius,
                bin_size=signal_bin_size,
                residual_radius=signal_residual_radius,
                min_matches=args.signal_drift_min_matches,
            )
            if str(drift_flag).startswith('Failed'):
                drift = np.array(drift_by_round.get(parent_round, np.zeros(3)), dtype=float)
                drift_flag = f'{drift_flag}; reused parent drift from {parent_round}'
            print(f'---Signal drift for round {round_name} calculated with {drift_flag}: {np.array(drift)}', flush=True)

        shifted_spots_by_channel = {}
        for color, spots in spots_by_channel.items():
            shifted_spots_by_channel[color] = alignment.shift_spots(spots, drift)

        aligned_spot_pools[round_name] = _pool_spot_coords(shifted_spots_by_channel, signal_drift_channels)
        drift_by_round[round_name] = np.array(drift, dtype=float)

        for color, bit in color_usage.items():
            _write_spot_group(output_file, bit, shifted_spots_by_channel[color], drift, drift_flag,
                              overwrite=args.overwrite, drift_method='signal',
                              parent_round=parent_round, drift_qc=drift_qc)
            print(f"---Spots for bit {bit} stored in hdf5 file", flush=True)

        print(f'-Finish analyzing images for round {round_name}.\n', flush=True)
        del dax_cls
        del shifted_segment

    return
