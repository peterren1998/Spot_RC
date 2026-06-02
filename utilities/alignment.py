### adapted from ImageAnalysis3 - for alignment of images based on fiducial beads
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import pdist, squareform, euclidean
import pickle

def correct_image3D_by_microscope_param(image3D:np.ndarray, microscope_params:dict):
    """Correct 3D image with microscopy parameter"""
    _image = image3D.copy()
    if not isinstance(microscope_params, dict):
        raise TypeError(f"Wrong inputt ype for microscope_params, should be a dict")
    # transpose
    if 'transpose' in microscope_params and microscope_params['transpose']:
        _image = _image.transpose((0,2,1))
    if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
        _image = np.flip(_image, 2)
    if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
        _image = np.flip(_image, 1)
    return _image


def _find_boundary(_ct, _radius, _im_size):
    _bds = []
    for _c, _sz in zip(_ct, _im_size):
        _bds.append([max(_c-_radius, 0), min(_c+_radius, _sz)])
    
    return np.array(_bds, dtype=int)


def generate_drift_crops(single_im_size, coord_sel=None, drift_size=None):
    """Function to generate drift crop from a selected center and given drift size
    keywards:
        single_im_size: single image size to generate crops, np.ndarray like;
        coord_sel: selected center coordinate to split image into 4 rectangles, np.ndarray like;
        drift_size: size of drift crop, int or np.int;
    returns:
        crops: 4x3x2 np.ndarray. 
    """
    # check inputs
    _single_im_size = np.array(single_im_size)
    if coord_sel is None:
        coord_sel = np.array(_single_im_size/2, dtype=int)
    if coord_sel[-2] >= _single_im_size[-2] or coord_sel[-1] >= _single_im_size[-1]:
        raise ValueError(f"wrong input coord_sel:{coord_sel}, should be smaller than single_im_size:{single_im_size}")
    if drift_size is None:
        drift_size = int(np.max(_single_im_size)/4)
        
    # generate crop centers
    crop_cts = [
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  coord_sel[-1]/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2], 
                  (coord_sel[-1]+_single_im_size[-1])/2,]),
        np.array([coord_sel[-3]/2, 
                  coord_sel[-2]/2, 
                  coord_sel[-1],]),
        np.array([coord_sel[-3]/2, 
                  (coord_sel[-2]+_single_im_size[-2])/2, 
                  coord_sel[-1],]),                               
    ]
    # generate boundaries
    crops = [_find_boundary(_ct, _radius=drift_size/2, _im_size=single_im_size) for _ct in crop_cts]
        
    return np.array(crops)

def shift_spots(spots, drift):
    """shift the spots based on drift"""
    if np.shape(spots)[1] == 11: # this means 3d fitting result
        coords = np.array(spots).copy()[:,1:4]
    else:
        raise ValueError(f"Wrong input coords")
    # apply drift
    corr_coords = coords + drift
    # update spots
    output_coords = np.array(spots).copy()
    output_coords[:, 1:4] = corr_coords
    return output_coords

def _spot_coords(spots):
    """Extract zxy coordinates from fitted spots or coordinate arrays."""
    _spots = np.array(spots)
    if _spots.size == 0:
        return np.empty((0, 3), dtype=float)
    if _spots.ndim != 2:
        raise ValueError(f"spots should be a 2D array, got shape {np.shape(_spots)}")
    if _spots.shape[1] == 11:
        return _spots[:, 1:4].astype(float)
    if _spots.shape[1] == 3:
        return _spots.astype(float)
    raise ValueError(f"spots should be Nx11 fitted spots or Nx3 coordinates, got shape {np.shape(_spots)}")


def _as_3vector(value, default, name):
    """Normalize scalar/list tuple parameters to zxy vectors."""
    if value is None:
        value = default
    _value = np.array(value, dtype=float)
    if _value.ndim == 0:
        _value = np.repeat(float(_value), 3)
    if _value.shape != (3,):
        raise ValueError(f"{name} should be a scalar or 3 values in z,x,y order")
    if np.any(_value <= 0):
        raise ValueError(f"{name} values should be positive")
    return _value


def align_spots_by_displacement(
    src_spots,
    ref_spots,
    search_radius=(4, 30, 30),
    bin_size=(1, 2, 2),
    residual_radius=(1, 3, 3),
    min_matches=20,
    min_inlier_fraction=0.02,
    max_iterations=3,
    max_candidate_pairs=200000,
):
    """Estimate translation from source spots to reference spots.

    The returned drift has the same convention as align_image: add it to
    source-round spot coordinates to put them into the reference coordinate
    frame. Candidate matches are selected in an anisotropic zxy search window,
    then the densest displacement cluster is refined by robust inliers.
    """
    src_coords = _spot_coords(src_spots)
    ref_coords = _spot_coords(ref_spots)
    search_radius = _as_3vector(search_radius, (4, 30, 30), "search_radius")
    bin_size = _as_3vector(bin_size, (1, 2, 2), "bin_size")
    residual_radius = _as_3vector(residual_radius, (1, 3, 3), "residual_radius")

    qc = {
        "num_source_spots": int(len(src_coords)),
        "num_reference_spots": int(len(ref_coords)),
        "num_candidate_pairs": 0,
        "num_inliers": 0,
        "inlier_fraction": 0.0,
        "median_residual": np.nan,
        "max_candidate_pairs_exceeded": False,
    }

    if len(src_coords) == 0 or len(ref_coords) == 0:
        return np.zeros(3), "Failed signal alignment: no spots", qc

    scaled_src = src_coords / search_radius
    scaled_ref = ref_coords / search_radius
    ref_tree = KDTree(scaled_ref)
    candidate_displacements = []

    for src_coord, scaled_coord in zip(src_coords, scaled_src):
        ref_inds = ref_tree.query_ball_point(scaled_coord, r=1)
        if len(ref_inds) == 0:
            continue
        for ref_ind in ref_inds:
            candidate_displacements.append(ref_coords[ref_ind] - src_coord)
            if len(candidate_displacements) >= max_candidate_pairs:
                qc["max_candidate_pairs_exceeded"] = True
                break
        if qc["max_candidate_pairs_exceeded"]:
            break

    if len(candidate_displacements) == 0:
        return np.zeros(3), "Failed signal alignment: no candidate matches", qc

    displacements = np.array(candidate_displacements, dtype=float)
    qc["num_candidate_pairs"] = int(len(displacements))

    bins = np.floor(displacements / bin_size).astype(int)
    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    mode_bin = unique_bins[np.argmax(counts)]
    drift = np.median(displacements[np.all(bins == mode_bin, axis=1)], axis=0)

    inlier_flags = np.zeros(len(displacements), dtype=bool)
    for _ in range(int(max_iterations)):
        residuals = np.linalg.norm((displacements - drift) / residual_radius, axis=1)
        updated_flags = residuals <= 1
        if np.array_equal(updated_flags, inlier_flags):
            break
        inlier_flags = updated_flags
        if np.count_nonzero(inlier_flags) == 0:
            break
        drift = np.median(displacements[inlier_flags], axis=0)

    residuals = np.linalg.norm((displacements - drift) / residual_radius, axis=1)
    inlier_flags = residuals <= 1
    num_inliers = int(np.count_nonzero(inlier_flags))
    qc["num_inliers"] = num_inliers
    qc["inlier_fraction"] = float(num_inliers / len(displacements))
    if num_inliers > 0:
        qc["median_residual"] = float(np.median(residuals[inlier_flags]))

    if num_inliers < min_matches:
        return drift, "Failed signal alignment: too few inliers", qc
    if qc["inlier_fraction"] < min_inlier_fraction:
        return drift, "Poor signal alignment", qc
    return drift, "Optimal signal alignment", qc

def align_image(
    src_im:np.ndarray, 
    ref_im:np.ndarray, 
    crop_list=None,
    precision_fold=100, 
    min_good_drifts=3, drift_diff_th=1., drift_pixel_threshold = 150, z_drift_threshold = 4):
    """Function to align one image by either FFT or spot_finding
        both source and reference images should be corrected
    """

    single_im_size = src_im.shape
    # check crop_list:
    if crop_list is None:
        crop_list = generate_drift_crops(single_im_size)
    for _crop in crop_list:
        if np.shape(np.array(_crop)) != (3,2):
            raise IndexError(f"crop should be 3x2 np.ndarray.")
    
    # define result flag
    _result_flag = 0
    
    if np.shape(src_im) != np.shape(ref_im):
        raise IndexError(f"shape of target image:{np.shape(src_im)} and reference image:{np.shape(ref_im)} doesnt match!")

    ## crop images
    _crop_src_ims, _crop_ref_ims = [], []
    for _crop in crop_list:
        _s = tuple([slice(*np.array(_c,dtype=int)) for _c in _crop])
        _crop_src_ims.append(src_im[_s])
        _crop_ref_ims.append(ref_im[_s])

    ## align two images
    _drifts = []
    for _i, (_sim, _rim) in enumerate(zip(_crop_src_ims, _crop_ref_ims)):
        # calculate drift with autocorr
        _dft, _error, _phasediff = phase_cross_correlation(_rim, _sim, 
                                                               upsample_factor=precision_fold)
        # append if the drift calculated pass certain criteria
        if (len(np.where(np.abs(_dft)>drift_pixel_threshold)[0])==0) & (len(np.where(_dft==0)[0])<=1) & (np.abs(_dft)[0]<z_drift_threshold):
            _drifts.append(_dft) 

        # detect variance within existing drifts
        _mean_dft = np.nanmean(_drifts, axis=0)
        if len(_drifts) >= min_good_drifts:
            _dists = np.linalg.norm(_drifts-_mean_dft, axis=1)
            _kept_drift_inds = np.where(_dists <= drift_diff_th)[0]
            if len(_kept_drift_inds) >= min_good_drifts:
                _updated_mean_dft = np.nanmean(np.array(_drifts)[_kept_drift_inds], axis=0)
                _result_flag = 'Optimal alignment'
                break
    
    # if no good drift and just one good drift is detected. just return
    if len(_drifts)==0:
        return [0, 0, 0], 'Failed alignment'
    elif len(_drifts)==1:
        return _drifts[0], 'Poor alignment'

    if '_updated_mean_dft' not in locals():
        _drifts = np.array(_drifts)
        # select top 3 drifts
        _dist_mat = squareform(pdist(_drifts))
        np.fill_diagonal(_dist_mat, np.inf)
        # select closest pair
        _sel_inds = np.array(np.unravel_index(np.argmin(_dist_mat), np.shape(_dist_mat)))
        _sel_drifts = list(_drifts[_sel_inds])
        # select closest 3rd drift
        third_drift = _drifts[np.argmin(_dist_mat[:, _sel_inds].sum(1))]
        diff_first_third = euclidean(third_drift, _sel_drifts[0])
        diff_second_third = euclidean(third_drift, _sel_drifts[1])
        diff_first_second = euclidean(_sel_drifts[0], _sel_drifts[1])
        _cv = np.std([diff_first_second, diff_first_third, diff_second_third])/np.mean([diff_first_second, diff_first_third, diff_second_third])
        if _cv<=0.5:
            _sel_drifts.append(_drifts[np.argmin(_dist_mat[:, _sel_inds].sum(1))])
        # return mean
        _updated_mean_dft = np.nanmean(_sel_drifts, axis=0)
        _result_flag = 'Suboptimal alignment'

    return  _updated_mean_dft, _result_flag



def microscope_translation_spot(spots, microscope_params, image_size = (50,2048,2048)):
    """Translate spots given microscope"""
    _fov_spots = spots.copy()
    # load microscope.json
    if _fov_spots.shape[1]==11:
        _coords = _fov_spots[:, 1:4]
    else:
        _coords = _fov_spots
    
    # transpose
    if 'transpose' in microscope_params and microscope_params['transpose']:
        _coords = _coords[:, np.array([0,2,1])]
    if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
        _coords[:,2] = -1 * (_coords[:,2] - image_size[2]/2) + image_size[2]/2
    if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
        _coords[:,1] = -1 * (_coords[:,1] - image_size[1]/2) + image_size[1]/2
    
    if _fov_spots.shape[1]==11:
        _fov_spots[:, 1:4] = _coords
    else:
        _fov_spots = _coords
    
    return _fov_spots

def reverse_microscope_translation_spot(spots, microscope_params, image_size = (50,2048,2048)):
    """Translate spots given microscope"""
    _fov_spots = spots.copy()
    # load microscope.json
    if _fov_spots.shape[1]==11:
        _coords = _fov_spots[:, 1:4]
    else:
        _coords = _fov_spots

    # flip
    if  'flip_vertical' in microscope_params and microscope_params['flip_vertical']:
        _coords[:,1] = -1 * (_coords[:,1] - image_size[1]/2) + image_size[1]/2
    if 'flip_horizontal' in microscope_params and microscope_params['flip_horizontal']:
        _coords[:,2] = -1 * (_coords[:,2] - image_size[2]/2) + image_size[2]/2
    # transpose
    if 'transpose' in microscope_params and microscope_params['transpose']:
        _coords = _coords[:, np.array([0,2,1])]
    
    if _fov_spots.shape[1]==11:
        _fov_spots[:, 1:4] = _coords
    else:
        _fov_spots = _coords
    
    return _fov_spots

def generate_polynomial_data(coords, max_order):
    """function to generate polynomial data
    Args:
        coords: coordinates, np.ndarray, n_points by n_dimensions
        max_order: maximum order of polynomial, int
    Return:
        _X: data for polynomial, n_points by n_columns
    """
    import itertools
    _X = []
    for _order in range(int(max_order)+1):
        for _lst in itertools.combinations_with_replacement(
                coords.transpose(), _order):
            # initialize one column
            _xi = np.ones(np.shape(coords)[0])
            # calculate product
            for _v in _lst:
                _xi *= _v
            # append
            _X.append(_xi)
    # transpose to n_points by n_columns
    _X = np.array(_X).transpose()
    
    return _X

def generate_chromatic_function(chromatic_const_file, drift=None):
    """Function to generate a chromatic abbrevation translation function from
    _const.pkl file"""

    if isinstance(chromatic_const_file, dict):
        _info_dict = {_k:_v for _k,_v in chromatic_const_file.items()}
    elif isinstance(chromatic_const_file, str):
        _info_dict = pickle.load(open(chromatic_const_file, 'rb'))
    elif chromatic_const_file is None:
        if drift is None:
            #print('empty_function')
            def _shift_function(_coords, _drift=drift): 
                return _coords
            return _shift_function
        else:
            _info_dict ={
                'constants': [np.array([0]) for _dft in drift],
                'fitting_orders': np.zeros(len(drift),dtype=np.int),
                'ref_center': np.zeros(len(drift)),
            }
    else:
        raise TypeError(f"Wrong input chromatic_const_file")

    # extract info
    _consts = _info_dict['constants']
    _fitting_orders = _info_dict['fitting_orders']
    _ref_center = _info_dict['ref_center']
    # drift
    if drift is None:
        _drift = np.zeros(len(_ref_center))
    else:
        _drift = drift[:len(_ref_center)]
    
    def _shift_function(_coords, _drift=_drift, 
                        _consts=_consts, 
                        _fitting_orders=_fitting_orders, 
                        _ref_center=_ref_center,
                        ):
        """generated translation function with constants and drift"""
        # return empty if thats the case
        if len(_coords) == 0:
            return _coords
        else:
            _coords = np.array(_coords)

        if np.shape(_coords)[1] == len(_ref_center):
            _new_coords = np.array(_coords).copy()
        elif np.shape(_coords)[1] == 11: # this means 3d fitting result
            _new_coords = np.array(_coords).copy()[:,1:1+len(_ref_center)]
        else:
            raise ValueError(f"Wrong input coords")

        _shifts = []
        for _i, (_const, _order) in enumerate(zip(_consts, _fitting_orders)):
            # calculate dX
            _X = generate_polynomial_data(_new_coords- _ref_center[np.newaxis,:], 
                                          _order)
            # calculate dY
            _dy = np.dot(_X, _const)
            _shifts.append(_dy)
        _shifts = np.array(_shifts).transpose()

        # generate corrected coordinates
        _corr_coords = _new_coords - _shifts + _drift

        # return as input
        if np.shape(_coords)[1] == len(_ref_center):
            _output_coords = _corr_coords
        elif np.shape(_coords)[1] == 11: # this means 3d fitting result
            _output_coords = np.array(_coords).copy()
            _output_coords[:,1:1+len(_ref_center)] = _corr_coords
        return _output_coords
    
    # return function
    return _shift_function
