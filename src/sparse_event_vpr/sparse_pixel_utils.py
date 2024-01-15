import numpy as np
from tqdm.auto import tqdm
import numba as nb
import tonic
import torch
import os

from .constants import path_to_gps_files
from .metrics import getPAt100RFromPRVals, getPRCurveWrapperFromDistMatrix, getPRCurveWrapperFromScores, getRAtXPFromPRVals
from .utils import get_distance_matrix, get_gps, get_score_ratio_test, get_short_traverse_name, create_event_frames


def adjust_and_normalize_probabilities(event_data, apply_outlier_correction=True):
    adjusted_probs = np.copy(event_data)

    if apply_outlier_correction:  # Reduce probability for potential outliers
        outlier_threshold = event_data.mean() + 2 * event_data.std()
        adjusted_probs[adjusted_probs > outlier_threshold] = 0.01

    total_prob = adjusted_probs.sum()
    normalized_probs = adjusted_probs / total_prob
    return normalized_probs


def get_random_pixels(num_pixels, im_width, im_height, local_suppression_radius, prob_to_draw_from=None):
    """
    Generate a list of random pixels within an image.

    Args:
        num_pixels (int): The number of random pixels to generate.
        im_width (int): The width of the image.
        im_height (int): The height of the image.
        local_suppression_radius (float): The radius for local suppression.
        prob_to_draw_from (ndarray, optional): The probability distribution to draw from. Defaults to None.

    Returns:
        list: A list of random pixels, each represented as a tuple (y, x).

    Raises:
        ValueError: If a new random pixel cannot be found after 100 iterations.
    """
    random_pixels = []
    num_subsequent_rejections = 0
    with tqdm(total=num_pixels, desc="Pick random pixels") as pbar:
        while len(random_pixels) < num_pixels:
            random_idx_flat = np.random.choice(
                np.arange(0, im_height * im_width),
                p=prob_to_draw_from.reshape(-1) if prob_to_draw_from is not None else None,
            )
            random_pixel = np.unravel_index(random_idx_flat, (im_height, im_width))
            if len(random_pixels) == 0 or np.all(np.linalg.norm(np.array(random_pixels) - np.array(random_pixel), axis=1) > local_suppression_radius):
                random_pixels.append(random_pixel)
                num_subsequent_rejections = 0
                pbar.update(1)
            else:
                num_subsequent_rejections = num_subsequent_rejections + 1
                if num_subsequent_rejections > 100:
                    raise ValueError("Could not find new random pixel after 100 iterations")

    # check that number of unique elements equals the number of requested pixels
    assert len(list(set(random_pixels))) == num_pixels

    return random_pixels


@nb.njit(parallel=True)
def in1d_vec_nb(matrix, index_to_keep):
    # matrix and index_to_remove have to be numpy arrays
    # if index_to_keep is a list with different dtypes this
    # function will fail
    out = np.ones(matrix.shape[0], dtype=nb.int16) * -1

    for i in nb.prange(matrix.shape[0]):
        for test_item_idx, test_item in enumerate(index_to_keep):
            if matrix[i] == test_item:
                out[i] = test_item_idx

    return out


def create_event_frames_sparse_pixels(event_streams_numpy, random_pixels, frame_length, event_count, dtype_pixel_indices, im_width):
    event_stream_subsets_with_pixel_indices_structured = []

    for event_stream_idx in range(len(event_streams_numpy)):
        pixel_idx_np = event_streams_numpy[event_stream_idx][:, 1] * im_width + event_streams_numpy[event_stream_idx][:, 2]
        random_pixel_idx = np.array([rand_pix[1] * im_width + rand_pix[0] for rand_pix in random_pixels]).astype(np.uint64)
        random_pixel_ids = in1d_vec_nb(pixel_idx_np.reshape(-1), random_pixel_idx)
        mask = random_pixel_ids != -1
        event_stream_subset = np.vstack((event_streams_numpy[event_stream_idx][mask, 0], random_pixel_ids[mask], event_streams_numpy[event_stream_idx][mask, 3])).T

        event_stream_subsets_with_pixel_indices_structured.append(
            np.lib.recfunctions.unstructured_to_structured(event_stream_subset, dtype_pixel_indices))

    # TODO: Make num polarities flexible
    sensor_size_subsets = (len(random_pixels), 1, 2)

    if frame_length is not None:
        time_slices = [np.array(tonic.functional.slicing.slice_by_time(es, frame_length)) for es in event_stream_subsets_with_pixel_indices_structured]
    else:
        time_slices = [np.array(tonic.functional.slicing.slice_by_event_count(es, event_count)) for es in event_stream_subsets_with_pixel_indices_structured]

    event_frames_subsets = create_event_frames(
        event_stream_subsets_with_pixel_indices_structured,
        sensor_size_subsets,
        frame_length=frame_length,
        event_count=event_count,
        time_slices=time_slices,
    )
    return event_frames_subsets[0], time_slices[0]


def get_times_for_subsets(time_slices_subset):
    times_subsets = []
    last_time = None
    for i in range(len(time_slices_subset)):
        if len(time_slices_subset[i]) > 0:
            last_time = time_slices_subset[i][0][0]
        if last_time is not None:
            times_subsets.append((last_time - time_slices_subset[0][0][0]) / 10e5)
    return times_subsets


def evaluate_pr(dist_matrix_dict, dist_to_gt_dict, gt_tolerance):
    prvals_dict_no_ratio = {}
    prvals_dict_ratio = {}
    recall_99p_dict = {}
    precision_100r_dict = {}
    auc_dict = {}
    pr_curves_xs = []
    pr_curves_ys = []
    pr_curves_keys = []

    for k in dist_matrix_dict.keys():
        prvals_dict_no_ratio[k] = getPRCurveWrapperFromDistMatrix(dist_matrix_dict[k], gt_tolerance, dist_to_gt=dist_to_gt_dict[k] if k in dist_to_gt_dict else None)
        prvals_dict_ratio[k] = getPRCurveWrapperFromScores(dist_matrix_dict[k].argmin(axis=0), get_score_ratio_test(dist_matrix_dict[k]), gt_tolerance, dist_to_gt=dist_to_gt_dict[k] if k in dist_to_gt_dict else None)
        auc_dict["auc_" + k + "_no_ratio"] = np.trapz(prvals_dict_no_ratio[k][:, 0].tolist(), prvals_dict_no_ratio[k][:, 1].tolist())
        auc_dict["auc_" + k + "_ratio"] = np.trapz(prvals_dict_ratio[k][:, 0].tolist(), prvals_dict_ratio[k][:, 1].tolist())
        recall_99p_dict["recall_99p_" + k + "_no_ratio"] = getRAtXPFromPRVals(prvals_dict_no_ratio[k], 0.99) * 100
        recall_99p_dict["recall_99p_" + k + "_ratio"] = getRAtXPFromPRVals(prvals_dict_ratio[k], 0.99) * 100
        precision_100r_dict["precision_100r_" + k + "_no_ratio"] = getPAt100RFromPRVals(prvals_dict_no_ratio[k]) * 100
        precision_100r_dict["precision_100r_" + k + "_ratio"] = getPAt100RFromPRVals(prvals_dict_ratio[k]) * 100
        pr_curves_xs.append(prvals_dict_no_ratio[k][:, 1].tolist())
        pr_curves_ys.append(prvals_dict_no_ratio[k][:, 0].tolist())
        pr_curves_keys.append("Precision " + k + "_no_ratio")
        pr_curves_xs.append(prvals_dict_ratio[k][:, 1].tolist())
        pr_curves_ys.append(prvals_dict_ratio[k][:, 0].tolist())
        pr_curves_keys.append("Precision " + k + "_ratio")

    return pr_curves_xs, pr_curves_ys, pr_curves_keys, recall_99p_dict, precision_100r_dict, auc_dict


def load_gps_gt(traverses_to_compare):
    """
    Load GPS ground truth from files.

    Args:
        traverses_to_compare (list): List of traverses to compare.

    Returns:
        list: List of GPS ground truth data.
    """
    gps_gt = []
    if os.path.isfile(path_to_gps_files + get_short_traverse_name(traverses_to_compare[0]) + ".nmea") and os.path.isfile(
        path_to_gps_files + get_short_traverse_name(traverses_to_compare[1]) + ".nmea"
    ):
        tqdm.write("Loading GPS ground truth")
        gps_gt.append(get_gps(path_to_gps_files + get_short_traverse_name(traverses_to_compare[0]) + ".nmea"))
        gps_gt.append(get_gps(path_to_gps_files + get_short_traverse_name(traverses_to_compare[1]) + ".nmea"))
    return gps_gt


def create_frames_subset(frames, random_pixels):
    """
    Creates a subset of frames based on a subset of pixels.

    :param frames: The original frames data.
    :param random_pixels: The indices of the pixels to be used in the subset.
    :return: A subset of the frames.
    """
    frames_subset = np.zeros((len(frames), len(random_pixels)))
    for p_idx in range(len(random_pixels)):
        frames_subset[:, p_idx] = frames[(slice(None),) + random_pixels[p_idx]]
    return frames_subset


def compute_distance_matrices(frames_sets, device, seq_length):
    precomputed_convWeight = torch.eye(seq_length, device=device).unsqueeze(0).unsqueeze(0)

    dist_matrix_dict = {}
    for key, (ref_frame, qry_frame) in frames_sets.items():
        # Compute the distance matrix
        dist_matrix = get_distance_matrix(ref_frame, qry_frame, device=device)
        tqdm.write(f"Dist matrix {key} shape: {dist_matrix.shape}")

        # Apply convolution and convert to NumPy array
        dist_matrix_seq = torch.nn.functional.conv2d(dist_matrix.unsqueeze(0).unsqueeze(0), precomputed_convWeight).squeeze().cpu().numpy() / seq_length
        dist_matrix_dict[f"{key}_seq"] = dist_matrix_seq
        dist_matrix_dict[key] = dist_matrix.cpu().numpy()

    return dist_matrix_dict
