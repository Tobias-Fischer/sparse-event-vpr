import os
import re
import glob

import numpy as np
import torch
import pandas as pd
import cv2
from scipy import interpolate

import tonic
import pynmea2

from tqdm.auto import tqdm

from .constants import (
    path_to_event_files,
    path_to_frames,
    video_beginning,
    qcr_traverses_first_times,
    qcr_traverses_last_times,
    brisbane_event_traverses_aliases,
    qcr_traverses_aliases,
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def none_or_str(value):
    if value == "None":
        return None
    return value


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def get_short_traverse_name(traverse_name):
    m = re.search(r"(\d)\D*$", traverse_name)
    traverse_short = traverse_name[: m.start() + 1]
    return traverse_short


def get_traverse_alias(traverse_name_short):
    if traverse_name_short in brisbane_event_traverses_aliases:
        return brisbane_event_traverses_aliases[traverse_name_short]
    elif traverse_name_short in qcr_traverses_aliases:
        return qcr_traverses_aliases[traverse_name_short]
    else:
        return traverse_name_short


def load_event_streams(event_streams_to_load, dir_to_load_from=path_to_event_files):
    event_streams = []
    for event_stream in tqdm(event_streams_to_load):
        event_streams.append(pd.read_parquet(os.path.join(dir_to_load_from, event_stream)))
    return event_streams


def get_size(event_stream):
    im_width, im_height = int(event_stream["x"].max() + 1), int(event_stream["y"].max() + 1)
    return im_width, im_height


def print_duration(event_stream):
    print(f'Duration: {((event_stream.iloc[-1]["t"] - event_stream.iloc[0]["t"]) / 1e6):.2f}s (which is {len(event_stream)} events)')


def create_event_frames(event_streams_numpy, sensor_size, frame_length=1e6, event_count=None, time_slices=None):
    if time_slices is None:
        time_slices = [None] * len(event_streams_numpy)

    event_frames = [
        tonic.functional.to_frame_numpy(
            event_stream_numpy,
            sensor_size,
            time_window=int(frame_length) if frame_length is not None else None,
            event_count=event_count,
            overlap=0,
            event_slices=event_slices,
        )
        for event_stream_numpy, event_slices in zip(event_streams_numpy, time_slices)
    ]
    event_frames_pos = [event_frame[:, 0, ...] for event_frame in event_frames]
    event_frames_neg = [event_frame[:, 1, ...] for event_frame in event_frames]

    # Combine negative and positive polarities
    event_frames_total = [event_frame_pos + event_frame_neg for event_frame_pos, event_frame_neg in zip(event_frames_pos, event_frames_neg)]
    return event_frames_total


def get_times_for_streams_const_time(event_stream, time_window, include_incomplete=False):
    times = event_stream["t"]
    stride = time_window

    last_time = times.iloc[-1] if isinstance(times, pd.Series) else times[-1]
    begin_time = times.iloc[0] if isinstance(times, pd.Series) else times[0]

    if include_incomplete:
        n_slices = int(np.ceil(((last_time - begin_time) - time_window) / stride) + 1)
    else:
        n_slices = int(np.floor(((last_time - begin_time) - time_window) / stride) + 1)

    window_start_times = np.arange(n_slices) * stride
    # window_end_times = window_start_times + time_window

    return window_start_times


def get_times_for_streams_const_count(event_stream, event_count, include_incomplete=False):
    n_events = len(event_stream)
    event_count = min(event_count, n_events)

    stride = event_count
    if stride <= 0:
        raise Exception("Inferred stride <= 0")

    if include_incomplete:
        n_slices = int(np.ceil((n_events - event_count) / stride) + 1)
    else:
        n_slices = int(np.floor((n_events - event_count) / stride) + 1)

    times = event_stream["t"].to_numpy()
    begin_time = times.iloc[0] if isinstance(times, pd.Series) else times[0]
    indices_start = (np.arange(n_slices) * stride).astype(int)

    return times[indices_start] - begin_time


def get_conventional_frames(traverse, desired_times):
    filenames = sorted(glob.glob(path_to_frames + "/" + get_short_traverse_name(traverse) + "/frames/*.png"))
    timestamps = np.array([float(os.path.basename(filename).replace(".png", "")) for filename in filenames])
    idx_to_load = []
    for time in desired_times:
        idx_to_load.append((np.abs(time - timestamps)).argmin())

    conventional_frames = []
    for idx in idx_to_load:
        conventional_frames.append(cv2.imread(filenames[idx], cv2.IMREAD_GRAYSCALE))

    return np.array(conventional_frames)


def sync_event_streams(event_streams, traverses_to_compare, gps_gt):
    event_streams_synced = []
    for event_stream_idx, (event_stream, name) in enumerate(zip(event_streams, traverses_to_compare)):
        short_name = get_short_traverse_name(name)
        start_time = event_stream.iloc[0]["t"]
        if short_name.startswith("bags_"):
            if short_name in qcr_traverses_first_times:
                first_idx = event_stream["t"].searchsorted(start_time + qcr_traverses_first_times[short_name])
            else:
                first_idx = 0
            if short_name in qcr_traverses_last_times:
                last_idx = event_stream["t"].searchsorted(start_time + qcr_traverses_last_times[short_name])
            else:
                last_idx = None
            event_streams_synced.append(event_stream[first_idx:last_idx].reset_index(drop=True))
        elif short_name.startswith("dvs_vpr_"):
            first_idx = event_stream["t"].searchsorted(video_beginning[short_name] * 1e6)
            if event_stream_idx < len(gps_gt):
                last_idx = event_stream["t"].searchsorted((video_beginning[short_name] + gps_gt[event_stream_idx][-1, 2]) * 1e6)
            else:
                last_idx = None
            event_streams_synced.append(event_stream[first_idx:last_idx].reset_index(drop=True))
        else:
            event_streams_synced.append(event_stream)
    return event_streams_synced


def remove_random_bursts(event_frames, threshold):
    event_frames[event_frames > threshold] = threshold
    return event_frames


def get_distance_matrix(ref_traverse: np.ndarray, qry_traverse: np.ndarray, metric="cityblock", device=None):
    dev = device if device != torch.device("mps") else None

    a = torch.from_numpy(ref_traverse.reshape(ref_traverse.shape[0], -1).astype(np.float32)).unsqueeze(0).to(dev)
    b = torch.from_numpy(qry_traverse.reshape(qry_traverse.shape[0], -1).astype(np.float32)).unsqueeze(0).to(dev)
    if metric == "cityblock":
        torch_dist = torch.cdist(a, b, 1)[0]
    elif metric == "euclidean":
        torch_dist = torch.cdist(a, b, 2)[0]
    elif metric == "cosine":
        def cosine_distance_torch(x1, x2=None, eps=1e-8):
            x2 = x1 if x2 is None else x2
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
            return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

        torch_dist = cosine_distance_torch(a.squeeze(0), b.squeeze(0))
    else:
        raise ValueError("Distance not supported")

    if device == torch.device("mps"):
        torch_dist = torch_dist.to(device)

    return torch_dist


def get_score_ratio_test(dist_matrix, neighborhood_exclusion_radius=3):
    match_scores_revised = np.empty(dist_matrix.shape[1], dtype=np.float32)
    for query in range(dist_matrix.shape[1]):
        refs_sorted = dist_matrix[:, query].argsort()
        best_match = refs_sorted[0]
        second_best_match = refs_sorted[np.abs(refs_sorted - best_match) >= neighborhood_exclusion_radius][0]
        if dist_matrix[second_best_match, query] == 0:  # Ignore division by zero
            match_scores_revised[query] = 1.0
        else:
            match_scores_revised[query] = dist_matrix[best_match, query] / dist_matrix[second_best_match, query]
    return match_scores_revised


def get_gps(nmea_file_path):
    nmea_file = open(nmea_file_path, encoding="utf-8")

    latitudes, longitudes, timestamps, distances = [], [], [], []

    first_timestamp = None
    previous_lat, previous_lon = None, None

    for line in nmea_file.readlines():
        try:
            msg = pynmea2.parse(line)
            if msg.sentence_type not in ["GSV", "VTG", "GSA"]:
                if first_timestamp is None:
                    first_timestamp = msg.timestamp
                    previous_lat, previous_lon = msg.latitude, msg.longitude

                # print(msg.timestamp, msg.latitude, msg.longitude)
                # print(repr(msg.latitude))
                dist_to_prev = np.linalg.norm(np.array([msg.latitude, msg.longitude]) - np.array([previous_lat, previous_lon]))
                if msg.latitude != 0 and msg.longitude != 0 and msg.latitude != previous_lat and msg.longitude != previous_lon and dist_to_prev > 0.0001:
                    timestamp_diff = (msg.timestamp.hour - first_timestamp.hour) * 3600 + (msg.timestamp.minute - first_timestamp.minute) * 60 + (msg.timestamp.second - first_timestamp.second)
                    latitudes.append(msg.latitude)
                    longitudes.append(msg.longitude)
                    timestamps.append(timestamp_diff)
                    distances.append(dist_to_prev)  # noqa
                    previous_lat, previous_lon = msg.latitude, msg.longitude

        except pynmea2.ParseError as e:  # noqa
            # print('Parse error: {} {}'.format(msg.sentence_type, e))
            continue

    return np.array(np.vstack((latitudes, longitudes, timestamps, distances))).T


def interpolate_gps(gps_data, desired_times=None):
    f_time_to_lat = interpolate.interp1d(gps_data[:, 2], gps_data[:, 0], fill_value="extrapolate")
    f_time_to_lon = interpolate.interp1d(gps_data[:, 2], gps_data[:, 1], fill_value="extrapolate")

    if desired_times is None:
        desired_times = np.arange(gps_data[-1, 2])

    new_lat = np.array([f_time_to_lat(t) for t in desired_times])
    new_lon = np.array([f_time_to_lon(t) for t in desired_times])
    return np.array(np.vstack((new_lat, new_lon, desired_times))).T


def check_parameters(args, traverses_to_compare):
    """
    Check and process the parameters for performing sparse event VPR.

    Args:
        args: The command line arguments.
        traverses_to_compare: The traverses to compare.

    Returns:
        A tuple containing the following:
        - gt_tolerance: The ground truth tolerance.
        - event_frame_length_all_pixels: The event frame length for all pixels.
        - event_frame_length_sparse_pixels: The event frame length for sparse pixels.
        - event_frame_num_sparse_pixels: The number of events per frame for sparse pixels.
        - suffix_no_seq: The suffix without sequence information.
    """
    suffix_no_seq = f"_{args.num_target_pixels}_pixels"
    if args.use_saliency: suffix_no_seq += "_saliency"
    else: suffix_no_seq += "_nosaliency"

    gt_tolerance = args.gt_tolerance_percentage if "bags_" in traverses_to_compare[0] else args.gt_tolerance_meters
    event_frame_length_all_pixels = args.event_frame_length_all_pixels
    assert not (event_frame_length_all_pixels and args.event_frame_count_all_pixels)

    event_frame_length_sparse_pixels = args.event_frame_length_sparse_pixels
    event_frame_num_sparse_pixels = int(args.multiplier_factor * args.num_target_pixels) if args.multiplier_factor is not None else None
    if event_frame_num_sparse_pixels is not None:
        suffix_no_seq += f"_events_per_frame_{event_frame_num_sparse_pixels}"

    if args.use_conventional_frames:
        suffix_no_seq += "_use_conventional_frames"

    assert not (event_frame_length_sparse_pixels and event_frame_num_sparse_pixels)
    assert event_frame_length_sparse_pixels or event_frame_num_sparse_pixels

    if event_frame_num_sparse_pixels is not None: tqdm.write(f"Use {event_frame_num_sparse_pixels} events in a frame")
    else: tqdm.write(f"A frame is {event_frame_length_sparse_pixels}μs long")

    if event_frame_length_sparse_pixels is not None:
        assert event_frame_length_sparse_pixels == event_frame_length_all_pixels
    return gt_tolerance, event_frame_length_all_pixels, event_frame_length_sparse_pixels, event_frame_num_sparse_pixels, suffix_no_seq
