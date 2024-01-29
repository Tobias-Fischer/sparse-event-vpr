#!/usr/bin/env python

import argparse
import os
from multiprocessing.dummy import freeze_support

import torch
import numpy as np
from codetiming import Timer
from tqdm.auto import tqdm

from sparse_event_vpr.constants import gt_times
from sparse_event_vpr.utils import (
    load_event_streams,
    get_size,
    print_duration,
    create_event_frames,
    get_conventional_frames,
    sync_event_streams,
    remove_random_bursts,
    get_short_traverse_name,
    str_to_bool,
    none_or_int,
    get_times_for_streams_const_time,
    get_times_for_streams_const_count,
    check_parameters,
)
from sparse_event_vpr.sparse_pixel_utils import (
    load_gps_gt,
    create_frames_subset,
    compute_distance_matrices,
    adjust_and_normalize_probabilities,
    get_random_pixels,
    create_event_frames_sparse_pixels,
    get_times_for_subsets,
    evaluate_pr,
)
from sparse_event_vpr.plots import (
    plot_distance_matrix,
    plot_gt_percentage,
    plot_gt_gps,
    plot_pixel_selection,
    plot_mean_number_of_events,
    plot_pr_curves,
    setup_plot_params,
)


if __name__ == "__main__":
    np.seterr(all="raise")
    freeze_support()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        tqdm.write("Use CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        tqdm.write("Use Mac GPU")
    else:
        device = torch.device("cpu")
        tqdm.write("Use CPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seq_length", default=[10], nargs="+", help="Sequence length(s) for sequence matcher (default: 10)")
    parser.add_argument("-t", "--num_target_pixels", type=int, default=100, help="How many pixels should be selected? (default: 100)")
    parser.add_argument("-l", "--event_frame_length_all_pixels", type=none_or_int, default=1e6, help="How long (in μs) should the event frame be when using all pixels? (default: 1000000.0)")
    parser.add_argument("-lf", "--event_frame_length_sparse_pixels", type=none_or_int, default=1e6, help="How long (in μs) should the event frame be when using sparse pixels? (default: 1000000.0)")
    parser.add_argument("-ec", "--event_frame_count_all_pixels", type=none_or_int, default=None, help="How many events should an event frame have?")
    parser.add_argument("-mf", "--multiplier_factor", type=none_or_int, default=None, help="How many events should be in a frame when using sparse pixels? Will be multiplied by the number of target pixels (good number: 3)")
    parser.add_argument("-gtp", "--gt_tolerance_percentage", type=int, default=2, help="What is the GT tolerance (in percent); QCR-Event-VPR dataset? (default: 2)")
    parser.add_argument("-gtm", "--gt_tolerance_meters", type=int, default=70, help="What is the GT tolerance (in meters); Brisbane-Event-VPR dataset? (default: 70)")
    parser.add_argument("-us", "--use_saliency", type=str_to_bool, nargs="?", const=True, default=True, help="Use mean/variance in event pixels to select pixels")
    parser.add_argument("-uf", "--use_conventional_frames", type=str_to_bool, nargs="?", const=True, default=False, help="Use DAVIS APS frames for comparison")
    parser.add_argument("-rt", "--ref_traverse", type=str, default="bags_2021-08-19-08-25-42_denoised.parquet")
    parser.add_argument("-qt", "--qry_traverse", type=str, default="bags_2021-08-19-08-28-43_denoised.parquet")

    args = parser.parse_args()

    setup_plot_params(font_scale=1.2)

    if isinstance(args.seq_length[0], str):
        args.seq_length = [int(x) for x in args.seq_length[0].split(" ")]  # noqa

    traverses_to_compare = [args.ref_traverse, args.qry_traverse]

    gps_gt = load_gps_gt(traverses_to_compare)

    path_to_outputs = os.path.dirname(os.path.abspath(__file__)) + "/../results/" + get_short_traverse_name(traverses_to_compare[0]) + "_" + get_short_traverse_name(traverses_to_compare[1]) + "/"
    os.makedirs(path_to_outputs, exist_ok=True)
    tqdm.write(f"Results dir: {path_to_outputs}")

    gt_tolerance, event_frame_length_all_pixels, event_frame_length_sparse_pixels, event_frame_num_sparse_pixels, suffix_no_seq = check_parameters(args, traverses_to_compare)

    with Timer(text="Load event streams: {:0.2f} seconds"):
        event_streams = load_event_streams(traverses_to_compare)
    with Timer(text="Sync event streams: {:0.2f} seconds"):
        event_streams = sync_event_streams(event_streams, traverses_to_compare, gps_gt)

    im_width, im_height = get_size(event_streams[0])
    sensor_size = (im_width, im_height, 2)

    dtype_pixel_indices = np.dtype([("t", np.uint64), ("x", np.uint16), ("p", np.uint8)])

    for event_stream in event_streams: print_duration(event_stream)

    event_streams_numpy_raw = [event_stream.to_numpy(np.uint64) for event_stream in event_streams]

    with Timer(text="Create event frames: {:0.2f} seconds"):
        event_frames_total = create_event_frames(event_streams, sensor_size,
            frame_length=event_frame_length_all_pixels, event_count=args.event_frame_count_all_pixels)

    if event_frame_length_all_pixels is not None:
        tqdm.write("get_times_for_streams_const_time")
        ref_times_total = get_times_for_streams_const_time(event_streams[0], event_frame_length_all_pixels) / 1e6
        qry_times_total = get_times_for_streams_const_time(event_streams[1], event_frame_length_all_pixels) / 1e6
    else:
        tqdm.write("get_times_for_streams_const_count")
        ref_times_total = get_times_for_streams_const_count(event_streams[0], args.event_frame_count_all_pixels, 0) / 1e6
        qry_times_total = get_times_for_streams_const_count(event_streams[1], args.event_frame_count_all_pixels, 0) / 1e6

    event_frames_no_bursts = [remove_random_bursts(event_frames, threshold=10) for event_frames in event_frames_total]

    event_means = [event_frame_total.mean(axis=0) for event_frame_total in event_frames_no_bursts]
    event_variances = [event_frame_total.var(axis=0) for event_frame_total in event_frames_no_bursts]

    tqdm.write(f"Ref traverse shape: {event_frames_no_bursts[0].shape}")
    tqdm.write(f"Qry traverse shape: {event_frames_no_bursts[1].shape}")

    if args.use_conventional_frames:
        with Timer(name="Load conventional frames"):
            conventional_frames = [get_conventional_frames(traverses_to_compare[0], ref_times_total + event_streams[0]["t"].iloc[0] / 1e6),
                                  get_conventional_frames(traverses_to_compare[1], qry_times_total + event_streams[1]["t"].iloc[0] / 1e6),]
        conventional_frames_means = [coventional_frame.mean(axis=0) for coventional_frame in conventional_frames]
        conventional_frames_variances = [coventional_frame.var(axis=0) for coventional_frame in conventional_frames]
        tqdm.write(f"Ref traverse (conventional frames) shape: {conventional_frames[0].shape}")
        tqdm.write(f"Qry traverse (conventional frames) shape: {conventional_frames[1].shape}")

    if args.use_saliency:
        tqdm.write("Get salient pixels (saliency)")
        prob_to_draw_from = adjust_and_normalize_probabilities(event_means[0])
    else:
        tqdm.write("Get random pixels (no saliency)")
        prob_to_draw_from = None

    # they are y, x!
    random_pixels = get_random_pixels(args.num_target_pixels, im_width=im_width, im_height=im_height, local_suppression_radius=7, prob_to_draw_from=prob_to_draw_from)
    tqdm.write(f"Got {args.num_target_pixels} random pixels")
    plot_pixel_selection(random_pixels, path_to_outputs, suffix_no_seq + "_ref", event_means[0])
    plot_pixel_selection(random_pixels, path_to_outputs, suffix_no_seq + "_qry", event_means[1])

    if args.use_conventional_frames:
        if args.use_saliency:
            tqdm.write("Get salient pixels for conventional frames (saliency)")
            prob_to_draw_from_conventional_frames = adjust_and_normalize_probabilities(conventional_frames_variances[0])
        else:
            tqdm.write("Get random pixels for conventional frames (no saliency)")
            prob_to_draw_from_conventional_frames = None

        random_pixels_coventional_frames = get_random_pixels(args.num_target_pixels, im_width=im_width, im_height=im_height, local_suppression_radius=7,
                                                             prob_to_draw_from=prob_to_draw_from_conventional_frames)

        plot_pixel_selection(random_pixels_coventional_frames, path_to_outputs, suffix_no_seq + "conventional_frames_var_ref", conventional_frames_variances[0])
        plot_pixel_selection(random_pixels_coventional_frames, path_to_outputs, suffix_no_seq + "conventional_frames_var_qry", conventional_frames_variances[1])
    else:
        tqdm.write("Do not use conventional frames for comparison")

    if event_frame_num_sparse_pixels is not None:
        tqdm.write("Create event frames sparse pixels")
        with Timer(text="Elapsed time create event frames sparse pixels ref: {:0.2f} seconds"):
            ref_event_frames_subset, ref_time_slices_subset = create_event_frames_sparse_pixels(
                [event_streams_numpy_raw[0]], random_pixels, event_frame_length_sparse_pixels,
                event_frame_num_sparse_pixels, dtype_pixel_indices, im_width)

        with Timer(text="Elapsed time create event frames sparse pixels qry: {:0.2f} seconds"):
            qry_event_frames_subset, qry_time_slices_subset = create_event_frames_sparse_pixels(
                [event_streams_numpy_raw[1]], random_pixels, event_frame_length_sparse_pixels,
                event_frame_num_sparse_pixels, dtype_pixel_indices, im_width,)

        ref_times_subsets = get_times_for_subsets(ref_time_slices_subset)
        qry_times_subsets = get_times_for_subsets(qry_time_slices_subset)

        event_frames_subsets = [ref_event_frames_subset, qry_event_frames_subset]
        event_frames_subsets = [remove_random_bursts(event_frames, threshold=10) for event_frames in event_frames_subsets]
    else:
        ref_times_subsets = None
        qry_times_subsets = None
        ref_event_frames_subset = create_frames_subset(event_frames_total[0], random_pixels)
        qry_event_frames_subset = create_frames_subset(event_frames_total[1], random_pixels)
        event_frames_subsets = [ref_event_frames_subset, qry_event_frames_subset]

    plot_mean_number_of_events(event_frames_no_bursts, event_frames_subsets, ref_times_total, qry_times_total,
                               ref_times_subsets, qry_times_subsets, args.num_target_pixels, path_to_outputs, suffix_no_seq)

    if args.use_conventional_frames:
        ref_conventional_frames_subset = create_frames_subset(conventional_frames[0], random_pixels_coventional_frames)
        qry_conventional_frames_subset = create_frames_subset(conventional_frames[1], random_pixels_coventional_frames)
        conventional_frames_subsets = [ref_conventional_frames_subset, qry_conventional_frames_subset]
        plot_mean_number_of_events(conventional_frames, conventional_frames_subsets, ref_times_total, qry_times_total, ref_times_total, qry_times_total,
                                   args.num_target_pixels, path_to_outputs, suffix_no_seq + "_conventional")

    for seq_length in tqdm(args.seq_length, desc="Sequence length"):
        tqdm.write(f"Compute for sequence length of {seq_length}")
        suffix = suffix_no_seq + f"_seqlen{seq_length}"

        # Define the frame sets
        frames_sets = {
            "all_pixels": (event_frames_no_bursts[0], event_frames_no_bursts[1]),
            "subset": (event_frames_subsets[0], event_frames_subsets[1])
        }

        # Add conventional frames if applicable
        if args.use_conventional_frames:
            frames_sets.update({
                "conventional_frames": (conventional_frames[0], conventional_frames[1]),
                "conventional_subset": (conventional_frames_subsets[0], conventional_frames_subsets[1])
            })

        # Compute the distance matrices
        dist_matrix_dict = compute_distance_matrices(frames_sets, device, seq_length)

        if traverses_to_compare[0] in gt_times and traverses_to_compare[1] in gt_times:
            tqdm.write("Use manually annotated ground truth")
            dist_to_gt_dict = plot_gt_percentage(traverses_to_compare, ref_times_subsets, qry_times_subsets, ref_times_total,
                                                 qry_times_total, seq_length, dist_matrix_dict)
        elif gps_gt:
            tqdm.write("Use GPS ground truth")
            dist_to_gt_dict = plot_gt_gps(gps_gt, ref_times_subsets, qry_times_subsets, ref_times_total, qry_times_total, seq_length, dist_matrix_dict)
        else:
            raise ValueError("No ground truth provided")

        dist_matrix_plot_dict = {
            "dm_all_pixels": plot_distance_matrix(dist_matrix_dict["all_pixels"], "event camera all pixels; no seq"),
            "dm_all_pixels_seq": plot_distance_matrix(dist_matrix_dict["all_pixels_seq"], f"event camera all pixels; seqlen={seq_length}"),
            "dm_subset": plot_distance_matrix(dist_matrix_dict["subset"], f"event camera {args.num_target_pixels} pixels; no seq"),
            "dm_subset_seq": plot_distance_matrix(dist_matrix_dict["subset_seq"], f"event camera {args.num_target_pixels} pixels; seqlen={seq_length} via subset"),
        }

        if "conventional_frames" in dist_matrix_dict:
            dist_matrix_plot_dict["dm_conventional_frames"] = plot_distance_matrix(dist_matrix_dict["conventional_frames"], f"conventional frames; no seq")
            dist_matrix_plot_dict["dm_conventional_frames_seq"] = plot_distance_matrix(dist_matrix_dict["conventional_frames_seq"], f"conventional frames; seqlen={seq_length}")
            dist_matrix_plot_dict["dm_conventional_subset"] = plot_distance_matrix(dist_matrix_dict["conventional_subset"], f"conventional subset; no seq")
            dist_matrix_plot_dict["dm_conventional_subset_seq"] = plot_distance_matrix(dist_matrix_dict["conventional_subset_seq"], f"conventional subset; seqlen={seq_length}")

        pr_curves_xs, pr_curves_ys, pr_curves_keys, recall_99p_dict, precision_100r_dict, auc_dict = evaluate_pr(dist_matrix_dict, dist_to_gt_dict, gt_tolerance)

        plot_pr_curves(pr_curves_xs, pr_curves_ys, pr_curves_keys, path_to_outputs, suffix, args.ref_traverse, args.qry_traverse, args.num_target_pixels, auc_dict)

    tqdm.write("Bye")
