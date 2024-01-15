import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate
import os
import shutil

from .constants import gt_times, gt_percentage_travelled
from .utils import interpolate_gps, get_short_traverse_name, get_traverse_alias


def setup_plot_params(font_scale=1.0):
    sns.set_context("paper", font_scale=font_scale)
    latex_cmd = shutil.which("latex")
    if latex_cmd is not None:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.sans-serif": ["Helvetica"]})
        custom_preamble = {
            "text.latex.preamble": r"\usepackage{amsmath}",  # for the align, center,... environment
        }
        plt.rcParams.update(custom_preamble)
    else:
        print("WARNING: LaTeX was not found")


def plot_distance_matrix(dMat, plot_title=None, plot_crosses=True):
    fig, ax = plt.subplots()
    cax = ax.imshow(dMat, cmap='viridis', interpolation='nearest')
    fig.colorbar(cax, ax=ax, label='Distance')

    ax.set_xlabel("Query number")
    ax.set_ylabel("Reference number")

    if plot_crosses:
        min_indices = dMat.argmin(axis=0)
        ax.scatter(np.arange(dMat.shape[1]), min_indices, color='LightSkyBlue', marker='x', alpha=0.5, s=2)

    if plot_title is not None:
        ax.set_title(plot_title)

    return fig


def plot_distance_to_gt(dist_matrix_dict, ref_percentages, query_percentages, key, title_suffix, is_sequence=False):
    if is_sequence:
        # Adjust the key and title for sequence data
        key += "_seq"
        title_suffix += " (sequence)"

    match_percentages = ref_percentages[dist_matrix_dict[key].argmin(axis=0)]
    if len(query_percentages) - len(match_percentages) == 1:
        query_percentages = query_percentages[:-1]

    dist_to_gt = np.abs(query_percentages[: match_percentages.shape[0]] - match_percentages)

    plt.figure()
    plt.plot(query_percentages[: match_percentages.shape[0]], dist_to_gt, label="Distance to GT over time")
    plt.xlabel('Query')
    plt.ylabel('Distance to GT')
    plt.title(f'Distance to GT over time ({title_suffix})')

    return dist_to_gt


def plot_gt_percentage(traverses_to_compare, ref_times_subsets, query_times_subsets, ref_times_total, query_times_total, seq_length, dist_matrix_dict):
    f_ref_time_to_percent = interpolate.interp1d(gt_times[traverses_to_compare[0]], gt_percentage_travelled[traverses_to_compare[0]], fill_value="extrapolate")
    f_query_time_to_percent = interpolate.interp1d(gt_times[traverses_to_compare[1]], gt_percentage_travelled[traverses_to_compare[1]], fill_value="extrapolate")

    print(gt_times[traverses_to_compare[0]])
    print(gt_percentage_travelled[traverses_to_compare[0]])

    print(gt_times[traverses_to_compare[1]])
    print(gt_percentage_travelled[traverses_to_compare[1]])

    dist_to_gt_dict = {}

    if ref_times_subsets is not None and query_times_subsets is not None:
        ref_percentages_subset_seq = np.array([f_ref_time_to_percent(ref_time) for ref_time in ref_times_subsets])
        query_percentages_subset_seq = np.array([f_query_time_to_percent(query_time) for query_time in query_times_subsets])

        if seq_length == 2:
            ref_percentages_subset_seq = ref_percentages_subset_seq[seq_length // 2 :]
            query_percentages_subset_seq = query_percentages_subset_seq[seq_length // 2 :]
        elif seq_length > 2:
            ref_percentages_subset_seq = ref_percentages_subset_seq[seq_length // 2 : -(seq_length // 2) + 1]
            query_percentages_subset_seq = query_percentages_subset_seq[seq_length // 2 : -(seq_length // 2) + 1]

        dist_to_gt_dict["subset_seq"] = plot_distance_to_gt(dist_matrix_dict, ref_percentages_subset_seq, query_percentages_subset_seq, "subset", "sparse pixels", is_sequence=True)

    ref_percentages_total = np.array([f_ref_time_to_percent(ref_time) for ref_time in ref_times_total])
    query_percentages_total = np.array([f_query_time_to_percent(query_time) for query_time in query_times_total])

    dist_to_gt_dict["all_pixels"] = plot_distance_to_gt(dist_matrix_dict, ref_percentages_total, query_percentages_total, "all_pixels", "all pixels")

    if "conventional_frames" in dist_matrix_dict:
        dist_to_gt_dict["conventional_frames"] = plot_distance_to_gt(dist_matrix_dict, ref_percentages_total, query_percentages_total, "conventional_frames", "conventional frames all pixels")
        dist_to_gt_dict["conventional_subset"] = plot_distance_to_gt(dist_matrix_dict, ref_percentages_total, query_percentages_total, "conventional_subset", "conventional frames sparse pixels")

    query_percentages_total_seq = query_percentages_total[seq_length // 2 :]
    ref_percentages_total_seq = ref_percentages_total[seq_length // 2 :]

    dist_to_gt_dict["all_pixels_seq"] = plot_distance_to_gt(dist_matrix_dict, ref_percentages_total_seq, query_percentages_total_seq, "all_pixels", "all pixels", is_sequence=True)

    if "conventional_frames_seq" in dist_matrix_dict:
        dist_to_gt_dict["conventional_frames_seq"] = plot_distance_to_gt(dist_matrix_dict, ref_percentages_total_seq, query_percentages_total_seq, "conventional_frames", "conventional frames all pixels", is_sequence=True)
        dist_to_gt_dict["conventional_subset_seq"] = plot_distance_to_gt(dist_matrix_dict, ref_percentages_total_seq, query_percentages_total_seq, "conventional_subset", "conventional frames sparse pixels", is_sequence=True)

    return dist_to_gt_dict


def plot_gt_gps(gps_gt, ref_times_subsets, query_times_subsets, ref_times_total, query_times_total, seq_length, dist_matrix_dict):
    # Initialize dictionary to store distances to ground truth
    dist_to_gt_dict = {}

    # Define the types of plots and corresponding parameters
    plot_types = {
        "subset": (dist_matrix_dict["subset"], query_times_subsets, ref_times_subsets, None),
        "subset_seq": (dist_matrix_dict["subset_seq"], query_times_subsets, ref_times_subsets, seq_length),
        "all_pixels": (dist_matrix_dict["all_pixels"], query_times_total, ref_times_total, None),
        "all_pixels_seq": (dist_matrix_dict["all_pixels_seq"], query_times_total, ref_times_total, seq_length)
    }

    # Add conventional frames if they exist
    if "conventional_frames" in dist_matrix_dict:
        plot_types.update({
            "conventional_frames": (dist_matrix_dict["conventional_frames"], query_times_total, ref_times_total, None),
            "conventional_frames_seq": (dist_matrix_dict["conventional_frames_seq"], query_times_total, ref_times_total, seq_length),
            "conventional_subset": (dist_matrix_dict["conventional_subset"], query_times_total, ref_times_total, None),
            "conventional_subset_seq": (dist_matrix_dict["conventional_subset_seq"], query_times_total, ref_times_total, seq_length)
        })

    # Iterate over each plot type and create the plots
    for plot_key, (matrix, query_times, ref_times, length) in plot_types.items():
        plot_title = f"manual_gt_plot_{plot_key}"
        dist_to_gt_dict[plot_key] = plot_gt_gps_individual(matrix, gps_gt, query_times, ref_times, length, plot_title)

    return dist_to_gt_dict


def plot_gt_gps_individual(dist_matrix, gps_gt, query_times, ref_times, seq_length=None, log_name="manual_gt_plot"):
    ref_gps = interpolate_gps(gps_gt[0], ref_times)[:, :2]
    query_gps = interpolate_gps(gps_gt[1], query_times)[:, :2]

    if seq_length is not None:
        ref_gps = ref_gps[seq_length // 2 :]
        query_gps = query_gps[seq_length // 2 :]
    match_gps = ref_gps[dist_matrix.argmin(axis=0)]
    dist_to_gt = np.linalg.norm(query_gps[: match_gps.shape[0], :2] - match_gps[:, :2], axis=1) * 100000

    if log_name is not None:
        plt.figure()
        plt.plot(range(len(dist_to_gt)), dist_to_gt, label="Distance to GT over time")

    return dist_to_gt


def plot_pixel_selection(random_pixels, path_to_outputs, suffix="", event_means=None):
    plt.figure(figsize=(3.5, 2))
    plt.title("Sparse pixel selection")
    if event_means is not None:
        sns.heatmap(event_means, cmap="Reds", rasterized=True)
    plt.plot(np.array(random_pixels)[:, 1], np.array(random_pixels)[:, 0], ".")
    plt.savefig(os.path.join(path_to_outputs, f"salient_pixel_selection{suffix}.pdf"), bbox_inches="tight", dpi=300)
    plt.show()


def plot_mean_number_of_events(
    event_frames_all_pixels, event_frames_subsets, ref_times_total, qry_times_total, ref_times_subsets, qry_times_subsets, num_target_pixels, path_to_outputs, suffix="", xlabel="Time (s)"
):
    plt.figure(figsize=(5, 1.8))
    if ref_times_subsets is None:
        print("Debug: use ref_times_total for ref_times_subsets")
        ref_times_subsets = ref_times_total
    if qry_times_subsets is None:
        print("Debug: use qry_times_total for qry_times_subsets")
        qry_times_subsets = qry_times_total
    plt.plot(ref_times_total, event_frames_all_pixels[0].mean(axis=(1, 2)), label="All pixels")
    plt.plot(ref_times_subsets, event_frames_subsets[0].mean(axis=1), label=f"{num_target_pixels} pixels")
    plt.xlabel(xlabel)
    plt.ylabel("Mean number of events")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    plt.savefig(os.path.join(path_to_outputs, f"events_profile{suffix}_{xlabel}_sub_diff.pdf"), bbox_inches="tight")
    plt.show()


def plot_pr_curves(xs_list, ys_list, label_list, path_to_outputs, suffix="", ref_traverse="", qry_traverse="", num_pixels=0, auc_dict=None):
    plt.figure(figsize=(2.5, 2.5 / 1.618))
    print(auc_dict.keys())

    ref_alias = get_traverse_alias(get_short_traverse_name(ref_traverse))
    qry_alias = get_traverse_alias(get_short_traverse_name(qry_traverse))

    for xs, ys, label in zip(xs_list, ys_list, label_list):
        if "subset_seq_ratio" in label:
            plt.plot(xs, ys, color='#76433A', label=rf"Proposed ({num_pixels} pixels)")
        elif "all_pixels_seq_ratio" in label:
            plt.plot(xs, ys, color='#7E51B2', label=rf"Milford et al. R:SSW")
        elif "conventional_frames_seq_ratio" in label:
            plt.plot(xs, ys, color='pink', label=rf"Conventional frames")
        elif "conventional_subset_seq_ratio" in label:
            plt.plot(xs, ys, color='black', label=rf"Conventional frames ({num_pixels} pixels)")
        else:
            pass

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(rf"\begin{{center}}Reference: {ref_alias}\\Query: {qry_alias}\end{{center}}")
    plt.savefig(os.path.join(path_to_outputs, f"pr_curves{suffix}_with_legend.pdf"), bbox_inches="tight")
    plt.show()
