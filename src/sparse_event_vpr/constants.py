import os

brisbane_event_traverses = [
    "dvs_vpr_2020-04-21-17-03-03_no_hot_pixels_nobursts_denoised.parquet",  # sunset1
    "dvs_vpr_2020-04-22-17-24-21_no_hot_pixels_nobursts_denoised.parquet",  # sunset2
    "dvs_vpr_2020-04-24-15-12-03_no_hot_pixels_nobursts_denoised.parquet",  # daytime
    "dvs_vpr_2020-04-28-09-14-11_no_hot_pixels_nobursts_denoised.parquet",  # morning
    "dvs_vpr_2020-04-29-06-20-23_no_hot_pixels_nobursts_denoised.parquet",  # sunrise
]

brisbane_event_traverses_aliases = {
    "dvs_vpr_2020-04-21-17-03-03": "Sunset 1",
    "dvs_vpr_2020-04-22-17-24-21": "Sunset 2",
    "dvs_vpr_2020-04-24-15-12-03": "Daytime",
    "dvs_vpr_2020-04-28-09-14-11": "Morning",
    "dvs_vpr_2020-04-29-06-20-23": "Sunrise",
}

video_beginning = {
    "dvs_vpr_2020-04-21-17-03-03": 1587452593.35,
    "dvs_vpr_2020-04-22-17-24-21": 1587540271.65,
    "dvs_vpr_2020-04-24-15-12-03": 1587705136.80,
    "dvs_vpr_2020-04-28-09-14-11": 1588029271.73,
    "dvs_vpr_2020-04-29-06-20-23": 1588105240.91,
}

qcr_traverses = [
    "bags_2021-08-19-08-25-42_denoised.parquet",  # S11 side-facing, normal
    "bags_2021-08-19-08-28-43_denoised.parquet",  # S11 side-facing, normal
    "bags_2021-08-19-09-45-28_denoised.parquet",  # S11 side-facing, normal
    "bags_2021-08-20-10-19-45_denoised.parquet",  # S11 side-facing, fast
    "bags_2022-03-28-11-51-26_denoised.parquet",  # S11 side-facing, normal
    "bags_2022-03-28-12-01-42_denoised.parquet",  # S11 side-facing, fast
    "bags_2022-03-28-12-03-44_denoised.parquet",  # S11 side-facing, slow
]

qcr_traverses_aliases = {
    "bags_2021-08-19-08-25-42": "Traverse 1 (normal speed)",
    "bags_2021-08-19-08-28-43": "Traverse 2 (normal speed)",
    "bags_2021-08-19-09-45-28": "Traverse 3 (normal speed)",
    "bags_2022-03-28-11-51-26": "Traverse 4 (normal speed)",
    "bags_2022-03-28-12-01-42": "Traverse 5 (fast)",
    "bags_2021-08-20-10-19-45": "Traverse 6 (fast)",
    "bags_2022-03-28-12-03-44": "Traverse 7 (slow)",
}

gt_times = {
    "bags_2021-08-19-08-25-42_denoised.parquet": [0, 6, 13, 31, 58, 77, 101.5, 124, 148, 157.7, 163.0],
    "bags_2021-08-19-08-28-43_denoised.parquet": [0, 6, 13, 31.5, 58, 76, 100.5, 123, 148.5, 156.5, 162.0],
    "bags_2021-08-19-09-45-28_denoised.parquet": [0, 6.5, 13.5, 31.5, 58.5, 77.5, 102.5, 126, 150, 158.5, 165.0],
    "bags_2021-08-20-10-19-45_denoised.parquet": [0, 2.5, 5.0, 12.7, 23.7, 33.6, 43.0, 52.4, 62.3, 65.4, 68],
    "bags_2022-03-28-11-51-26_denoised.parquet": [0, 6.7, 13.2, 31, 57, 74, 97, 119, 141, 148, 154],
    "bags_2022-03-28-12-01-42_denoised.parquet": [0, 2.9, 5.8, 12.9, 23.5, 32, 41.5, 50, 59.3, 62, 64.5],
    "bags_2022-03-28-12-03-44_denoised.parquet": [0, 15.5, 30.0, 63.5, 110.0, 141.0, 185.0, 217.0, 246.5, 256.0, 263.0],
}

qcr_traverses_first_times = {
    "bags_2021-08-19-08-25-42": 2e6,
    "bags_2021-08-19-09-45-28": 2e6,
    "bags_2022-03-28-11-51-26": 8e6,
    "bags_2022-03-28-12-01-42": 7.7e6,
    "bags_2022-03-28-12-03-44": 12e6,
}

qcr_traverses_last_times = {
    "bags_2021-08-19-08-28-43": gt_times["bags_2021-08-19-08-28-43_denoised.parquet"][-1] * 1e6,
    "bags_2021-08-20-10-19-45": gt_times["bags_2021-08-20-10-19-45_denoised.parquet"][-1] * 1e6,
    "bags_2021-08-19-08-25-42": qcr_traverses_first_times["bags_2021-08-19-08-25-42"] + gt_times["bags_2021-08-19-08-25-42_denoised.parquet"][-1] * 1e6,
    "bags_2021-08-19-09-45-28": qcr_traverses_first_times["bags_2021-08-19-09-45-28"] + gt_times["bags_2021-08-19-09-45-28_denoised.parquet"][-1] * 1e6,
    "bags_2022-03-28-11-51-26": qcr_traverses_first_times["bags_2022-03-28-11-51-26"] + gt_times["bags_2022-03-28-11-51-26_denoised.parquet"][-1] * 1e6,
    "bags_2022-03-28-12-01-42": qcr_traverses_first_times["bags_2022-03-28-12-01-42"] + gt_times["bags_2022-03-28-12-01-42_denoised.parquet"][-1] * 1e6,
    "bags_2022-03-28-12-03-44": qcr_traverses_first_times["bags_2022-03-28-12-03-44"] + gt_times["bags_2022-03-28-12-03-44_denoised.parquet"][-1] * 1e6,
}

gt_percentage_travelled = {}
for traverse, traverse_gt_times in gt_times.items():
    gt_percentage_travelled[traverse] = [(gt_time / traverse_gt_times[-1]) * 100 for gt_time in traverse_gt_times]

path_to_event_files = os.path.dirname(os.path.abspath(__file__)) + "/../../data/input_parquet_files/"
path_to_frames = os.path.dirname(os.path.abspath(__file__)) + "/../../data/input_frames/"
path_to_gps_files = os.path.dirname(os.path.abspath(__file__)) + "/../../data/gps_data/"
