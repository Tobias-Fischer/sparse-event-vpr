# How Many Events Do You Need? Event-based Visual Place Recognition Using Sparse But Varying Pixels.

This repository contains code for our paper "How Many Events Do You Need? Event-based Visual Place Recognition Using Sparse But Varying Pixels".

If you use this code, please refer to our [paper](https://doi.org/10.1109/LRA.2022.3216226):
```bibtex
@article{FischerRAL2022ICRA2023,
    title={How Many Events do You Need? Event-based Visual Place Recognition Using Sparse But Varying Pixels},
    author={Tobias Fischer and Michael Milford},
    journal={IEEE Robotics and Automation Letters},
    volume={7},
    number={4},
    pages={12275--12282},
    year={2022},
    doi={10.1109/LRA.2022.3216226},
}
```

## QCR-Event-VPR Dataset
The associated QCR-Event-VPR dataset can be found on [Zenodo](https://zenodo.org/records/10494919). The code can also handle data from our previous [Brisbane-Event-VPR dataset](https://huggingface.co/datasets/TobiasRobotics/brisbane-event-vpr/tree/main).

Please download the dataset, and place the `parquet` files into the `./data/input_parquet_files` directory.
If you want to work with the DAVIS conventional frames, please download the `zip` files, and extract them so that an image files is located in e.g. `./data/input_frames/bags_2021-08-19-08-25-42/frames/1629325542.939281225204.png`.
For the Brisbane-Event-VPR dataset, place the `nmea` files into the `./data/gps_data` directory.

## Install dependencies
We recommend using `conda` (in particular, `mamba`, which can be installed via [Miniforge3](https://github.com/conda-forge/miniforge#miniforge3):
```bash
mamba create -n sparse-event-vpr pip pytorch codetiming tqdm pandas numpy scipy matplotlib seaborn numba pynmea2 opencv pypng h5py importrosbag pbr pyarrow fastparquet
mamba activate sparse-event-vpr
pip install git+https://github.com/Tobias-Fischer/tonic.git@develop --no-deps
```

## Usage
First, install the package via
```
pip install -e .  # you need to run this command inside the `sparse-event-vpr` directory
```

The main script file is `perform_sparse_event_vpr`. You can run it with Python and see all options that are exposed:
```
python ./scripts/perform_sparse_event_vpr.py --help
```
