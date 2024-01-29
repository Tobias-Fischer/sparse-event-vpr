from setuptools import find_packages, setup

setup(
    name="sparse_event_vpr",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    version="1.0.0",
    author="Tobias Fischer",
    author_email="tobias.fischer@qut.edu.au",
    python_requires=">=3.6",
    entry_points={
        "console_scripts": ["sparse-event-vpr=sparse_event_vpr:main"],
    },
    install_requires=[
        "pytorch",
        "codetiming",
        "tqdm",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "plotly",
        "numba",
        "git+https://github.com/Tobias-Fischer/tonic.git",
        "pynmea2",
        "seaborn",
        "opencv-python",
        "pypng",
    ],
)
