import os
import shutil
import tarfile
import urllib.request
from tqdm import tqdm

from score_sde.utils import register_dataset

import geomstats as gs
import jax.numpy as jnp

data_dir = "/data/localhost/not-backed-up/mhutchin/score-sde/data/abdb"

_dataset_url = "http://www.abybank.org/abdb/Data/LH_Combined_Martin.tar.bz2"
_redundant_antibody_url = (
    "http://www.abybank.org/abdb/Data/Redundant_files/Redundant_H_Combined_Martin.txt"
)
_free_antibody_complex_list = (
    "http://www.abybank.org/abdb/Data/Martin_logs/Heavy_HeavyAntigen.list"
)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def clean_files():
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)


def download_files():
    os.makedirs(data_dir, exist_ok=True)

    download_url(_dataset_url, os.path.join(data_dir, "raw_data.tar.bz2"))
    download_url(
        _redundant_antibody_url,
        os.path.join(data_dir, "redundant_antibodies.txt"),
    )
    download_url(
        _free_antibody_complex_list,
        os.path.join(data_dir, "free_antibody_complex_list.txt"),
    )


def extract_files():
    tar = tarfile.open(os.path.join(data_dir, "raw_data.tar.bz2"), "r:bz2")
    tar.extractall(data_dir)
    os.rename(
        os.path.join(data_dir, "LH_Combined_Martin"),
        os.path.join(data_dir, "raw_antibodies"),
    )


clean_files()
download_files()
extract_files()
