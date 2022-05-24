import os
import shutil
import tarfile
import urllib.request
from tqdm import tqdm

from score_sde.utils import register_dataset

import geomstats as gs
import jax.numpy as jnp


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


class AbBd:
    _dataset_url = "http://www.abybank.org/abdb/Data/H_Combined_Martin.tar.bz2"
    _redundant_antibody_url = "http://www.abybank.org/abdb/Data/Redundant_files/Redundant_H_Combined_Martin.txt"
    _free_antibody_complex_list = (
        "http://www.abybank.org/abdb/Data/Martin_logs/Heavy_HeavyAntigen.list"
    )

    def __init__(self, data_dir="data", redownload=False):
        self.data_dir = os.path.join(data_dir, "abdb")

        if not os.path.isdir(self.data_dir) or redownload:
            self.clean_files()
            self.download_files()
            self.extract_files()

    def clean_files(self):
        if os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir)

    def download_files(self):
        os.makedirs(self.data_dir, exist_ok=True)

        download_url(self._dataset_url, os.path.join(self.data_dir, "raw_data.tar.bz2"))
        download_url(
            self._redundant_antibody_url,
            os.path.join(self.data_dir, "redundant_antibodies.txt"),
        )
        download_url(
            self._free_antibody_complex_list,
            os.path.join(self.data_dir, "free_antibody_complex_list.txt"),
        )

    def extract_files(self):
        tar = tarfile.open(os.path.join(self.data_dir, "raw_data.tar.bz2"), "r:bz2")
        tar.extractall(self.data_dir)
        os.rename(
            os.path.join(self.data_dir, "H_Combined_Martin"),
            os.path.join(self.data_dir, "raw_antibodies"),
        )
