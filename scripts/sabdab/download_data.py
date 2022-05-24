import os
import shutil
import zipfile
import urllib.request
from tqdm import tqdm

import geomstats as gs
import jax.numpy as jnp

data_dir = "/data/localhost/not-backed-up/mhutchin/score-sde/data/sabdab"

_dataset_url = "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"


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

    download_url(_dataset_url, os.path.join(data_dir, "all_structures.zip"))


def extract_files():
    with zipfile.ZipFile(os.path.join(data_dir, "all_structures.zip"), "r") as zip_ref:
        zip_ref.extractall(os.path.join(data_dir))


clean_files()
download_files()
extract_files()
