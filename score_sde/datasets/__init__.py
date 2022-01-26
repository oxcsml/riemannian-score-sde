import tensorflow as tf
import tensorflow_datasets as tfds

from score_sde.utils import register_dataset


from .mixture import *
from .unimodal import *
from .tensordataset import *
from .split import *


def create_prefetch_dataset(
    dataset_builder,
    split,
    num_epochs,
    batch_dims,
    preprocess_fn,
    shuffle_buffer_size=10000,
    prefetch_size=tf.data.experimental.AUTOTUNE,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
        dataset_builder.download_and_prepare()
        ds = dataset_builder.as_dataset(
            split=split, shuffle_files=True, read_config=read_config
        )
    else:
        ds = dataset_builder.with_options(dataset_options)
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
    for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)
