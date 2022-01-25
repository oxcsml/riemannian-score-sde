from math import prod, floor

import jax
import jax.numpy as jnp
import numpy as np


class TensorDataset:
    def __init__(self, data, batch_dims, rng, shuffle=True, drop_last=False):

        self.data = jnp.array(data)
        if isinstance(batch_dims, int):
            batch_dims = (batch_dims,)
        self.batch_dims = batch_dims
        self.rng, self.iterator_rng = jax.random.split(rng)

        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        bs = prod(self.batch_dims)
        N = floor(self.data.shape[0] / bs)
        return N if self.drop_last else N + 1

    def __iter__(self):
        return TensorDatasetIterator(self)

    def __next__(self):
        rng, next_rng = jax.random.split(self.rng)
        self.rng = rng

        indices = jax.random.choice(
            next_rng, self.data.shape[0], shape=(prod(self.batch_dims),)
        )

        return self.data[indices].reshape((*self.batch_dims, *self.data.shape[1:]))


class TensorDatasetIterator:
    def __init__(self, dataset: TensorDataset):
        self.dataset = dataset
        rng, self.dataset.iterator_rng = jax.random.split(self.dataset.iterator_rng)
        if self.dataset.shuffle:
            self.indicies = jax.random.permutation(rng, self.dataset.data.shape[0])
        else:
            self.indicies = jnp.arange(self.dataset.data.shape[0])
        self.bs = prod(self.dataset.batch_dims)
        self.N = floor(self.dataset.data.shape[0] / self.bs)
        self.n = 0

    def __next__(self):
        if self.n < self.N:
            batch = self.dataset.data[
                self.indicies[self.bs * self.n : self.bs * (self.n + 1)], ...
            ]
            self.n = self.n + 1
            batch = batch.reshape(
                (*self.dataset.batch_dims, *self.dataset.data.shape[1:])
            )
        elif (self.n == self.N) and not self.dataset.drop_last:
            batch = self.dataset.data[self.indicies[self.bs * self.n :]]
            self.n = self.n + 1
            # TODO: This only works for 1D batch dims rn
            batch = batch.reshape((-1, *self.dataset.data.shape[1:]))
        else:
            raise StopIteration

        return batch


class CSVDataset(TensorDataset):
    def __init__(self, file, batch_dims, rng, delimiter=",", skip_header=1):
        data = np.genfromtxt(file, delimiter=delimiter, skip_header=skip_header)
        super().__init__(data, batch_dims, rng)
