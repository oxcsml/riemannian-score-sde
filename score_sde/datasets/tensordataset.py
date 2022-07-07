from math import prod, floor

import jax
import jax.numpy as jnp
import numpy as np


class TensorDataset:
    def __init__(self, data):
        self.data = jnp.array(data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class DataLoader:
    def __init__(self, dataset, batch_dims, rng, shuffle=False, drop_last=False):
        self.dataset = dataset
        assert isinstance(batch_dims, int)
        self.batch_dims = batch_dims
        self.rng = rng

        self.shuffle = shuffle
        self.drop_last = drop_last

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        bs = self.batch_dims
        N = floor(len(self.dataset) / bs)
        return N if self.drop_last else N + 1

    def __iter__(self):
        return DatasetIterator(self)

    def __next__(self):
        rng, next_rng = jax.random.split(self.rng)
        self.rng = rng

        indices = jax.random.choice(next_rng, len(self.dataset), shape=(self.batch_dims,))

        return self.dataset[indices], None
        # return self.data[indices].reshape((self.batch_dims, *self.dataset.shape[1:]))


class DatasetIterator:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        rng, self.dataloader.rng = jax.random.split(self.dataloader.rng)
        if self.dataloader.shuffle:
            self.indices = jax.random.permutation(rng, len(self.dataloader.dataset))
        else:
            self.indices = jnp.arange(len(self.dataloader.dataset))
        self.bs = self.dataloader.batch_dims
        self.N = floor(len(dataloader.dataset) / self.bs)
        self.n = 0

    def __next__(self):
        if self.n < self.N:
            batch = self.dataloader.dataset[
                self.indices[self.bs * self.n : self.bs * (self.n + 1)], ...
            ]
            self.n = self.n + 1
            # batch = batch.reshape(
            #     (self.dataset.batch_dims, *self.dataset.data.shape[1:])
            # )
        elif (self.n == self.N) and not self.dataloader.drop_last:
            batch = self.dataloader.dataset[self.indices[self.bs * self.n :]]
            self.n = self.n + 1
            # TODO: This only works for 1D batch dims rn
            # batch = batch.reshape((-1, *self.dataset.data.shape[1:]))
        else:
            raise StopIteration

        return batch, None


# TODO: assumes 1d batch_dims
class SubDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return self.indices.shape[0]


class CSVDataset(TensorDataset):
    def __init__(self, file, delimiter=",", skip_header=1):
        data = np.genfromtxt(file, delimiter=delimiter, skip_header=skip_header)
        super().__init__(data)
