import logging
from abc import ABC, abstractmethod

import torch
from math import floor, ceil
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataloader import DataLoader
from .datasets import LotkaVolterraDataset, FlashingRatchetDatasetOneHot, LotkaVolterraDatasetOneHot

sampler = torch.utils.data.RandomSampler
DistributedSampler = torch.utils.data.distributed.DistributedSampler

logger = logging.getLogger(__name__)


class ADataLoader(ABC):
    def __init__(self, device: torch.device, batch_size: int, rank: int = 0, world_size: int = -1):
        self.device = device
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank

    @property
    @abstractmethod
    def train(self):
        ...

    @property
    @abstractmethod
    def validate(self):
        ...

    @property
    @abstractmethod
    def test(self):
        ...

    @property
    def n_train_batches(self):
        return len(self.train.dataset) // self.batch_size // abs(self.world_size)

    @property
    def n_validate_batches(self):
        return max(len(self.validate.dataset) // self.batch_size // abs(self.world_size), 1)

    @property
    def n_test_batches(self):
        return max(len(self.test.dataset) // self.batch_size // abs(self.world_size), 1)

    @property
    def train_set_size(self):
        return len(self.train.dataset)

    @property
    def validation_set_size(self):
        return len(self.validate.dataset)

    @property
    def test_set_size(self):
        return len(self.test.dataset)


class FileDataLoader(ADataLoader):
    def __init__(self, device: torch.device, rank: int = 0, world_size=-1, **kwargs):
        bs = kwargs.get("batch_size")
        super(FileDataLoader, self).__init__(device, bs, rank, world_size)
        self.device = device

        self.scaler_obs = MinMaxScaler()
        self.scaler_time = MinMaxScaler()

        self._train_dl, self._valid_dl, self._test_dl = self._init_data_loader(**kwargs)

    def _init_data_loader(self, **kwargs):
        """
        Load, split and normalize data.
        Return dataloaders for train, test and validation.
        """

        # Load dataset
        dataset = kwargs.pop("dataset")

        if dataset == "LotkaVolterraDataset":
            ds = LotkaVolterraDataset(**kwargs)
        elif dataset == "LotkaVolterraDatasetOneHot":
            ds = LotkaVolterraDatasetOneHot(**kwargs)
        elif dataset == "FlashingRatchetDatasetOneHot":
            ds = FlashingRatchetDatasetOneHot(**kwargs)

        # Train, Test, Validation split
        train_fraction = kwargs.get("train_fraction", 0.9)
        n_series = ds.observations.size(0)

        n_train_samples = floor(train_fraction * n_series)
        n_valid_samples = ceil((1 - train_fraction) / 2 * n_series)

        randperm = torch.randperm(n_series)

        train_idx = randperm[:n_train_samples]
        valid_idx = randperm[n_train_samples : (n_train_samples + n_valid_samples)]
        test_idx = randperm[(n_train_samples + n_valid_samples) :]

        train_ds = torch.utils.data.Subset(ds, train_idx)
        valid_ds = torch.utils.data.Subset(ds, valid_idx)
        test_ds = torch.utils.data.Subset(ds, test_idx)

        # Normalize data
        self.normalize_obs = kwargs.get("normalize_obs", False)
        self.normalize_times = kwargs.get("normalize_times", False)

        self._fit_obs_scaler(train_ds[:]["observations"])
        self._fit_times_scaler(train_ds[:]["obs_times"])

        if self.normalize_obs:
            ds = self._apply_obs_scaler(ds)
        if self.normalize_times:
            ds = self._apply_time_scaler(ds)

        # Put data on device
        ds.observations = ds.observations.to(self.device)
        ds.obs_times = ds.obs_times.to(self.device)
        ds.delta_times = ds.delta_times.to(self.device)

        # Initialize all dataloaders
        train_sampler = None
        valid_sampler = None
        test_sampler = None

        if self.world_size != -1:
            train_sampler = DistributedSampler(train_ds, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_ds, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_ds, self.world_size, self.rank)

        train_dl = DataLoader(
            train_ds,
            drop_last=True,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            batch_size=self.batch_size,
        )
        valid_dl = DataLoader(
            valid_ds,
            drop_last=True,
            sampler=valid_sampler,
            shuffle=valid_sampler is None,
            batch_size=self.batch_size,
        )
        test_dl = DataLoader(
            test_ds,
            drop_last=True,
            sampler=test_sampler,
            shuffle=test_sampler is None,
            batch_size=self.batch_size,
        )

        return train_dl, valid_dl, test_dl

    def _apply_obs_scaler(self, ds):
        """
        Apply fitted scaler for observations to a dataset.
        """
        obs = ds.observations

        shape = obs.shape
        obs = obs.reshape(-1, 1)
        obs = self.scaler_obs.transform(obs)
        obs = torch.from_numpy(obs).float()
        obs = obs.reshape(shape)

        ds.observations = obs

        return ds

    def _apply_time_scaler(self, ds):
        """
        Apply fitted scaler for times to a dataset.
        """
        t = ds.obs_times
        dt = ds.delta_times

        shape = t.shape
        t = self.scaler_time.transform(t.reshape(-1, 1))
        t = torch.from_numpy(t).float()
        t = t.reshape(shape)

        dt = dt / (self.scaler_time.data_max_[0] - self.scaler_time.data_min_[0])

        ds.obs_times = t
        ds.delta_times = dt

        return ds

    def _fit_obs_scaler(self, x):
        """
        x: [B, T, *]
        """
        x = x.reshape(-1, 1)
        self.scaler_obs.fit(x)

    def _fit_times_scaler(self, t):
        """
        t: [B, T, 1]
        """
        self.scaler_time.fit(t.reshape(-1, 1))

    def _denormalize_obs(self, x):
        """
        x: [B, T, n_proc, D]
        """
        shape = x.shape
        x = x.reshape(-1, 1)
        x = self.scaler_obs.inverse_transform(x)
        x = torch.from_numpy(x).float()
        x = x.reshape(shape)

        return x

    def _denormalize_times(self, t, dt):
        """
        t: [B, T, 1]
        dt: [B, T-1, 1]
        """
        shape = t.shape
        t = self.scaler_time.inverse_transform(t.reshape(-1, 1))
        t = torch.from_numpy(t).float()
        t = t.reshape(shape)

        dt = dt * (self.scaler_time.data_max_[0] - self.scaler_time.data_min_[0])

        return t, dt

    def get_params(self):
        if self.normalize_times:
            max_obs_time = 1
            time_scaling = (self.scaler_time.data_max_ - self.scaler_time.data_min_).item()
        else:
            max_obs_time = self.scaler_time.data_max_.item()
            time_scaling = 1

        return {"max_obs_time": max_obs_time, "time_scaling": time_scaling}

    @property
    def train(self):
        return self._train_dl

    @property
    def validate(self):
        return self._valid_dl

    @property
    def test(self):
        return self._test_dl
