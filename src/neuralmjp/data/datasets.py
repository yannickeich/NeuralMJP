import os

import numpy as np
import torch
from ..utils.file_operations import load_csv
from torch.utils.data import Dataset


class FileInMemoryDataset(Dataset):
    def __init__(self, file_path: str, max_size: int = None):
        super(FileInMemoryDataset, self).__init__()

        self.data = torch.from_numpy(load_csv(file_path, max_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def split(
        cls,
        root_dir: str,
        train_file: str = "train.csv",
        validate_file: str = "valid.csv",
        test_file: str = "test.csv",
        max_size=None,
    ):
        train_ds = cls(os.path.join(root_dir, train_file), max_size=max_size)
        valid_ds = cls(os.path.join(root_dir, validate_file), max_size=max_size)
        test_ds = cls(os.path.join(root_dir, test_file), max_size=max_size)
        return train_ds, valid_ds, test_ds


class IonChannelDataset(Dataset):
    def __init__(self, data_dir):
        super(IonChannelDataset, self).__init__()

        data = torch.from_numpy(np.load(data_dir))
        self.observations, self.obs_times, self.delta_times = self.preprocess(data)

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return {
            "observations": self.observations[idx],
            "obs_times": self.obs_times[idx],
            "delta_times": self.delta_times[idx],
        }

    def preprocess(self, data):
        """
        data: [B, T, 2]
        """
        obs_times = data[:, :, 0]
        delta_t = obs_times[:, 1:] - obs_times[:, :-1]
        obs = data[:, :, 1:]  # [B, T, 2]

        return obs[:, :-1], obs_times.unsqueeze(-1)[:, :-1], delta_t.unsqueeze(-1)


class MJPDataset(Dataset):
    def __init__(self, **kwargs):
        super(MJPDataset, self).__init__()
        self.kwargs = kwargs

        self.observations, self.obs_times, self.delta_times = self.preprocess()

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return {
            "observations": self.observations[idx],
            "obs_times": self.obs_times[idx],
            "delta_times": self.delta_times[idx],
        }

    def preprocess(self):
        """Returns observations, obs_times and delta_times"""


class LotkaVolterraDataset(MJPDataset):
    def __init__(self, **kwargs):
        super(LotkaVolterraDataset, self).__init__(**kwargs)

    def preprocess(self):
        root_dir = self.kwargs.get("root_dir")

        raw_data = torch.from_numpy(np.load(root_dir))  # [B, T, 2 + 1]

        obs_times = raw_data[:, :, 0]
        delta_t = obs_times[:, 1:] - obs_times[:, :-1]
        obs = raw_data[:, :, 1:]  # [B, T, 2]

        return obs[:, :-1], obs_times.unsqueeze(-1)[:, :-1], delta_t.unsqueeze(-1)


class LotkaVolterraDatasetOneHot(MJPDataset):
    def __init__(self, **kwargs):
        self.max_state = kwargs.get("max_state", 60)
        super(LotkaVolterraDatasetOneHot, self).__init__(**kwargs)

    def preprocess(self):
        root_dir = self.kwargs.get("root_dir")

        raw_data = torch.from_numpy(np.load(root_dir))  # [B, T, 1 + 2]

        obs_times = raw_data[:, :, 0]
        delta_t = obs_times[:, 1:] - obs_times[:, :-1]

        # convert raw obs to one-hot vectors
        B, T, _ = raw_data.shape

        raw_data = raw_data.clamp(max=self.max_state - 1)

        all_states = torch.arange(start=0, end=self.max_state, step=1).view(1, 1, 1, -1, 1)
        raw_obs = raw_data[:, :, 1:].view(B, T, 2, 1, 1).expand(-1, -1, -1, self.max_state, -1)
        obs = torch.all(raw_obs == all_states, dim=-1)  # [B, T, 2, self.max_state]

        # flatten one hot vectors for each process into one
        obs = obs.flatten(start_dim=-2)  # [B, T, 2 * self.max_state]

        return obs[:, :-1], obs_times.unsqueeze(-1)[:, :-1], delta_t.unsqueeze(-1)


class FlashingRatchetDatasetOneHot(MJPDataset):
    def __init__(self, **kwargs):
        super(FlashingRatchetDatasetOneHot, self).__init__(**kwargs)

    def preprocess(self):
        root_dir = self.kwargs.get("root_dir")

        raw_data = torch.from_numpy(np.load(root_dir))  # [B, T, 2 + 1]

        obs_times = raw_data[:, :, 0]
        delta_t = obs_times[:, 1:] - obs_times[:, :-1]

        B, T, _ = raw_data.shape

        # convert raw obs to one-hot vectors
        all_states = torch.tensor([0, 1, 2, 3, 4, 5]).view(1, 1, 6, 1)
        raw_obs = raw_data[:, :, 1:].unsqueeze(-2).expand(-1, -1, 6, -1)
        obs = torch.all(raw_obs == all_states, dim=-1)  # [B, T, 6]

        return obs[:, :-1], obs_times.unsqueeze(-1)[:, :-1], delta_t.unsqueeze(-1)


class ADPDataset(MJPDataset):
    def __init__(self, **kwargs):
        super(ADPDataset, self).__init__(**kwargs)

    def preprocess(self):
        root_dir = self.kwargs.get("root_dir")
        seq_length = self.kwargs.pop("seq_length", 100)
        atoms = self.kwargs.pop("atoms", (5, 7, 9, 11, 15, 17))
        dt = self.kwargs.pop("dt", 0.004)

        raw_data = torch.from_numpy(np.load(root_dir))  # [B, T, *, 3]

        atoms_corrected = [a - 1 for a in atoms]

        selected_atoms = [raw_data[:, a, :] for a in atoms_corrected]
        selected_atoms = torch.stack(selected_atoms, dim=1)

        obs = torch.stack(torch.split(selected_atoms, seq_length), dim=0)
        obs = torch.flatten(obs, start_dim=2)  # [B, T, 3*]

        B = obs.shape[0]

        obs_times = (
            torch.arange(start=0, end=seq_length).unsqueeze(0).expand(B, seq_length).unsqueeze(-1)
        )

        delta_t = dt * torch.ones_like(obs_times)

        return obs, obs_times, delta_t
