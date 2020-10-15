import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(torch.utils.data.Dataset):
    """Sliding dataset over an array with window size n, return n+1 as target."""

    def __init__(self, tensor, size):
        self.data = tensor
        self.window_size = size

    def __len__(self):
        return len(self.data) - self.window_size  # Number of sequences

    def __getitem__(self, idx):
        # Get single sequence by idx and split the last element away as target
        return (
            self.data[idx : idx + self.window_size],
            self.data[(idx + self.window_size)],
        )
