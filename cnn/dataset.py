import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset


def crop_black_area(sample: torch.Tensor) -> torch.Tensor:
    nonzero_indices = torch.nonzero(sample != 0)

    h_min = nonzero_indices[:, -2].min()
    w_min = nonzero_indices[:, -1].min()
    h_max = nonzero_indices[:, -2].max() + 1
    w_max = nonzero_indices[:, -1].max() + 1

    return sample[:, h_min:h_max, w_min:w_max]


class MaskedNormalization:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        normalized_sample = (sample - self.mean) / self.std
        normalized_sample[:, torch.all(sample == 0, dim=0)] = 0
        return normalized_sample


class Subset2Dataset(Dataset):
    def __init__(self, subset: Subset, transform: transforms.Compose | None = None) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple:
        sample, target = self.subset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self) -> int:
        return len(self.subset)
