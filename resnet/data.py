from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class Cifar10DatasetFromFile(Dataset):

    def __init__(self, img_file: str, label_file: str, transforms=None):
        """
        Data is small so keep it in memory

        Data is expected in (S, C, H, W) and permuted -> (S, H, W, C)
        """
        super().__init__()
        self.images = np.transpose(np.load(img_file), (0, 2, 3, 1))
        self.labels = np.load(label_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        image = self.images[index]
        if self.transforms:
            image = self.transforms(image)
        return image, torch.Tensor(label)


class Cifar10DatasetFromArray(Dataset):

    def __init__(self, images: Sequence[np.uint8], labels: Sequence[np.uint8], transforms=None):
        """
        Numpy images, (S, C, H, W) -> (S, H, W, C)
        Transforms should be compose
        """
        super().__init__()
        self.images = np.transpose(images, (0, 2, 3, 1))
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        image = self.images[index]
        if self.transforms:
            image = self.transforms(image)
        return image, torch.Tensor(label)
