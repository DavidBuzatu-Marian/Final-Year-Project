from logging import error
import torch.utils.data as data
import torch
import cv2
import os
from glob import glob
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision import transforms


class CustomDataset(data.Dataset):
    def __init__(self, data_path="", labels_path=""):
        super(CustomDataset, self).__init__()
        self.data = self.__get_images(data_path)
        self.labels = self.__get_images(labels_path)

    def __get_images(self, path):
        imgs = []
        for filename in os.listdir(os.path.join(path)):
            imgs.append(os.path.join(path, filename))
        imgs.sort()
        return sorted(imgs, key=len)

    def __getitem__(self, index):
        data = cv2.imread(self.data[index], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(self.labels[index], cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.data)


def get_dataloader(data_path, labels_path, hyperparameters):
    return DataLoader(
        dataset=CustomDataset(data_path, labels_path),
        num_workers=hyperparameters["num_workers"],
        batch_size=hyperparameters["batch_size"],
        shuffle=hyperparameters["shuffle"],
        drop_last=hyperparameters["drop_last"],
    )

# With support from https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c


def compute_mean_and_std(dataloader):
    sum_of_channels = 0
    sum_squared_of_channels = 0
    iterations_of_batches = 0

    for input, _ in dataloader:
        if len(input.shape) == 3:
            # missing color channel
            input = torch.unsqueeze(input, 1)
        # skip channel as this is sum over channels
        sum_of_channels = torch.mean(input, dim=[0, 2, 3])
        sum_squared_of_channels = torch.mean(input ** 2, dim=[0, 2, 3])  # same as above
        iterations_of_batches += 1

    mean = sum_of_channels / iterations_of_batches
    std = (sum_squared_of_channels / iterations_of_batches - mean ** 2) ** 0.5  # by sigma's formula
    return mean, std


def reshape_data(data, hyperparameters):
    if "reshape" in hyperparameters:
        size = eval(hyperparameters["reshape"])
        return data.reshape(size)
    return data


def normalize_data(data, mean, std):
    normalizer = Normalize(mean, std)
    return normalizer(data)


def normalize(hyperparameters):
    return "normalize" in hyperparameters
