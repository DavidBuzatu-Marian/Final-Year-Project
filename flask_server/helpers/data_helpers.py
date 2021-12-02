import torch.utils.data as data
import torch
import cv2
import os
from glob import glob
from torch.utils.data import DataLoader


class CustomDataset(data.Dataset):
    def __init__(self, data_path="", labels_path=""):
        super(CustomDataset, self).__init__()
        self.data = self.__get_images(data_path)
        self.labels = self.__get_images(labels_path)

    def __get_images(self, path):
        imgs = []
        for filename in glob(os.path.join(path)):
            imgs.append(os.path.join(path, filename))
        return imgs

    def __getitem__(self, index):
        data = cv2.imread(self.data[index], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(self.labels[index], cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.data)


def get_dataloader(data_path, labels_path, hyperparameters):
    return DataLoader(
        dataset=CustomDataset(os.getenv(data_path, labels_path)),
        num_workers=hyperparameters["num_workers"],
        batch_size=hyperparameters["batch_size"],
        shuffle=hyperparameters["shuffle"],
    )


def reshape_data(data, hyperparameters):
    if "reshape" in hyperparameters:
        size = hyperparameters["reshape"]
        return data.reshape(size)
    return data


def normalize_data(data, hyperparameters):
    if "normalizer" in hyperparameters:
        normalizer = hyperparameters["normalizer"]
        return data / normalizer  # TODO: Research different normalizers
    return data
