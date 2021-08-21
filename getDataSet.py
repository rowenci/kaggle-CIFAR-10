import logConfig # logging
import numpy as np
import pandas as pd
import torch


logger = logConfig.getLogger("logs/imageProcessing.log")


class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self):
        images = np.load("datas/pixeled_train/images.npy", allow_pickle=True)
        labels = np.load("datas/pixeled_train/labels.npy", allow_pickle=True)

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = (self.images[index], self.labels[index])
        return data


class TestDataSet(torch.utils.data.Dataset):
    def __init__(self):
        images = np.load("datas/pixeled_test/images.npy", allow_pickle=True)
        labels = np.load("datas/pixeled_test/labels.npy", allow_pickle=True)

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = (self.images[index], self.labels[index])
        return data

