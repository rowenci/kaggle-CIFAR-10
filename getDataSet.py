import torch
import numpy as np
import pandas as pd
import torch

class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self):
        images = np.load("datas/pixeled_train/images.npy", allow_pickle=True)
        labels = np.load("datas/pixeled_train/labels.npy", allow_pickle=True)

        #由于读入的numpy数组里的元素是object类型，无法将这种类型转换成tensor。 所以，将numpy数组进行强制类型转换成float类型
        images = images.astype(np.float32)
        labels = labels.astype(np.float32)
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = (self.images[index], self.labels[index])
        return data


class TestDataSet(torch.utils.data.Dataset):
    def __init__(self):
        images = np.load("datas/pixeled_test/images.npy", allow_pickle=True)
        labels = np.load("datas/pixeled_test/labels.npy", allow_pickle=True)

        images = images.astype(np.float32)
        labels = labels.astype(np.float32)
        
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = (self.images[index], self.labels[index])
        return data

