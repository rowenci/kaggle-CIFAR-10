import torch
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self):
        images = np.load("datas/pixeled_train/images.npy", allow_pickle=True)
        labels = np.load("datas/pixeled_train/labels.npy", allow_pickle=True)

        #由于读入的numpy数组里的元素是object类型，无法将这种类型转换成tensor。 所以，将numpy数组进行强制类型转换成float类型
        images = images.astype(np.float32)
        labels = labels.astype(np.float32)

        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        for i in range(len(images)):
            img = images[i]
            trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            images[i] = trans_norm(img)

        self.images = images
        self.labels = labels.long()

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

        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        for i in range(len(images)):
            img = images[i]
            trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            images[i] = trans_norm(img)

        self.images = images
        self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = (self.images[index], self.labels[index])
        return data

