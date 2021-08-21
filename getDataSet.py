from PIL import Image
from torchvision import transforms
import logConfig # logging
import numpy as np
import pandas as pd
import torch


logger = logConfig.getLogger("logs/imageProcessing.log")

def loadTrainImages():
    # image list
    images = np.ndarray((45000, 3, 32, 32), dtype=np.float32)
    while True:
        i = 0
        try:
            # open a image
            imageLabel = i + 1
            img_path = "datas/train/" + str(imageLabel) + ".png"
            img = Image.open(img_path)
        except FileNotFoundError: # 没有该图片或者图片读取完成
            if i == 45001:
                logger.info("image loader finished")
            else:
                logger.error("image not found", img_path)
            break
        else:
            # transfer image type into numpy
            img = np.array(img, dtype=np.float32)
            img.resize(3, 32, 32)
            images[i, :, :, :] = img
            i += 1
    return images

def loadTrainLabels():
    labels = pd.read_csv("datas/trainLabels.csv")
    labels = np.array(labels["label"])
    return labels[:45001]

class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self):
        images = loadTrainImages()
        labels = loadTrainLabels()

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = (self.images[index], self.labels[index])
        return data


def loadTestImages():
    # image list
    images = np.ndarray((5000, 3, 32, 32), dtype=np.float32)
    while True:
        i = 0
        try:
            # open a image
            imageLabel = i + 1
            img_path = "datas/train/" + str(imageLabel) + ".png"
            img = Image.open(img_path)
        except FileNotFoundError: # 没有该图片或者图片读取完成
            if i == 50001:
                logger.info("image loader finished")
            else:
                logger.error("image not found", img_path)
            break
        else:
            # transfer image type into numpy
            img = np.array(img, dtype=np.float32)
            img.resize(3, 32, 32)
            images[i, :, :, :] = img
            i += 1
    return images

def loadTestLabels():
    labels = pd.read_csv("datas/trainLabels.csv")
    labels = np.array(labels["label"])
    return labels[45001:]

class TestDataSet(torch.utils.data.Dataset):
    def __init__(self):
        images = loadTestImages()
        labels = loadTestLabels()

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = (self.images[index], self.labels[index])
        return data

