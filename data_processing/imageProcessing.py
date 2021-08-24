from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import sys
sys.path.append("d:\\Codes\\AI\\kaggle\\kaggle-CIFAR-10\\")
import logConfig # logging


logger = logConfig.getLogger("logs/imageProcessing.log")

def loadImages():
    # image list
    images = np.zeros((50000, 3, 32, 32))
    logger.info("begining loading images")
    i = 0
    while True:
        logger.info(i)
        try:
            # open a image
            imageLabel = i + 1
            img_path = "datas/train/" + str(imageLabel) + ".png"
            img = Image.open(img_path)
        except FileNotFoundError: # 没有该图片或者图片读取完成
            break
        else:
            # transfer image type into numpy
            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.transpose(0, 2)
            img = img.transpose(1, 2)
            images[i, :, :, :] = img
            i += 1
    return images

def loadLabels():
    logger.info("begining loading labels")
    labels = pd.read_csv("datas/trainLabels.csv")
    labels = np.array(labels["label"])
    label_dict = {'airplane' : 0, 'automobile' : 1, 'bird' : 2, 'cat' : 3, 'deer' : 4, 'dog' : 5, 'frog' : 6, 'horse' : 7, 'ship' : 8, 'truck' : 9}
    i = 0
    while i < 50000:
        labels[i] = label_dict[labels[i]]
        i += 1
    return labels

images = loadImages()
labels = loadLabels()

np.save("datas/processed_images/images.npy", images)
np.save("datas/processed_images/labels.npy", labels)
