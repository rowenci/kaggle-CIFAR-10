from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import sys
sys.path.append("d:\\Codes\\AI\\kaggle\\kaggle-CIFAR-10\\")

def loadImages():
    # image list
    images = np.zeros((300000, 3, 32, 32))
    print("begining loading images")
    i = 0
    while True:
        print(i)
        try:
            # open a image
            imageLabel = i + 1
            img_path = "datas/test/" + str(imageLabel) + ".png"
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

images = loadImages()

np.save("test_images.npy", images)
