from PIL import Image
from torchvision import transforms
import logConfig # logging
import numpy as np
import pandas as pd


logger = logConfig.getLogger("logs/imageProcessing.log")

def loadImages():
    # image list
    images = np.ndarray((50000, 3, 32, 32), dtype=np.float32)
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

def loadLabels():
    logger.info("begining loading labels")
    labels = pd.read_csv("datas/trainLabels.csv")
    labels = np.array(labels["label"])
    return labels

images = loadTrainImages()
labels = loadTrainLabels()

np.save("datas/pixeled_train/images.npy", images)
np.save("datas/pixeled_train/labels.npy", labels)
