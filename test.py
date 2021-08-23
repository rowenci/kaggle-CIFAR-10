import torch
import numpy as np
import getDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, dataloader
from PIL import Image

train_dataset = getDataset.TrainDataSet()
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)


#writer = SummaryWriter("tensorboardLog/test")

for data in train_loader:
    imgs, labels = data
    print(imgs[0])
    break

"""
images = np.ndarray((50000, 3, 32, 32), dtype=np.float32)
img_path = "datas/train/1.png"
img = Image.open(img_path)
img = np.array(img, dtype=np.float32)
img = torch.from_numpy(img)
img = img.transpose(0, 2)
img = img.transpose(1, 2)
writer.add_image('data', img)

img = Image.open(img_path)
img2 = np.array(img)
img2 = torch.from_numpy(img2)
img2 = img2.transpose(0, 2)
img2 = img2.transpose(1, 2)
writer.add_image('data2', img2)
"""