import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import getDataset

batch_size = 10

train_dataset = getDataset.TrainDataSet()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = getDataset.TestDataSet()
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

"""
for data in train_dataloader:
    imgs, labels = data
    print(imgs)
    print(labels)
    print(imgs.shape)
    print(labels.shape)
    break
"""

