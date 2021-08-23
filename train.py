import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import model.ResNet
import getDataset

batch_size = 10
lr = 1e-3
epochs = 100

train_dataset = getDataset.TrainDataSet()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = getDataset.TestDataSet()
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

net = model.ResNet.getResNet()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

net.train()
for epoch in epochs:
    # train
    for data in test_dataloader:
        imgs, labels = data
        
        break
