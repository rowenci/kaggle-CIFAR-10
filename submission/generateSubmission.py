import torch
import pandas as pd
from torch.utils.data import DataLoader
from submission.getTestDataset import TestDataSet

batch_size = 128
model = torch.load("model/trained_models/trained_resnet8.pth")

test_dataset = TestDataSet()
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for data in test_loader:
    imgs = data
    imgs = imgs.to("cuda:0")
    outputs = model(imgs)
    outputs = outputs.argmax(1)
    print(outputs)
    outputs = outputs.to("cpu")
    df = pd.DataFrame(outputs)
    df.to_csv("submission.csv", mode='a', header=0)