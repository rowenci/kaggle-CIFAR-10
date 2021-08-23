import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pyttsx3 # 语音播报
import datetime # 计时
import logConfig

import getDataset
import model.ResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr = 1e-3
epochs = 50

train_dataset = getDataset.TrainDataSet()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = getDataset.TestDataSet()
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = model.ResNet.getResNet()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter("tensorboardLog")
logger = logConfig.getLogger("logs/traininglog.log")

train_step = 0
test_step = 0


logger.info("training on {}".format(device))
for epoch in range(epochs):
    begin_time = datetime.datetime.now()
    logger.info("-------epoch {}-------".format(epoch + 1))
    # train
    model.train()
    train_loss = 0
    for data in train_loader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        ouputs = model(imgs)

        loss = criterion(ouputs, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step +=1

    logger.info("train_step : {}".format(train_step))
    logger.info("train_loss : {}".format(train_loss / len(train_loader)))
    writer.add_scalar("train_loss", train_loss / len(train_loader), epoch)

    model.eval()
    test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            accuracy = (outputs.argmax(1) == labels).sum() / len(outputs)
            total_accuracy += accuracy

    # results in one epoch
    logger.info("test_loss is : {}".format(test_loss / len(test_loader)))
    logger.info("total_accuracy is {}".format(total_accuracy / len(test_loader)))
    writer.add_scalar("test_loss", test_loss / len(test_loader), test_step)
    writer.add_scalar("test_accuracy", total_accuracy / len(test_loader), test_step)
    test_step += 1

    # save model in every epoch
    torch.save(model, "model/trained_models/trained_resnet{}.pth".format(epoch + 1))
    logger.info("model has been saved")
    end_time = datetime.datetime.now()
    cost_time = (end_time - begin_time).seconds
    logger.info("time cost : {} seconds".format(cost_time))


# finish
writer.close()

# 训练完成提示
engine = pyttsx3.init() 
volume = engine.getProperty('volume')
engine.setProperty('volume', 1)
engine.say('训练完成，训练完成，训练完成')
# 等待语音播报完毕 
engine.runAndWait()
