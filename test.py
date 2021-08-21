import numpy as np
import pandas as pd
import torch

labels = pd.read_csv("datas/trainLabels.csv")
labels = np.array(labels["label"])
print(labels)