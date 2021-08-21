import numpy as np
import pandas as pd
import torch


labels = np.load("datas/processed_images/labels.npy", allow_pickle=True)
print(labels)
