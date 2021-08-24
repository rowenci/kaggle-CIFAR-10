import pandas as pd
import numpy as np

df = pd.read_csv("submission_id.csv")

i = 0
label_dict = {0 : 'airplane', 1 : 'automobile', 2 : 'bird', 3 : 'cat', 4 : 'deer', 5 : 'dog', 6 : 'frog', 7 : 'horse', 8 : 'ship', 9 : 'truck'}

while i < 300000:
    idx = df["label"][i]
    print(idx)
    label = label_dict[idx]
    df["label"][i] = label
    i += 1

df.to_csv("submission.csv")