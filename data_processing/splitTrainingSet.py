import numpy as np
images = np.load("datas/processed_images/images.npy", allow_pickle=True)
labels = np.load("datas/processed_images/labels.npy", allow_pickle=True)

training_size = 45001
images = images[:training_size]
labels = labels[:training_size]
np.save("datas/pixeled_train/images.npy", images)
np.save("datas/pixeled_train/labels.npy", labels)