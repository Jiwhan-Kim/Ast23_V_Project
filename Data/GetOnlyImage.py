import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .makeNumPy import ImgToNpConverter

class MyDataSet(Dataset):
    def __init__(self, image, label, avg_image, std_image):
        self.images = torch.tensor(image, dtype=torch.float).permute(0, 3, 1, 2)
        self.labels = torch.tensor(label, dtype=torch.long)

        self.images -= (avg_image)
        self.images /= (std_image)

        print(avg_image, std_image)
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def getOnlyImage(batch_size=1):
    data_image  = ImgToNpConverter()

    avg_image   = np.average(data_image)
    std_image   = np.std    (data_image)

    train_image = np.empty(shape=(0, 128, 128, 3))

    test_image  = np.empty(shape=(0, 128, 128, 3))

    train_label = np.empty(shape=0)
    test_label  = np.empty(shape=0)
    for i in range(10):
        for j in range(30):
            temp_img = data_image[i * 30 + j].reshape(1, 128, 128, 3)
            temp_img_flipped = np.flip(temp_img, axis=2).copy()
            if j < 25:
                train_image = np.append(train_image, temp_img, axis=0)
                train_image = np.append(train_image, temp_img_flipped, axis=0)
                train_label = np.append(train_label, i)
                train_label = np.append(train_label, i)
            else:
                test_image  = np.append(test_image,  temp_img, axis=0)
                test_label  = np.append(test_label,  i)
    
    train_ds    = MyDataSet(image=train_image, label=train_label,
                         avg_image=avg_image, std_image=std_image,
                         )

    test_ds     = MyDataSet(image=test_image,  label=test_label,
                         avg_image=avg_image, std_image=std_image,
                         )

    train_load  = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_load   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print("Data Loading Completed")
    return train_load, test_load

