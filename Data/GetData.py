import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from makeNumPy import ImgToNpConverter, WavToNpConverter

class MyDataSet(Dataset):
    def __init__(self, image, sound, label, avg_image, std_image, avg_sound, std_sound):
        self.images = torch.tensor(image / 256., dtype=torch.float).permute(0, 3, 1, 2)
        self.sounds = torch.tensor(sound / 256., dtype=torch.float)
        self.labels = torch.tensor(label, dtype=torch.long)

        self.images -= avg_image
        self.images /= std_image

        self.sounds -= avg_sound
        self.sounds /= std_image

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.sounds[index], self.labels[index]


def getData(batch_size=1):
    data_image  = ImgToNpConverter()
    data_sound  = WavToNpConverter()
    
    avg_image   = np.average(data_image)
    std_image   = np.std    (data_image)
    avg_sound   = np.average(data_sound)
    std_sound   = np.std    (data_sound)

    train_image = np.empty(0, 128, 128, 3)
    train_sound = np.empty(0, 1, 32768)

    test_image  = np.empty(0, 128, 128, 3)
    test_sound  = np.empty(0, 1, 32768)

    train_label = np.empty(shape=0)
    test_label  = np.empty(shape=0)
    for i in range(10):
        for j in range(30):
            if j < 25:
                train_image = np.append(train_image, data_image[i * 30 + j])
                train_sound = np.append(train_sound, data_sound[i * 30 + j])
                train_label = np.append(train_label, i)
            else:
                test_image  = np.append(test_image,  data_image[i * 30 + j])
                test_sound  = np.append(test_sound,  data_sound[i * 30 + j])
                test_label  = np.append(test_label,  i)
    
    train_ds    = MyDataSet(image=train_image, sound=train_sound, label=train_label,
                         avg_image=avg_image, std_image=std_image,
                         avg_sound=avg_sound, std_sound=std_sound
                         )

    test_ds     = MyDataSet(image=test_image,  sound=test_sound,  label=test_label,
                         avg_image=avg_image, std_image=std_image,
                         avg_sound=avg_sound, std_sound=std_sound
                         )

    train_load  = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_load   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print("Data Loading Completed")
    return train_load, test_load

