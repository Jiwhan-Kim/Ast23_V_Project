'''
Asterisk 2023
Asterisk Vanadium Project
Training Mode
'''

import torch
import numpy as np
from os import path

import Models   as M
import Trainers as T
import Data     as D

# Set Devices (M1/M2 mps, NVIDIA cuda:0, else cpu)
device = None
if   torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

category = [
    "Cat",      # 0
    "Chicken",  # 1
    "Cow",      # 2
    "Dog",      # 3
    "Duck",     # 4
    "Eagle",    # 5
    "Lion",     # 6
    "Pig",      # 7
    "Sheep",    # 8
    "Tiger"     # 9
]

batch_size = 50
epoch      = 30


def train(loader, n_epoch):
    sum_loss = 0
    model.train()
    for image, label in loader:
        sum_loss = trainer.step(image, label)
        print("epoch: {}, loss: {}".format(n_epoch, sum_loss))



if __name__ == "__main__":
    print("Training Mode")
    print("Device on Working: ", device)

    model   = M.simpleCNN().to(device)
    trainer = T.trainer_img(0.001, model, device)
    train_load, test_load = D.getOnlyImage(batch_size)

    '''
    if path.exists("./model_params.pth"):
        model.load_state_dict(torch.load("./model_params.pth"))
    '''
    for i in range(1, epoch + 1):
        train(train_load, i)
    
    # Training Done
    with torch.no_grad():
        model.eval()
        val = np.zeros(10, dtype=int)
        correct = 0

        for image, label in test_load:
            images = image.to(device)
            labels = label.to(device)

            output = model.forward(images)
            result = torch.argmax(output, dim=1)
            for res, ans in zip(result, labels):
                print(res, ans)
                if res == ans:
                    val[res] += 1
                    correct += 1
        
        print("\nFinal Result - Accuracy: {}\n".format(100 * correct / 50))
        for i in range(10):
            print("class: {}: {} / 5".format(category[i], val[i]))
        
        torch.save(model.state_dict(), 'model_params.pth')

