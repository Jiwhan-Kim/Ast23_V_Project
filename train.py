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
    "Sheep"     # 8
]

batch_size = 8
epoch      = 20


def train(loader, n_epoch):
    sum_loss = 0
    model.train()
    for image, sound, label in loader:
        sum_loss = trainer.step(image, sound, label)
        print("epoch: {}, loss: {}".format(n_epoch, sum_loss))



if __name__ == "__main__":
    print("Training Mode")
    print("Device on Working: ", device)

    model   = M.CNN().to(device)
    trainer = T.trainer(0.0001, model, device)
    train_load, test_load = D.getData(batch_size)

    '''
    if path.exists("./model_params_cnn.pth"):
        model.load_state_dict(torch.load("./model_params_cnn.pth"))
    '''

    for i in range(1, epoch + 1):
        train(train_load, i)
    
    # Training Done
    with torch.no_grad():
        model.eval()
        val = np.zeros(9, dtype=int)
        correct = 0

        for image, sound, label in test_load:
            images = image.to(device)
            sounds = sound.to(device)
            labels = label.to(device)

            output = model.forward(images, sounds)
            result = torch.argmax(output, dim=1)
            for res, ans in zip(result, labels):
                print("Predicted: {} / Answer: {}".format(category[res], category[ans]))
                if res == ans:
                    val[res] += 1
                    correct += 1
        
        print("Final Result - Accuracy: {}\n\n".format(100 * correct / 45))
        for i in range(9):
            print("class: {}: {} / 5".format(category[i], val[i]))
        
        torch.save(model.state_dict(), 'model_params_cnn.pth')

