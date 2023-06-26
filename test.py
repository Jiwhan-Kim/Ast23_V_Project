'''
Asterisk 2023
Asterisk Vanadium Project
Test Mode
'''

import torch
import numpy    as np

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

batch_size = 16
epoch      = 200

if __name__ == "__main__":
    print("Training Mode")
    print("Device on Working: ", device)

    model   = M.CNN.to(device)
    trainer = T.trainer(0.0001, model, device)
    
