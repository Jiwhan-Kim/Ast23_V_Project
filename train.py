'''
Asterisk 2023
Asterisk Vanadium Project
Training Mode
'''

import torch
import numpy as np

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

batch_size = 16
epoch      = 200


if __name__ == "__main__":
    print("Training Mode")
    print("Device on Working: ", device)

    model   = M.CNN.to(device)
    trainer = T.trainer(0.0001, model, device)
    
