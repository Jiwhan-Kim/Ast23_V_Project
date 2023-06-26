import torch
import torch.nn as nn
import torch.optim as optim


class trainer:
    def __init__(self, lr, model, device):
        self.lr = lr
        self.model = model
        self.device = device
        self.lossF = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def step(self, image: torch.tensor, sound: torch.tensor, label: torch.tensor) -> float:
        x1 = image.to(self.device)
        x2 = sound.to(self.device)
        y  = label.to(self.device)
        self.optimizer.zero_grad()
        output = self.model.forward(x1, x2)
        loss = self.lossF(output, y)
        loss.backward()
        self.optimizer.step()
        return loss