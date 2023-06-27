import torch
import torch.nn as nn

class simpleCNN(nn.Module):
    def __init__(self, width=128, height=128):
        super(simpleCNN, self).__init__()
        self.img_layer = nn.Sequential(
            # Image Layer

            # batch x 3 x width x height -> batch x 64 x width x height
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # batch x 64 x width x height -> batch x 64 x width / 2 x height / 2
            nn.MaxPool2d(2, 2),

            # batch x 64 x width / 2 x height / 2 -> batch x 32 x width / 2 x height / 2
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # batch x 32 x width / 2 x height / 2 -> batch x 32 x width / 4 x height / 4
            nn.MaxPool2d(2, 2),

            # batch x 32 x width / 4 x height / 4 -> batch x 16 x width / 4 x height / 4
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # batch x 16 x width / 4 x height / 4 -> batch x 16 x width / 8 x height / 8
            nn.MaxPool2d(2, 2)
        )

        self.fc_Layer = nn.Sequential(
            # width * height = 16,384
            nn.Linear(((width * height) >> 2), ((width * height) >> 5)), # 4096 -> 512
            nn.ReLU(),
            nn.Linear(((width * height) >> 5), ((width * height) >> 8)), # 512  -> 64
            nn.ReLU(),
            nn.Linear(((width * height) >> 8), 10)                       # 64   -> 10
        )

    def forward(self, images):
        out_image = self.img_layer(images)
        out_image = out_image.view(images.shape[0], -1)
        out = self.fc_Layer(out_image)
        return out
