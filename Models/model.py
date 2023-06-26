import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, width=64, height=64):
        super(CNN, self).__init__()
        self.img_layer = nn.Sequential(
            # Image Layer

            # batch x 3 x width x height -> batch x 64 x width x height
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
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

        self.audio_layer = nn.Sequential(
            # Audio Layer

            nn.Conv1d(in_channels=)
        )
        self.fcLayer = nn.Sequential(
            # width * height = 4096
            nn.Linear((width * height) >> 2, (width * height) >> 2), # 1024 -> 256
            nn.ReLU(),
            nn.Linear((width * height) >> 2, (width * height) >> 4), # 256 -> 64
            nn.ReLU(),
            nn.Linear((width * height) >> 4, 10) # 64-> 10
        )

    def forward(self, x):
        out = self.img_layer(x)
        out = out.view(x.shape[0], -1)
        out = self.fclayer(out)
        return out

