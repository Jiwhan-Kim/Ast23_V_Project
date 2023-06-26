import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, width=128, height=128, samples=32768):
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
            
            # batch x 1 x samples -> batch x 16 x samples / 4
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # batch x 16 x samples / 4 -> batch x 8 x samples / 16
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # batch x 8 x samples / 16 -> batch x 4 x samples / 64
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.fc_Layer = nn.Sequential(
            # width * height = 16,384
            # samples        = 32,768
            nn.Linear(((width * height) >> 2) + (samples >> 4), ((width * height) >> 5) + (samples >> 7)),  # 4096 + 2048 -> 512 + 256
            nn.ReLU(),
            nn.Linear(((width * height) >> 5) + (samples >> 7), ((width * height) >> 8) + (samples >> 10)), # 512  + 256  -> 64 + 32
            nn.ReLU(),
            nn.Linear(((width * height) >> 8) + (samples >> 10), 10)                                        # 64   + 32   -> 10
        )

    def forward(self, images, sounds):
        out_image = self.img_layer(images)
        out_sound = self.img_layer(sounds)

        out_image = out.view(images.shape[0], -1)
        out_sound = out.view(sounds.shape[0], -1)

        out = torch.concat((out_image, out_sound), dim=0)
        out = self.fclayer(out)
        return out
