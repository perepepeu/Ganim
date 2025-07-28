# ganim/models.py
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latentDim, channels, imageSize):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latentDim, imageSize * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(imageSize * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(imageSize * 8, imageSize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(imageSize * 4, imageSize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(imageSize * 2, imageSize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize),
            nn.ReLU(True),
            nn.ConvTranspose2d(imageSize, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, channels, imageSize):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, imageSize, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(imageSize, imageSize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(imageSize * 2, imageSize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(imageSize * 4, imageSize * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(imageSize * 8, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x):
        return self.model(x)




