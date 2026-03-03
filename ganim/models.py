import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, channels, image_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size * 4, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size * 2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(image_size, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, channels, image_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, image_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 8, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x):
        return self.model(x)
