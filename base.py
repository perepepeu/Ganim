import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pygame
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import platform
import warnings
from tqdm import tqdm

# --- 1. Hiperparâmetros ---
IMG_SIZE = 16
DISPLAY_SIZE = 256
CHANNELS = 3
LATENT_DIM = 100
EPOCHS = 20000
BATCH_SIZE = 64
SAMPLE_INTERVAL = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0002
BETA1 = 0.5
WORKERS = 2 if platform.system() != "Windows" else 0

REAL_LABEL = 0.9
FAKE_LABEL = 0.1
D_UPDATES_PER_G_UPDATE = 1

# --- 2. Rede Geradora (Sem alterações) ---
class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

# --- 3. Rede Discriminadora (Sem alterações) ---
class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x):
        return self.model(x)

# --- 4. Inicialização dos pesos (Sem alterações) ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- 5. Visualização com Pygame (CORRIGIDA) ---
def update_display(screen, fixed_noise, epoch, generator, d_loss, g_loss):
    generator.eval()
    with torch.no_grad():
        img = generator(fixed_noise).detach().cpu()
    generator.train()

    img_grid = torchvision.utils.make_grid(img, normalize=True, nrow=4)
    # Converte para (Altura, Largura, Canais) - Padrão NumPy
    img_np = np.transpose(img_grid.numpy(), (1, 2, 0))
    
    # ################################################################## #
    # ## CORREÇÃO APLICADA AQUI ## #
    # Converte para array de inteiros e transpõe de (H, W, C) para (W, H, C) para o Pygame
    img_to_display = np.transpose(np.uint8(img_np * 255), (1, 0, 2))
    img_surface = pygame.surfarray.make_surface(img_to_display)
    # ################################################################## #

    img_surface_resized = pygame.transform.scale(img_surface, (DISPLAY_SIZE*2, DISPLAY_SIZE*2))
    screen.blit(img_surface_resized, (0, 0))
    caption = f"Epoch: {epoch} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}"
    pygame.display.set_caption(caption)
    pygame.display.flip()

# --- 6. Treinamento da GAN ---
def train():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if CHANNELS == 1 else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        dataset = ImageFolder('./imgs', transform=transform)
        if not dataset:
            raise ValueError("A pasta './imgs' está vazia ou não contém subpastas.")
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        print("Certifique-se que a estrutura é: ./imgs/subpasta/imagem.png")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)

    G = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
    D = Discriminator(CHANNELS).to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)
    d_losses, g_losses = [], []

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_SIZE*2, DISPLAY_SIZE*2))
    clock = pygame.time.Clock()

    print("Iniciando Loop de Treinamento Otimizado...")
    for epoch in range(1, EPOCHS + 1):
        # Usamos uma cópia da lista de lotes para poder usar tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        for i, (real_imgs, _) in enumerate(pbar):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("Treinamento interrompido pelo usuário.")
                    torch.save(G.state_dict(), "generator_interrupted.pth")
                    torch.save(D.state_dict(), "discriminator_interrupted.pth")
                    return

            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)

            # --- Treina D ---
            D.zero_grad()
            real_labels = torch.full((batch_size,), REAL_LABEL, device=DEVICE)
            fake_labels = torch.full((batch_size,), FAKE_LABEL, device=DEVICE)
            
            d_output_real = D(real_imgs).view(-1)
            loss_d_real = criterion(d_output_real, real_labels)

            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = G(noise)
            d_output_fake = D(fake_imgs.detach()).view(-1)
            loss_d_fake = criterion(d_output_fake, fake_labels)

            d_loss = loss_d_real + loss_d_fake
            d_loss.backward()
            opt_D.step()

            # --- Treina G ---
            if i % D_UPDATES_PER_G_UPDATE == 0:
                G.zero_grad()
                output_g = D(fake_imgs).view(-1)
                g_loss = criterion(output_g, real_labels)
                g_loss.backward()
                opt_G.step()
            else:
                g_loss = g_loss if 'g_loss' in locals() else torch.tensor(0)
            
            pbar.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if epoch % SAMPLE_INTERVAL == 0 or epoch == 1:
            update_display(screen, fixed_noise, epoch, G, d_loss.item(), g_loss.item())

    pygame.quit()
    torch.save(G.state_dict(), "generator_final.pth")
    torch.save(D.state_dict(), "discriminator_final.pth")
    print("Modelos salvos.")

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="D Loss")
    plt.plot(g_losses, label="G Loss")
    plt.xlabel("Épocas")
    plt.ylabel("Perda")
    plt.title("Evolução da Perda - GAN (Otimizado)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print(f"Usando dispositivo: {DEVICE}")
    train()