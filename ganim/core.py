# ganim/core.py
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .config import settings
from .models import Generator, Discriminator
from .utils import weightsInit, show

def setup(**kwargs):
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            print(f"Aviso: A configuração '{key}' não é reconhecida e será ignorada.")
    
    if 'device' in kwargs:
        settings.device = torch.device(kwargs['device'])
    print(f"Ganim configurado para usar o dispositivo: {settings.device}")

def fit(data):
    print("Iniciando o processo de treinamento 'fit'...")
    
    transform = transforms.Compose([
        transforms.Resize(settings.imageSize),
        transforms.CenterCrop(settings.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    try:
        dataset = ImageFolder(root=data, transform=transform)
        dataloader = DataLoader(dataset, batch_size=settings.batchSize, shuffle=True, num_workers=settings.workers)
    except Exception as e:
        print(f"Erro ao carregar o dataset de '{data}': {e}")
        return None, None

    netG = Generator(settings.latentDim, settings.channels, settings.imageSize).to(settings.device)
    netD = Discriminator(settings.channels, settings.imageSize).to(settings.device)
    netG.apply(weightsInit)
    netD.apply(weightsInit)

    criterion = torch.nn.BCEWithLogitsLoss()
    
    # --- ATUALIZAÇÃO ---
    # Usa a configuração 'previewImageCount' para o ruído fixo
    fixed_noise = torch.randn(settings.previewImageCount, settings.latentDim, 1, 1, device=settings.device)

    optimizerD = optim.Adam(netD.parameters(), lr=settings.learningRate, betas=(settings.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=settings.learningRate, betas=(settings.beta1, 0.999))

    g_losses = []
    d_losses = []

    print("Iniciando loop de treino...")
    for epoch in range(settings.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{settings.epochs}")
        for i, (real_data, _) in enumerate(pbar):
            netD.zero_grad()
            real_cpu = real_data.to(settings.device)
            b_size = real_cpu.size(0)
            
            label_real = torch.full((b_size,), settings.realLabel, device=settings.device)
            output_real = netD(real_cpu).view(-1)
            errD_real = criterion(output_real, label_real)
            
            noise = torch.randn(b_size, settings.latentDim, 1, 1, device=settings.device)
            fake = netG(noise)
            label_fake = torch.full((b_size,), settings.fakeLabel, device=settings.device)
            output_fake = netD(fake.detach()).view(-1)
            errD_fake = criterion(output_fake, label_fake)

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            if i % settings.dUpdatesPerG == 0:
                netG.zero_grad()
                output = netD(fake).view(-1)
                errG = criterion(output, label_real)
                errG.backward()
                optimizerG.step()
            
            pbar.set_postfix(D_loss=errD.item(), G_loss=errG.item())

        g_losses.append(errG.item())
        d_losses.append(errD.item())

        # --- ATUALIZAÇÃO ---
        # Usa as novas configs para mostrar a prévia
        if (epoch + 1) % settings.sampleInterval == 0:
            with torch.no_grad():
                preview_images = netG(fixed_noise).detach().cpu()
            # Passa o tamanho da janela de prévia para a função show!
            show(
                preview_images, 
                f"Ganim - Preview Epoch {epoch+1}",  # Corrigido para não usar acentos
                window_size=settings.previewWindowSize
            )

    history = {'d_loss': d_losses, 'g_loss': g_losses}
    print("Treinamento concluído.")
    return netG, history

def save(model, path="ganim_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Modelo salvo em: {path}")

def load(path):
    model = Generator(settings.latentDim, settings.channels, settings.imageSize).to(settings.device)
    model.load_state_dict(torch.load(path, map_location=settings.device))
    model.eval()
    print(f"Modelo carregado de: {path}")
    return model

def sample(model, count=1):
    noise = torch.randn(count, settings.latentDim, 1, 1, device=settings.device)
    with torch.no_grad():
        images = model(noise)
    print(f"{count} imagens geradas.")
    return images