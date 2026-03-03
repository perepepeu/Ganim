import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .config import settings
from .models import Generator, Discriminator
from .utils import show, weights_init


def setup(**kwargs):
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            print(f"Aviso: A configuracao '{key}' nao e reconhecida e sera ignorada.")

    if "device" in kwargs:
        settings.device = torch.device(kwargs["device"])
    print(f"Ganim configurado para usar o dispositivo: {settings.device}")


def fit(data):
    print("Iniciando o processo de treinamento 'fit'...")

    transform = transforms.Compose([
        transforms.Resize(settings.image_size),
        transforms.CenterCrop(settings.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    try:
        dataset = ImageFolder(root=data, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=settings.workers,
        )
    except Exception as exc:
        print(f"Erro ao carregar o dataset de '{data}': {exc}")
        return None, None

    generator = Generator(settings.latent_dim, settings.channels, settings.image_size).to(settings.device)
    discriminator = Discriminator(settings.channels, settings.image_size).to(settings.device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = torch.nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(
        settings.preview_image_count,
        settings.latent_dim,
        1,
        1,
        device=settings.device,
    )

    optimizer_discriminator = optim.Adam(
        discriminator.parameters(),
        lr=settings.learning_rate,
        betas=(settings.beta1, 0.999),
    )
    optimizer_generator = optim.Adam(
        generator.parameters(),
        lr=settings.learning_rate,
        betas=(settings.beta1, 0.999),
    )

    g_losses = []
    d_losses = []

    print("Iniciando loop de treino...")
    for epoch in range(settings.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{settings.epochs}")
        for batch_index, (real_data, _) in enumerate(pbar):
            discriminator.zero_grad()
            real_images = real_data.to(settings.device)
            batch_size = real_images.size(0)
            
            label_real = torch.full((batch_size,), settings.real_label, device=settings.device)
            output_real = discriminator(real_images).view(-1)
            err_discriminator_real = criterion(output_real, label_real)
            
            noise = torch.randn(batch_size, settings.latent_dim, 1, 1, device=settings.device)
            fake_images = generator(noise)
            label_fake = torch.full((batch_size,), settings.fake_label, device=settings.device)
            output_fake = discriminator(fake_images.detach()).view(-1)
            err_discriminator_fake = criterion(output_fake, label_fake)

            err_discriminator = err_discriminator_real + err_discriminator_fake
            err_discriminator.backward()
            optimizer_discriminator.step()

            if batch_index % settings.d_updates_per_g == 0:
                generator.zero_grad()
                output_generator = discriminator(fake_images).view(-1)
                err_generator = criterion(output_generator, label_real)
                err_generator.backward()
                optimizer_generator.step()
            
            pbar.set_postfix(D_loss=err_discriminator.item(), G_loss=err_generator.item())

        g_losses.append(err_generator.item())
        d_losses.append(err_discriminator.item())

        if (epoch + 1) % settings.sample_interval == 0:
            with torch.no_grad():
                preview_images = generator(fixed_noise).detach().cpu()
            show(
                preview_images,
                f"Ganim - Preview Epoch {epoch+1}",
                window_size=settings.preview_window_size,
            )

    history = {"d_loss": d_losses, "g_loss": g_losses}
    print("Treinamento concluido.")
    return generator, history


def save(model, path="ganim_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Modelo salvo em: {path}")


def load(path):
    model = Generator(settings.latent_dim, settings.channels, settings.image_size).to(settings.device)
    model.load_state_dict(torch.load(path, map_location=settings.device))
    model.eval()
    print(f"Modelo carregado de: {path}")
    return model


def sample(model, count=1):
    noise = torch.randn(count, settings.latent_dim, 1, 1, device=settings.device)
    with torch.no_grad():
        images = model(noise)
    print(f"{count} imagens geradas.")
    return images
