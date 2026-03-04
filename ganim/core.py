import sys

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .config import settings
from .models import Generator, Discriminator
from .utils import show, weights_init


def _resolve_device(device_value):
    try:
        requested_device = torch.device(device_value)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Dispositivo invalido '{device_value}': {exc}") from exc

    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("Aviso: CUDA solicitado, mas nao esta disponivel. Usando CPU.")
        return torch.device("cpu")

    mps_backend = getattr(torch.backends, "mps", None)
    if requested_device.type == "mps" and (mps_backend is None or not mps_backend.is_available()):
        print("Aviso: MPS solicitado, mas nao esta disponivel. Usando CPU.")
        return torch.device("cpu")

    return requested_device


def _validate_settings():
    errors = []

    if settings.channels not in (1, 3):
        errors.append("`channels` deve ser 1 (cinza) ou 3 (RGB).")
    if settings.sample_interval < 1:
        errors.append("`sample_interval` deve ser maior ou igual a 1.")
    if settings.d_updates_per_g < 1:
        errors.append("`d_updates_per_g` deve ser maior ou igual a 1.")
    if settings.batch_size < 1:
        errors.append("`batch_size` deve ser maior ou igual a 1.")
    if settings.epochs < 1:
        errors.append("`epochs` deve ser maior ou igual a 1.")
    if settings.preview_image_count < 1:
        errors.append("`preview_image_count` deve ser maior ou igual a 1.")
    if settings.image_size < 1:
        errors.append("`image_size` deve ser maior ou igual a 1.")

    if errors:
        raise ValueError("Configuracao invalida:\n- " + "\n- ".join(errors))


def _build_transform():
    transform_steps = [
        transforms.Resize(settings.image_size),
        transforms.CenterCrop(settings.image_size),
    ]

    if settings.channels == 1:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
        mean = (0.5,)
        std = (0.5,)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(transform_steps)


def _extract_model_config(model):
    latent_dim = int(getattr(model, "latent_dim", settings.latent_dim))
    channels = int(getattr(model, "channels", settings.channels))
    image_size = int(getattr(model, "image_size", settings.image_size))

    layers = list(getattr(model, "model", []))
    if layers:
        first_layer = layers[0]
        if hasattr(first_layer, "in_channels"):
            latent_dim = int(first_layer.in_channels)

        conv_transpose_layers = [layer for layer in layers if layer.__class__.__name__ == "ConvTranspose2d"]
        if conv_transpose_layers:
            last_conv = conv_transpose_layers[-1]
            if hasattr(last_conv, "out_channels"):
                channels = int(last_conv.out_channels)
            if hasattr(last_conv, "in_channels"):
                image_size = int(last_conv.in_channels)

    return {
        "latent_dim": latent_dim,
        "channels": channels,
        "image_size": image_size,
    }


def setup(**kwargs):
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            print(f"Aviso: A configuracao '{key}' nao e reconhecida e sera ignorada.")

    if "device" in kwargs:
        settings.device = _resolve_device(kwargs["device"])
    else:
        settings.device = _resolve_device(settings.device)

    if getattr(settings, "progress_update_interval", 1) < 1:
        settings.progress_update_interval = 1
    if getattr(settings, "workers", 0) < 0:
        settings.workers = 0

    _validate_settings()
    print(f"Ganim configurado para usar o dispositivo: {settings.device}")


def fit(data):
    print("Iniciando o processo de treinamento 'fit'...")

    settings.device = _resolve_device(settings.device)
    _validate_settings()
    transform = _build_transform()
    try:
        dataset = ImageFolder(root=data, transform=transform)
        if len(dataset) == 0:
            raise ValueError("Dataset vazio.")

        dataloader_kwargs = {
            "batch_size": settings.batch_size,
            "shuffle": True,
            "num_workers": settings.workers,
            "pin_memory": settings.device.type == "cuda",
        }
        if settings.workers > 0:
            dataloader_kwargs["persistent_workers"] = True
        dataloader = DataLoader(dataset, **dataloader_kwargs)
    except Exception as exc:
        print(f"Erro ao carregar o dataset de '{data}': {exc}")
        return None, None

    if settings.device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    generator = Generator(settings.latent_dim, settings.channels, settings.image_size).to(settings.device)
    discriminator = Discriminator(settings.channels, settings.image_size).to(settings.device)
    generator.latent_dim = settings.latent_dim
    generator.channels = settings.channels
    generator.image_size = settings.image_size
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
    progress_update_interval = max(1, int(settings.progress_update_interval))
    show_progress = sys.stdout.isatty()
    batches_per_epoch = len(dataloader)
    err_generator = torch.tensor(0.0, device=settings.device)
    err_discriminator = torch.tensor(0.0, device=settings.device)

    print("Iniciando loop de treino...")
    for epoch in range(settings.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{settings.epochs}", disable=not show_progress)
        for batch_index, (real_data, _) in enumerate(pbar):
            optimizer_discriminator.zero_grad(set_to_none=True)
            real_images = real_data.to(settings.device, non_blocking=True)
            batch_size = real_images.size(0)

            output_real = discriminator(real_images).view(-1)
            label_real = torch.full_like(output_real, settings.real_label, device=settings.device)
            err_discriminator_real = criterion(output_real, label_real)

            noise = torch.randn(batch_size, settings.latent_dim, 1, 1, device=settings.device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach()).view(-1)
            label_fake = torch.full_like(output_fake, settings.fake_label, device=settings.device)
            err_discriminator_fake = criterion(output_fake, label_fake)

            err_discriminator = err_discriminator_real + err_discriminator_fake
            err_discriminator.backward()
            optimizer_discriminator.step()

            if batch_index % settings.d_updates_per_g == 0:
                optimizer_generator.zero_grad(set_to_none=True)
                output_generator = discriminator(fake_images).view(-1)
                label_generator = torch.full_like(output_generator, settings.real_label, device=settings.device)
                err_generator = criterion(output_generator, label_generator)
                err_generator.backward()
                optimizer_generator.step()

            if (batch_index + 1) % progress_update_interval == 0 or (batch_index + 1) == batches_per_epoch:
                pbar.set_postfix(D_loss=err_discriminator.item(), G_loss=err_generator.item())

        g_losses.append(err_generator.item())
        d_losses.append(err_discriminator.item())

        if (epoch + 1) % settings.sample_interval == 0:
            was_training = generator.training
            generator.eval()
            with torch.inference_mode():
                preview_images = generator(fixed_noise).detach().cpu()
            if was_training:
                generator.train()
            show(
                preview_images,
                f"Ganim - Preview Epoch {epoch+1}",
                window_size=settings.preview_window_size,
            )

    history = {"d_loss": d_losses, "g_loss": g_losses}
    print("Treinamento concluido.")
    return generator, history


def save(model, path="ganim_model.pth"):
    model_config = _extract_model_config(model)
    payload = {
        "format": "ganim_generator",
        "format_version": 1,
        "config": model_config,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    print(f"Modelo salvo em: {path}")


def load(path):
    settings.device = _resolve_device(settings.device)
    checkpoint = torch.load(path, map_location=settings.device)

    model_config = {
        "latent_dim": settings.latent_dim,
        "channels": settings.channels,
        "image_size": settings.image_size,
    }
    state_dict = checkpoint

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        saved_config = checkpoint.get("config", {})
        if isinstance(saved_config, dict):
            for key in model_config:
                if key in saved_config:
                    try:
                        model_config[key] = int(saved_config[key])
                    except (TypeError, ValueError):
                        pass
        state_dict = checkpoint["state_dict"]

    model = Generator(
        model_config["latent_dim"],
        model_config["channels"],
        model_config["image_size"],
    ).to(settings.device)
    model.load_state_dict(state_dict)
    model.latent_dim = model_config["latent_dim"]
    model.channels = model_config["channels"]
    model.image_size = model_config["image_size"]
    model.eval()
    print(f"Modelo carregado de: {path}")
    return model


def sample(model, count=1):
    if count < 1:
        raise ValueError("`count` deve ser maior ou igual a 1.")

    latent_dim = int(getattr(model, "latent_dim", settings.latent_dim))
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = settings.device

    noise = torch.randn(count, latent_dim, 1, 1, device=model_device)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        images = model(noise)
    if was_training:
        model.train()
    print(f"{count} imagens geradas.")
    return images
