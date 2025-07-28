# ganim/utils.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import cv2

def weightsInit(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- ATUALIZAÇÃO ---
def show(images_tensor, window_title="Ganim", window_size=None):
    """Usa OpenCV para exibir um tensor de imagens, com controle de tamanho."""
    if not isinstance(images_tensor, torch.Tensor):
        raise TypeError("A entrada para 'show' deve ser um tensor do PyTorch.")

    images_tensor = images_tensor.detach().cpu()
    
    # Define o número de colunas do grid para ser a raiz quadrada da contagem de imagens
    nrow = int(np.sqrt(images_tensor.size(0)))
    grid = vutils.make_grid(images_tensor, padding=2, normalize=True, nrow=nrow)
    
    numpy_grid = np.transpose(grid.numpy(), (1, 2, 0))
    # Converte de RGB para BGR para o OpenCV
    bgr_grid = cv2.cvtColor(numpy_grid, cv2.COLOR_RGB2BGR)
    
    # Redimensiona a imagem para o tamanho da janela desejado
    if window_size:
        # Usa INTER_NEAREST para manter os pixels nítidos, ideal para pixel art
        bgr_grid = cv2.resize(bgr_grid, (window_size, window_size), interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_title, bgr_grid)
    print(f"Janela '{window_title}' aberta. Pressione qualquer tecla na janela para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot(history, title="Performance do Treino"):
    if not isinstance(history, dict) or 'g_loss' not in history or 'd_loss' not in history:
        raise ValueError("O histórico deve ser um dicionário com as chaves 'g_loss' e 'd_loss'.")
        
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(history['g_loss'], label="Perda do Gerador (G)")
    plt.plot(history['d_loss'], label="Perda do Discriminador (D)")
    plt.xlabel("Épocas")
    plt.ylabel("Perda")
    plt.legend()
    plt.grid(True)
    plt.show()