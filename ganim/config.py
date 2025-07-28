# ganim/config.py
import torch

class GanimConfig:
    def __init__(self):
        # Configurações do Modelo e Treinamento
        self.imageSize = 64
        self.channels = 3
        self.latentDim = 100
        self.epochs = 5000
        self.batchSize = 64
        self.learningRate = 0.0002
        self.beta1 = 0.5
        
        # Lógica de Treino
        self.realLabel = 0.9
        self.fakeLabel = 0.1
        self.dUpdatesPerG = 1

        # Configurações de Ambiente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.workers = 2 if self.device.type == 'cuda' else 0
        
        # --- ATUALIZAÇÃO ---
        # Configurações de Saída e Visualização
        self.sampleInterval = 100      # A cada quantos epochs gerar uma prévia
        self.previewImageCount = 16    # Quantas imagens na prévia
        self.previewWindowSize = 512   # Tamanho da janela de prévia em pixels (ex: 512x512)
        self.finalWindowSize = 768     # Tamanho padrão para a janela final do ganim.show()

# Instância global que será usada pela biblioteca
settings = GanimConfig()