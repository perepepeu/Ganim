import torch


class GanimConfig:
    def __init__(self):
        self.imageSize = 64
        self.channels = 3
        self.latentDim = 100
        self.epochs = 5000
        self.batchSize = 64
        self.learningRate = 0.0002
        self.beta1 = 0.5

        self.realLabel = 0.9
        self.fakeLabel = 0.1
        self.dUpdatesPerG = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.workers = 2 if self.device.type == "cuda" else 0

        self.sampleInterval = 100
        self.previewImageCount = 16
        self.previewWindowSize = 512
        self.finalWindowSize = 768


settings = GanimConfig()
