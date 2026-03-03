import torch


class GanimConfig:
    _legacy_keys = {
        "imageSize": "image_size",
        "latentDim": "latent_dim",
        "batchSize": "batch_size",
        "learningRate": "learning_rate",
        "realLabel": "real_label",
        "fakeLabel": "fake_label",
        "dUpdatesPerG": "d_updates_per_g",
        "sampleInterval": "sample_interval",
        "previewImageCount": "preview_image_count",
        "previewWindowSize": "preview_window_size",
        "finalWindowSize": "final_window_size",
    }

    def __setattr__(self, name, value):
        mapped_name = type(self)._legacy_keys.get(name, name)
        super().__setattr__(mapped_name, value)

    def __getattr__(self, name):
        mapped_name = type(self)._legacy_keys.get(name)
        if mapped_name:
            return object.__getattribute__(self, mapped_name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __init__(self):
        self.image_size = 64
        self.channels = 3
        self.latent_dim = 100
        self.epochs = 5000
        self.batch_size = 64
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        self.real_label = 0.9
        self.fake_label = 0.1
        self.d_updates_per_g = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.workers = 2 if self.device.type == "cuda" else 0

        self.sample_interval = 100
        self.preview_image_count = 16
        self.preview_window_size = 512
        self.final_window_size = 768


settings = GanimConfig()
