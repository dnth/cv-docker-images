from abc import ABCMeta

import torch
from PIL import Image as PILImage


class BaseGPUModel(metaclass=ABCMeta):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

    def run_inference(self, images: list[PILImage.Image]):
        raise NotImplementedError()
