import os
from abc import ABCMeta

import torch
from PIL.Image import Image
from vllm import LLM


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

    def run_inference(self, images: list[Image]):
        raise NotImplementedError()


class Molmo(BaseGPUModel):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model = LLM(model=model_path, trust_remote_code=True, dtype=self.dtype)

    def run_inference(self, images: list[Image]) -> list[str]:
        print(f"Running inference on {len(images)} images")


if __name__ == "__main__":
    # If model path does not exist, download the model from Hugging Face
    model_path = "molmo_7b_d_0924/"
    if not os.path.exists(model_path):
        model_path = "allenai/Molmo-7B-D-0924"

    model = Molmo(model_path)
    model.run_inference([])
