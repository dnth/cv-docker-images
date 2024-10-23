import os
from abc import ABCMeta

import PIL
import torch
from loguru import logger
from PIL.Image import Image
from vllm import LLM, SamplingParams


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

        logger.info(f"Initialized Molmo model from: {model_path}")
        logger.info(f"Using device: {self.device}, dtype: {self.dtype}")

    def run_inference(self, images: list[Image]) -> list[str]:
        prompt = "Describe the image."
        sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

        batch_inputs = [
            {
                "prompt": f"USER: <image>\n{prompt}\nASSISTANT:",
                "multi_modal_data": {"image": image.convert("RGB")},
            }
            for image in images
        ]

        results = self.model.generate(batch_inputs, sampling_params)
        return [output.outputs[0].text for output in results]


if __name__ == "__main__":
    # If model path does not exist, download the model from Hugging Face
    model_path = "molmo_7b_d_0924/"
    if not os.path.exists(model_path):
        model_path = "allenai/Molmo-7B-D-0924"

    model = Molmo(model_path)
    images = [PIL.Image.open("../../sample_images/0a6ee446579d2885.jpg")]
    print(model.run_inference(images))
