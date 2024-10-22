# Ref: https://github.com/vikhyat/moondream
import time
from urllib.request import urlopen

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision, cache_dir="moondream2"
).to(device=device, dtype=dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)

# Single inference
start_time = time.time()
enc_image = model.encode_image(image)
answer = model.answer_question(enc_image, "Describe this image.", tokenizer)
single_inference_time = time.time() - start_time

print(answer)
print(f"Single inference time: {single_inference_time:.2f} seconds")

# Batch inference
start_time = time.time()


batch_size = 10
images = [image] * batch_size
prompts = ["Describe this image."] * batch_size

answers = model.batch_answer(
    images=images,
    prompts=prompts,
    tokenizer=tokenizer,
)
batch_inference_time = time.time() - start_time

print(answers)
print(
    f"Batch inference time: {batch_inference_time:.2f} seconds for {batch_size} images"
)
