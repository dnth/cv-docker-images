# Ref: https://github.com/vikhyat/moondream
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
enc_image = model.encode_image(image)
print(model.answer_question(enc_image, "Describe this image.", tokenizer))

# Batch inference
answers = model.batch_answer(
    images=[image, image],
    prompts=["Describe this image.", "Are there people in this image?"],
    tokenizer=tokenizer,
)

print(answers)
