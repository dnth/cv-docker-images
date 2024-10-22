# Ref: https://github.com/vikhyat/moondream
import time
from urllib.request import urlopen

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id, revision, cache_dir="moondream2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Device: {device}, Dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision, cache_dir=cache_dir
    ).to(device=device, dtype=dtype)
    model.eval()
    model = torch.compile(model, mode="max-autotune")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    return model, tokenizer


def load_image_from_url(url):
    return Image.open(urlopen(url))


def single_inference(model, tokenizer, image, question):
    start_time = time.time()
    enc_image = model.encode_image(image)
    answer = model.answer_question(enc_image, question, tokenizer, max_new_tokens=20)
    inference_time = time.time() - start_time
    print(f"Single inference time: {inference_time:.2f} seconds")
    return answer


def batch_inference(model, tokenizer, images, prompts):
    start_time = time.time()
    answers = model.batch_answer(
        images=images, prompts=prompts, tokenizer=tokenizer, max_new_tokens=20
    )
    inference_time = time.time() - start_time
    print(
        f"Batch inference time: {inference_time:.2f} seconds for {len(images)} images"
    )
    return answers


def main():
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"

    model, tokenizer = load_model_and_tokenizer(model_id, revision)

    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    image = load_image_from_url(image_url)

    # Single inference
    question = "Describe this image."
    answer = single_inference(model, tokenizer, image, question)
    print(answer)

    # Batch inference
    batch_size = 10
    images = [image] * batch_size
    prompts = [question] * batch_size
    answers = batch_inference(model, tokenizer, images, prompts)
    print(answers)


if __name__ == "__main__":
    main()
