# Ref: https://github.com/vikhyat/moondream
import time
from urllib.request import urlopen

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_id, revision, cache_dir="moondream2", use_flash_attention=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Device: {device}, Dtype: {dtype}")

    if use_flash_attention and device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        ).to(device=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        ).to(device=device)

    model.eval()
    model = torch.compile(model, mode="max-autotune")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    return model, tokenizer


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
    use_flash_attention = True

    model, tokenizer = load_model_and_tokenizer(
        model_id, revision, use_flash_attention=use_flash_attention
    )

    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    image = Image.open(urlopen(image_url))

    # Single inference
    prompt = "Describe this image."
    answer = single_inference(model, tokenizer, image, prompt)
    print(answer)

    # Batch inference
    batch_size = 10
    images = [image] * batch_size
    prompts = [prompt] * batch_size
    answers = batch_inference(model, tokenizer, images, prompts)
    print(answers)


if __name__ == "__main__":
    main()
