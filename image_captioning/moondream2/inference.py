# Ref: https://github.com/vikhyat/moondream
from urllib.request import urlopen

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
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
