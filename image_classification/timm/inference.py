from urllib.request import urlopen

import timm
import torch
from PIL import Image

print("Hello, World from timm!")
print(timm.__version__)

model = timm.create_model("resnet18", pretrained=True)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

img = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)

with torch.inference_mode():
    output = model(transforms(img).unsqueeze(0))

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

print(top5_probabilities)
print(top5_class_indices)
