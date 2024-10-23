# Molmo
Setup and run Molmo-7B-D-0924 for image captioning.

## Installation

You can run the model with pip or Docker.

### Pip

Make sure you have Python 3.10 or higher and cuda 12.1 or higher.

First, install the requirements:

```bash
pip install -r requirements.txt
```

Run the inference script:
```python
python inference.py
```

### Docker

Build the image:
```bash
docker build -t molmo .
```

Run the container:

```bash
docker run --gpus all -it molmo
```

## Run from Docker Hub
Alternatively, you can run the container from Docker Hub without going through the hassle of building the image yourself:

```bash
docker run --gpus all -it dnth/molmo
```

Note: The model weights (32GB) is part of the container image, so it might take a while to download. 
