# Molmo
Setup and run Molmo-7B-D-0924 for image captioning.

## Installation

With pip:

```bash
pip install -r requirements.txt
```

```python
python inference.py
```

With Docker:

```bash
docker build -t molmo .
docker run --gpus all -it molmo
```

## Run from Docker Hub

```bash
docker run --gpus all -it dnth/molmo
```
