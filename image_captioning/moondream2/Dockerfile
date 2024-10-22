FROM nvcr.io/nvidia/pytorch:23.11-py3

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers \
    einops
    
COPY . /app

CMD ["python", "inference.py"]
