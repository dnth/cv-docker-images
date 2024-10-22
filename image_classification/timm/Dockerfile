FROM nvcr.io/nvidia/pytorch:23.11-py3

WORKDIR /app

RUN pip install --no-cache-dir \
    timm 
    
COPY . /app

CMD ["python", "inference.py"]
