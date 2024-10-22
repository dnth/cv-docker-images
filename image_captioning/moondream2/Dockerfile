FROM nvcr.io/nvidia/pytorch:23.11-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
    
COPY . /app

# Copy model files to be part of the Docker image
COPY moondream2 /app/moondream2

CMD ["python", "inference.py"]
