FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
    
COPY . /app

# Copy model files to be part of the Docker image
COPY moondream2 /app/moondream2

CMD ["python", "inference.py"]
