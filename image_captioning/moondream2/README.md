# Image Captioning with Moondream2
This repository contains a Dockerfile and a Python script for image captioning using the [Moondream2](https://github.com/vikhyat/moondream) model.

Prerequisite:
- Docker.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

> [!NOTE]
> Only tested on Linux machine with GPU.

To run this model, you can build the Docker container yourself or use the pre-built image from Docker Hub.

## Build the Docker image

Download the flash attention wheel

```bash
wget https://github.com/dnth/cv-docker-images/releases/download/v0.0.0/flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl
```

Build image:

```bash
docker build -t moondream2 .
```

Run container after build completes:

To use GPU, you need to run the container with the `--gpus` flag.
```bash
docker run --gpus all -it moondream2
```


## Run from Docker Hub
Optionally, you can pull the pre-built image from Docker Hub without having to build the image yourself.

```bash
docker run --gpus all -it dnth/moondream2
```
