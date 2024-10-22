# Image Captioning with Moondream2
This repository contains a Dockerfile and a Python script for image captioning using the [Moondream2](https://github.com/vikhyat/moondream) model.

To use GPU, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

To run the model, you can build and run the Docker container as follows:

- Build Docker Image

```bash
docker build -t moondream2 .
```


- Run Docker Container

To use GPU, you need to run the container with the `--gpus` flag.
```bash
docker run --gpus all -it moondream2
```
