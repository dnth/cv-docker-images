# Image Captioning with Moondream2
This repository contains a Dockerfile and a Python script for image captioning using the [Moondream2](https://github.com/vikhyat/moondream) model.

To run the model, you can build and run the Docker container as follows:

- Build Docker Image

```bash
docker build -t moondream2 .
```


- Run Docker Container

```bash
docker run --gpus all -it moondream2
```
