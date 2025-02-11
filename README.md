# Image Generation Demo using ControlNet and FastAPI

## Overview
This repository provides a FastAPI-based application for image generation using a model from Hugging Face. It includes:
- **`awesomedemo_added_features.py`**: Contains the core logic for generating images.
- **`main.py`**: Implements a FastAPI server to expose the image generation functionality.
- **`index.html`**: Frontend UI to interact with the FastAPI app.
- **Dockerfile**: Configuration for containerizing the application.

---

## Prerequisites
Before proceeding, ensure you have the following installed:
- **Git**
- **Conda**
- **Docker**
- **NVIDIA GPU** (for GPU acceleration)

---

## Setup Instructions

### 1. Clone the ControlNet Repository & Create Virtual Environment
```sh
$ git clone https://github.com/lllyasviel/ControlNet.git
$ cd ControlNet
$ conda env create -f environment.yaml
$ conda activate control
```

### 2. Download the Model from Hugging Face
Download the **[`control_sd15_canny.pth`](https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_canny.pth)** model (5.71GB) and place it in the `ControlNet/.models` folder.

### 3. Download and Move Required Files
Download awesomedemo_added_feature.py, main.py, index.html and Dockerfile and move those to ControlNet/
Download mri_brain.jpg and move it to ControlNet/test_imgs/
```sh
$ mv path_to/awesomedemo_added_feature.py ControlNet/
$ mv path_to/main.py ControlNet/
$ mv path_to/index.html ControlNet/
$ mv path_to/Dockerfile ControlNet/
$ mv path_to/mri_brain.jpg ControlNet/test_imgs/
```
Note: Make sure to change the Dockerfile extension if it gets downloaded as text file.

### 4. Build & Run the Docker Container
```sh
$ docker build -t image-generate-demo .
$ docker run --gpus all -p 8000:8000 image-generate-demo
```

---

## Usage
Once the container is running, access the FastAPI app at:
- **API**: [http://localhost:8000/generate](http://localhost:8000/generate)
- **Frontend UI**: Open in a browser or preview in VScode after running docker container.

You can upload an image and receive a generated output based on the model's inference.

---

## Notes
- Ensure Docker is running with GPU support enabled.
- The model download may take time due to its large size.
- Modify `main.py` if you need to customize the API endpoints.

---

## Acknowledgments
- **[ControlNet](https://github.com/lllyasviel/ControlNet)** for providing the base implementation.
- **Hugging Face** for hosting the model.
- **Carl Zeiss AG** for basic logic.

Happy coding! 🚀
