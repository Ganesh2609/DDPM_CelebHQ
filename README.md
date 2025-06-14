# Denoising Diffusion Probabilistic Model for CelebA-HQ

<p align="center">
  <video src="https://github.com/user-attachments/assets/466f8070-5d4d-49aa-8ecc-f99911ac1c88" width="512" controls autoplay loop muted>
</p>

This project is an implementation of a Denoising Diffusion Probabilistic Model (DDPM) for generating high-quality celebrity faces. The model is trained on the CelebA-HQ dataset and utilizes a U-Net architecture with attention mechanisms to progressively denoise an image from pure noise to a clean, realistic face.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Results](#results)
  - [Diffusion Process Video](#diffusion-process-video)
  - [Generated Image Samples Over Training Epochs](#generated-image-samples-over-training-epochs)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Generating a Diffusion Video](#generating-a-diffusion-video)
- [Training Logs](#training-logs)
- [Project Structure](#project-structure)

## Overview

This repository provides a complete pipeline for training a DDPM on the CelebA-HQ dataset and generating new images. The core idea behind DDPMs is to reverse a forward diffusion process that gradually adds Gaussian noise to an image, until it becomes pure noise. The model learns to predict the noise at each timestep, and by iteratively subtracting the predicted noise, we can generate a clean image from a random noise vector.

## Model Architecture

The model architecture is based on a U-Net, which is a popular choice for image-to-image tasks due to its encoder-decoder structure with skip connections. This allows the model to capture both high-level and low-level features effectively.

Key components of the architecture include:

-   **U-Net Structure:** The model consists of down-sampling blocks, a bottleneck, and up-sampling blocks.
-   **Time Embeddings:** Sinusoidal embeddings are used to encode the timestep `t`, which are then passed through a small MLP to get a time embedding vector. This vector is injected into the residual blocks of the U-Net to make the model aware of the current noise level.
-   **Attention Mechanism:** Self-attention blocks are incorporated at various resolutions in the U-Net to help the model capture long-range dependencies and generate more coherent images.
-   **Residual Connections:** Residual connections are used throughout the network to facilitate gradient flow and improve training stability.

You can find the detailed implementation in `model.py` and `model_utils.py`.

## Results

### Generated Image Samples Over Training Epochs

Here are some sample images generated by the model at different stages of training. This shows how the model's ability to generate realistic faces improves over time.

<p align="center">
  <b>Epoch 1</b><br>
  <img src="Train Data/Results/train 1 celeb/Epoch_1.png" width="512">
</p>
<p align="center">
  <b>Epoch 10</b><br>
  <img src="Train Data/Results/train 1 celeb/Epoch_10.png" width="512">
</p>
<p align="center">
  <b>Epoch 20</b><br>
  <img src="Train Data/Results/train 1 celeb/Epoch_20.png" width="512">
</p>
<p align="center">
  <b>Epoch 30</b><br>
  <img src="Train Data/Results/train 1 celeb/Epoch_30.png" width="512">
</p>
<p align="center">
  <b>Epoch 40</b><br>
  <img src="Train Data/Results/train 1 celeb/Epoch_40.png" width="512">
</p>
<p align="center">
  <b>Epoch 45</b><br>
  <img src="Train Data/Results/train 1 celeb/Epoch_45.png" width="512">
</p>

## Dataset

The model is trained on the **CelebA-HQ** dataset, which contains 30,000 high-resolution celebrity face images. The `dataset.py` script handles loading, transforming, and creating data loaders for the training and validation sets.

## Setup and Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/DDPM_CelebHQ.git](https://github.com/your-username/DDPM_CelebHQ.git)
    cd DDPM_CelebHQ
    ```

2.  Install the required dependencies. It is recommended to use a virtual environment:
    ```bash
    pip install torch torchvision Pillow opencv-python tqdm numpy matplotlib
    ```

## Usage

### Training the Model

1.  Download the CelebA-HQ dataset and place it in a directory.
2.  Update the `data_root` variable in `training_ddpm.py` to point to your dataset directory.
3.  Run the training script:
    ```bash
    python training_ddpm.py
    ```
You can customize the training parameters such as `img_size`, `batch_size`, `learning_rate`, `num_epochs`, etc., directly in the `training_ddpm.py` script. The script also supports resuming training from a checkpoint.

### Generating a Diffusion Video

To generate a video of the diffusion process, you can use the `generate_and_save.py` script.

1.  Update the `checkpoint_path` in `generate_and_save.py` to point to your trained model checkpoint (e.g., `Train Data/Checkpoints/train 1 celeb/model_epoch_46.pth`).
2.  Run the script:
    ```bash
    python generate_and_save.py
    ```
This will create a video file named `video_2.mp4` in the `Results` directory.

## Training Logs

The training progress is logged to both the console and a log file. You can find the log files in the `Train Data/Logs/` directory. The logger records training and testing losses, as well as other important information.

Here is a sample from the `train_1_celeb.log` file:
```
2025-06-12 17:21:08,942 - INFO - Training Loss for Epoch 46: 0.0182
2025-06-12 17:21:31,456 - INFO - Epoch 46 Average Testing Loss: 0.0179
2025-06-12 17:21:31,845 - INFO - Checkpoint (Epoch 46) saved to Train Data/Checkpoints/train 1 celeb\model_epoch_46.pth
```

## Project Structure
```
DDPM_CelebHQ/
├── model.py                # U-Net model definition
├── model_utils.py          # Helper modules for the U-Net model
├── training_utils.py       # Diffusion helper class
├── dataset.py              # Data loading and preprocessing
├── trainer.py              # Main training class
├── training_ddpm.py        # Script to run the training
├── generate_and_save.py    # Script to generate diffusion video
├── logger.py               # Logging utility
├── Train Data/
│   ├── Checkpoints/        # Saved model checkpoints
│   ├── Logs/               # Training logs
│   └── Results/            # Generated images and loss plots
├── Old Train Data/         # Old training data
└── README.md               # This file
```