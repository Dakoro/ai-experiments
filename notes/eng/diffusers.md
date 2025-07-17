# Diffusers Notebook Report

## Overview
This notebook demonstrates the implementation and training of diffusion models for image generation, specifically focusing on denoising diffusion probabilistic models (DDPMs) using the MNIST dataset. The notebook progresses from a basic custom UNet implementation to using the HuggingFace Diffusers library.

## Key Components

### 1. Data Preparation
- **Dataset**: MNIST handwritten digits (28x28 grayscale images)
- **Batch Size**: 128 for training
- **Data Loading**: Uses PyTorch DataLoader with shuffling enabled

### 2. Noise Corruption Function
```python
def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount
```
This function adds uniform noise to clean images, creating a linear interpolation between the original image and random noise.

### 3. Basic UNet Architecture
- **Custom Implementation**: A minimal UNet with 309,057 parameters
- **Structure**:
  - 3 downsampling layers (32, 64, 64 channels)
  - 3 upsampling layers with skip connections
  - SiLU activation function
  - MaxPool2d for downsampling, Upsample for upsampling

### 4. Training Process
- **Loss Function**: MSE loss between predicted clean images and ground truth
- **Optimizer**: Adam with learning rate 1e-3
- **Training Strategy**: 
  - Apply random noise corruption to clean images
  - Train model to predict the original clean image from noisy input
  - 3 epochs of training

### 5. Sampling Strategy
The notebook demonstrates two sampling approaches:
- **5-step sampling**: Iteratively moves from random noise toward clean images
- **40-step sampling**: More refined sampling with better quality results

### 6. HuggingFace Diffusers Integration
- **UNet2DModel**: Professional implementation with 1.7M parameters
- **Enhanced Features**:
  - Attention mechanisms (AttnDownBlock2D, AttnUpBlock2D)
  - ResNet blocks with GroupNorm
  - Time embedding for proper timestep conditioning
  - More sophisticated architecture with skip connections

### 7. DDPM Scheduler
- **DDPMScheduler**: Implements the standard DDPM noise schedule
- **Noise Schedule**: Uses 1000 timesteps with proper α and β scheduling
- **Visualization**: Shows the noise schedule curves for √α̅_t and √(1-α̅_t)

## Results and Observations

### Model Performance
1. **Basic UNet**: Successfully learns to denoise MNIST images with simple architecture
2. **Diffusers UNet**: Better quality generation with attention mechanisms and proper time conditioning
3. **Sampling Quality**: 40-step sampling produces cleaner, more recognizable digits compared to 5-step sampling

### Training Characteristics
- Both models converge within 3 epochs
- Loss curves show steady improvement
- Generated samples demonstrate the model's ability to learn the MNIST digit distribution

## Technical Implementation Details

### Hardware
- CUDA-enabled GPU acceleration
- Efficient batch processing for training and inference

### Libraries Used
- PyTorch for neural network implementation
- Torchvision for MNIST dataset and image utilities
- HuggingFace Diffusers for professional diffusion model components
- Matplotlib for visualization

### Key Concepts Demonstrated
1. **Diffusion Process**: Forward noising and reverse denoising
2. **UNet Architecture**: Encoder-decoder with skip connections
3. **Attention Mechanisms**: Spatial self-attention for better feature learning
4. **Noise Scheduling**: Proper timestep-based noise addition
5. **Iterative Sampling**: Gradual denoising from random noise to clean images

## Practical Applications
This notebook serves as an educational foundation for understanding:
- Diffusion model principles
- Image generation techniques
- UNet architectures for computer vision
- Professional ML library integration (HuggingFace Diffusers)

The implementation demonstrates both the theoretical concepts and practical implementation details necessary for building diffusion models for image generation tasks.