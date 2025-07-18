# TransUNet
This repo holds code for [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)

## 📰 News

- [10/15/2023] 🔥 3D version of TransUNet is out! Our 3D TransUNet surpasses nn-UNet with 88.11% Dice score on the BTCV dataset and outperforms the top-1 solution in the BraTs 2021 challenge. Please take a look at the [code](https://github.com/Beckschen/3D-TransUNet/tree/main) and [paper](https://arxiv.org/abs/2310.07781).

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets/README.md) for details, or use the [preprocessed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing) and [data2](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4?usp=drive_link) for research purposes.

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Citations

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```
# Coronary Artery Segmentation with 3D U-Net

A deep learning project for 3D coronary artery segmentation using U-Net architecture with PyTorch and MONAI.

## Overview

This project implements a 3D U-Net model for segmenting coronary arteries from medical imaging data. The implementation uses the MONAI framework for medical image processing and PyTorch for deep learning.

## Features

- **3D U-Net Architecture**: Supports both BasicUNet and standard U-Net configurations
- **Comprehensive Data Augmentation**: Random rotations, flips, zooming, intensity adjustments
- **Flexible Training Pipeline**: Customizable ROI sizes, batch sizes, and training parameters
- **Evaluation Metrics**: Dice score calculation for segmentation quality assessment
- **NIfTI Support**: Save predictions as NIfTI files with original metadata
- **Visualization**: Built-in visualization of training progress and predictions

## Requirements

```bash
pip install pytorch-ignite monai nibabel matplotlib torch torchvision
```

## Dataset Structure

The code expects NIfTI files (.nii.gz) with the following naming convention:
- Images: `{number}.img.nii.gz`
- Labels: `{number}.label.nii.gz`

Example:
```
dataset/
├── 001.img.nii.gz
├── 001.label.nii.gz
├── 002.img.nii.gz
├── 002.label.nii.gz
...
```

## Usage

### Basic Training

```python
from coronary_segmentation import CoronaryArterySegmentation

# Create model instance
model = CoronaryArterySegmentation(
    roi=(96, 96, 96), 
    batch_size=1, 
    use_basic_unet=True
)

# Train the model
model.fit(train_loader, val_loader, num_epochs=10)

# Evaluate on validation set
dice_scores = model.predict(val_loader, num_samples=5)
```

### Configuration Options

- **ROI Size**: `(96, 96, 96)` - Region of interest dimensions
- **Batch Size**: `1` - Training batch size
- **Model Type**: `use_basic_unet=True` - Choose between BasicUNet and standard U-Net
- **Augmentation**: Comprehensive set of 3D augmentations included

### Data Augmentation

The pipeline includes:
- Random rotations (±30°)
- Random flips on all axes
- Random zooming (0.9-1.1x)
- Random affine transformations
- Intensity adjustments (contrast, shift, scale)
- Gaussian noise addition

## Model Architecture

### BasicUNet Configuration
- **Spatial Dimensions**: 3D
- **Input Channels**: 1 (grayscale)
- **Output Channels**: 2 (background + artery)
- **Features**: (32, 64, 128, 256, 512, 32)
- **Dropout**: 0.1

### Standard U-Net Configuration
- **Channels**: (16, 32, 64, 128)
- **Strides**: (2, 2, 2)
- **Residual Units**: 2 per block
- **Normalization**: Batch normalization

## Training Process

1. **Data Loading**: Automatic pairing of image and label files
2. **Preprocessing**: Spatial padding, intensity normalization, resampling
3. **Augmentation**: Random transformations during training
4. **Training**: Dice loss optimization with Adam optimizer
5. **Validation**: Dice score evaluation on validation set
6. **Model Saving**: Best model based on validation Dice score

## Evaluation

The model is evaluated using:
- **Dice Score**: Primary metric for segmentation quality
- **Visual Inspection**: Side-by-side comparison of predictions vs ground truth
- **NIfTI Export**: Save predictions for external analysis

## Results Visualization

The training process includes:
- Real-time loss monitoring
- Validation Dice score tracking
- Training history plots
- Prediction visualizations with overlays



## Performance Notes

- **Memory Usage**: Optimized for single batch processing
- **GPU Support**: Automatic CUDA detection and usage
- **Scalability**: Configurable ROI sizes for different memory constraints
- **Reproducibility**: Consistent results with proper random seeding







