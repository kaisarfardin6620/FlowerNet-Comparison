# Flower Classification using Deep Learning

This repository contains implementations of flower classification using deep learning models.

## Overview

The project provides two different approaches to flower classification:

1. A single-model implementation using InceptionV3
2. A comprehensive comparison of 9 different CNN architectures

## Single Model Implementation (`flower_images.py`)

This script implements a focused approach using InceptionV3 as the base model.

### Key Features

- Transfer learning with InceptionV3 pre-trained on ImageNet
- Custom classification head with multiple dense layers
- Advanced data augmentation pipeline
- Adaptive learning rate scheduling
- Early stopping with best weights restoration

### Architecture Details

- Base Model: InceptionV3 (pre-trained on ImageNet)
- Classification Head:
  - Dense layers: 1024 → 512 → 256 → 128 → num_classes
  - LeakyReLU activation
  - Batch Normalization
  - Progressive dropout (0.5 → 0.2)

## Multi-Model Comparison (`flower_9model.py`)

This script provides a comprehensive comparison of 9 state-of-the-art CNN architectures.

### Models Evaluated

1. MobileNetV2
2. InceptionV3
3. VGG16
4. VGG19
5. ResNet50
6. ResNet101
7. DenseNet121
8. Xception
9. InceptionResNetV2

### Analysis Features

- Automated training pipeline for all models
- Comparative performance metrics
- Detailed visualization tools
- Comprehensive logging system
- Statistical analysis of results

## Technical Requirements

```python
tensorflow >= 2.0.0
numpy >= 1.19.2
pandas >= 1.2.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
```

## Project Structure

```plaintext
.
├── flower_images.py         # Single model implementation
├── flower_9model.py        # Multi-model comparison
├── flower_images_split/    # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
└── flower_images_output/   # Output directory
    ├── models/
    ├── plots/
    └── logs/
```

## Usage Instructions

For single model implementation:

```bash
python flower_images.py
```

For multi-model comparison:

```bash
python flower_9model.py
```

## Configuration Parameters

Common settings for both implementations:

- Image Resolution: 224x224x3
- Batch Size: 32
- Random Seed: 42
- Initial Learning Rate: 0.001
- Early Stopping Patience: 5

## Output and Visualizations

Both scripts generate:

- Training and validation curves
- Confusion matrices
- Detailed classification reports
- Saved model checkpoints

The multi-model script additionally provides:

- Comparative model performance plots
- Cross-model analysis
- Detailed logging reports

## Model Artifacts

Models are saved in the `flower_images_output` directory:

- Single Model: `flower_InceptionV3.keras`
- Multi-Model: `flower_[MODEL_NAME].keras`

## Performance Metrics

Both implementations track:

- Training/validation accuracy
- Training/validation loss
- Test set performance
- Per-class metrics
- Confusion matrices

## Author

kaisarfardin6620

## License

This project is licensed under the MIT License. See the LICENSE file for details.
