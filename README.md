# ChestXRay Multi-Label Image Classification using Deep Learning architecture

Deep learning models for multi-label classification of chest X-ray pathologies using the NIH ChestX-ray14 dataset.

## 🎯 Overview

This project implements multiple deep learning architectures for automated detection of 14 different thoracic pathologies in chest X-ray images:
- Atelectasis
- Cardiomegaly  
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia

## 🏗️ Architecture

### Supported Models

1. **SimpleCNN**: Baseline convolutional neural network
2. **DenseNet121**: Pre-trained DenseNet with transfer learning
3. **EfficientNetB0**: EfficientNet architecture with transfer learning
4. **DEAFNet**: Dense-Efficient Attention-Fusion Network (novel fusion architecture)

### DEAFNet Architecture

DEAFNet combines DenseNet121 and EfficientNetB3 backbones with a cross-attention mechanism for enhanced feature learning and superior performance.

## 📁 Project Structure

```
chest-xray-classification/
├── config.py                 # Configuration file
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation and visualization
├── scripts/               # Training and test scripts
├── notebooks/             # Jupyter notebooks for experiments
├── outputs/               # Model weights, logs

```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- Google Colab (recommended) or local GPU
- Google Drive account (for dataset storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chest-xray-classification.git
cd chest-xray-classification

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Dataset Setup

1. Download the NIH ChestX-ray14 dataset
2. Upload to Google Drive in the following structure:
```
Chest XRay/
├── archive/
│   ├── Data_Entry_2017.csv
│   └── images_001/
│       └── images/
│           └── *.png
```

3. Update paths in `config/config.py`:
```python
GDRIVE_ROOT = "/content/drive/MyDrive/Chest XRay"
```

### Training

#### Quick Start (Development Mode)

```bash
# Train with a small subset for testing
python scripts/train.py --model DEAFNet --dev
```

#### Full Training

```bash
# Train DEAFNet model
python scripts/train.py --model DEAFNet

# Train with custom parameters
python scripts/train.py \
    --model DenseNet121 \
    --batch-size 64 \
    --epochs-stage1 15 \
    --epochs-stage2 30
```

#### Available Options

- `--model`: Choose from SimpleCNN, DenseNet121, EfficientNetB0, DEAFNet
- `--dev`: Use development config (subset of data)
- `--gdrive-root`: Custom Google Drive path
- `--batch-size`: Training batch size
- `--epochs-stage1`: Epochs for head training
- `--epochs-stage2`: Epochs for fine-tuning

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --model DEAFNet --weights outputs/models/deafnet_best.weights.h5
```

### Inference

```bash
# Predict on new images
python scripts/predict.py --model DEAFNet --image path/to/xray.png
```

## 📊 Features

### Data Processing
- Multi-label stratified splitting
- Data augmentation (rotation, flip, zoom, brightness)
- Efficient TensorFlow data pipeline
- Patient-level data splitting (no data leakage)

### Training
- Focal Loss for handling class imbalance
- Two-stage transfer learning
- Early stopping and model checkpointing
- Mixed precision training support
- Comprehensive logging

### Evaluation
- Class-wise ROC-AUC scores
- ROC curve visualization
- Monte Carlo Dropout for uncertainty estimation
- Confusion matrices and performance metrics

## 🔬 Model Performance

| Model | Test AUC | Parameters | Training Time |
|-------|----------|------------|---------------|
| SimpleCNN | 0.78 | 2M | 1h |
| DenseNet121 | 0.82 | 8M | 3h |
| EfficientNetB0 | 0.84 | 5M | 2.5h |
| **DEAFNet** | **0.87** | 15M | 5h |

*Tested on subset of NIH ChestX-ray14 dataset*


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NIH Clinical Center for the ChestX-ray14 dataset
- Original papers on DenseNet and EfficientNet
- TensorFlow and Keras communities
