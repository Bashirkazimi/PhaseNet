# PhaseNet: Phase Classification Setup

## Overview

PhaseNet has been successfully modified from the Lightning-Hydra-Template to support phase classification from 4D-STEM diffraction patterns, similar to the OriNet project.

## Key Components Added

### 1. Data Processing
- **`src/data/phase_stem_dataset.py`**: Dataset class for loading diffraction patterns and phase labels
- **`src/data/phase_stem_datamodule.py`**: Lightning DataModule with train/val/test splits
- **`src/data/components/fourd_stem_utils.py`**: Utilities for loading and processing diffraction patterns

### 2. Model Architecture
- **`src/models/components/phase_networks.py`**: Neural network architectures including:
  - Custom CNN encoder for diffraction patterns
  - Pretrained model wrapper (ResNet, EfficientNet, etc.)
  - Complete phase classifier with classification head
- **`src/models/phase_classification_module.py`**: Lightning module with:
  - Class imbalance handling (inverse/effective weighting)
  - Reliability-based sample weighting
  - Comprehensive metrics (accuracy, F1-score)
  - Configurable optimizers and schedulers

### 3. Configuration Files
- **`configs/data/phase_stem.yaml`**: Data loading configuration
- **`configs/model/phase_classification.yaml`**: Model architecture and training settings
- **`configs/train_phase_classification.yaml`**: Main training configuration
- **`configs/experiment/phase_classification_example.yaml`**: Example experiment setup

### 4. Utilities
- **`scripts/convert_phase_map.py`**: Convert RGB phase maps to integer label maps
- **`demo_phase_classification.py`**: Demo script for testing the setup
- **`test_setup.py`**: Setup verification script

## Data Structure Expected

```
data/phase_stem/
├── DiffractionPatterns/
│   ├── Image-00001.tif
│   ├── Image-00002.tif
│   └── ...
├── phase_id_map.npy          # Integer phase labels (H, W)
└── phase_reliability_map.tif  # Optional reliability weights
```

## Key Features

### Class Imbalance Handling
- **Inverse frequency weighting**: `1/count` for each class
- **Effective number weighting**: Advanced rebalancing using effective sample numbers
- **Manual class weights**: Custom weights for specific classes

### Reliability Weighting
- Optional per-pixel reliability scores
- Configurable reliability transformation (power scaling, epsilon clamping)
- Integrates seamlessly with loss computation

### Pretrained Model Support
- Integration with `timm` library for 20+ pretrained models
- Automatic adaptation for single-channel diffraction inputs
- Flexible backbone freezing/unfreezing

### Advanced Training Features
- **Learning rate scheduling**: Cosine annealing with warmup
- **Model compilation**: PyTorch 2.0+ acceleration
- **Comprehensive logging**: TensorBoard, Weights & Biases support
- **Automatic checkpointing**: Best model selection based on validation accuracy

## Usage Examples

### Basic Training
```bash
python src/train.py -cn train_phase_classification
```

### Custom Configuration
```bash
python src/train.py -cn train_phase_classification \
  data.data_root=data/your_4dstem_data/ \
  model.num_classes=4 \
  model.network_type=efficientnet_b0 \
  trainer.max_epochs=100
```

### Using Experiments
```bash
python src/train.py experiment=phase_classification_example
```

### Class Imbalance Handling
```bash
python src/train.py -cn train_phase_classification \
  model.class_weighting=effective \
  model.effective_beta=0.9999
```

## Integration with Original Template

The phase classification components are fully integrated with the original Lightning-Hydra-Template:

- ✅ **Preserves MNIST example**: Original functionality unchanged
- ✅ **Hydra configuration**: Full config composition and override support  
- ✅ **Lightning training**: Automatic logging, checkpointing, multi-GPU support
- ✅ **Experiment tracking**: TensorBoard, W&B, MLflow compatibility
- ✅ **Testing framework**: Pytest integration for validation
- ✅ **Code quality**: Pre-commit hooks, formatting, linting

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Prepare your data** in the expected format
3. **Run setup test**: `python test_setup.py`
4. **Start training**: `python src/train.py -cn train_phase_classification`

The PhaseNet framework is now ready for phase classification tasks with the same professional software engineering practices as the original OriNet project!
