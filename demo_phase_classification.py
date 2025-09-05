"""
Demo script for PhaseNet phase classification.

This script demonstrates how to use PhaseNet for phase classification
from 4D-STEM diffraction patterns.
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
from lightning import LightningModule

from src.data.phase_stem_datamodule import PhaseSTEMDataModule
from src.models.phase_classification_module import PhaseClassificationModule

log = logging.getLogger(__name__)


def demo_phase_classification(cfg: DictConfig) -> None:
    """Run phase classification demo."""
    
    log.info("🚀 Starting PhaseNet Phase Classification Demo")
    
    # Initialize the datamodule
    log.info("📊 Initializing datamodule...")
    datamodule: PhaseSTEMDataModule = hydra.utils.instantiate(cfg.data)
    
    # Setup the datamodule
    datamodule.setup("fit")
    
    log.info(f"✅ Datamodule ready!")
    log.info(f"   - Number of classes: {datamodule.num_classes}")
    log.info(f"   - Training samples: {len(datamodule.data_train)}")
    log.info(f"   - Validation samples: {len(datamodule.data_val)}")
    
    # Initialize the model
    log.info("🧠 Initializing model...")
    model: PhaseClassificationModule = hydra.utils.instantiate(cfg.model)
    
    log.info(f"✅ Model ready!")
    log.info(f"   - Network type: {model.network_type}")
    log.info(f"   - Number of classes: {model.num_classes}")
    log.info(f"   - Pretrained: {model.pretrained}")
    
    # Test data loading
    log.info("🔄 Testing data loading...")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    log.info(f"✅ Data loading successful!")
    log.info(f"   - Batch size: {batch['pattern'].shape[0]}")
    log.info(f"   - Pattern shape: {batch['pattern'].shape}")
    log.info(f"   - Phase IDs shape: {batch['phase_id'].shape}")
    if 'reliability' in batch:
        log.info(f"   - Reliability shape: {batch['reliability'].shape}")
    
    # Test model forward pass
    log.info("⚡ Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        predictions = model(batch['pattern'])
    
    log.info(f"✅ Forward pass successful!")
    log.info(f"   - Predictions shape: {predictions.shape}")
    log.info(f"   - Predicted classes: {torch.argmax(predictions, dim=1)}")
    log.info(f"   - True classes: {batch['phase_id']}")
    
    # Compute loss
    log.info("📏 Testing loss computation...")
    loss, preds, targets = model.model_step(batch)
    
    log.info(f"✅ Loss computation successful!")
    log.info(f"   - Loss value: {loss.item():.4f}")
    log.info(f"   - Accuracy: {(preds == targets).float().mean().item():.4f}")
    
    log.info("🎉 Demo completed successfully!")
    log.info("You can now train the model with:")
    log.info("   python src/train.py -cn train_phase_classification")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_phase_classification.yaml")
def main(cfg: DictConfig) -> None:
    """Main demo function."""
    demo_phase_classification(cfg)


if __name__ == "__main__":
    main()
