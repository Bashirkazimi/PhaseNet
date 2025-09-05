"""LightningModule for phase (material) classification from 4D-STEM diffraction patterns."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from src.models.components.phase_networks import create_phase_network

log = logging.getLogger(__name__)


class PhaseClassificationModule(LightningModule):
    """Lightning module for phase classification from diffraction patterns."""

    def __init__(
        self,
        num_classes: int = 3,
        network_type: str = "resnet50",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        compile_model: bool = False,
        class_weights: Optional[Dict[int, float]] = None,
        class_counts: Optional[Dict[int, int]] = None,
        class_weighting: str = "none",  # none|inverse|effective
        effective_beta: float = 0.9999,
        class_weight_cap: float = 20.0,
        use_reliability: bool = True,
        reliability_epsilon: float = 0.05,
        reliability_power: float = 1.0,
        logit_adjustment: bool = False,
        tau: float = 0.5,
        network_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        # Model configuration
        self.num_classes = num_classes
        self.network_type = network_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.use_reliability = use_reliability
        self.reliability_epsilon = reliability_epsilon
        self.reliability_power = reliability_power
        self.logit_adjustment = logit_adjustment
        self.tau = tau
        
        # Create network
        network_config = network_config or {}
        self.net = create_phase_network(
            network_type=network_type,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **network_config
        )
        
        # Compile model if requested
        if compile_model:
            self.net = torch.compile(self.net)
        
        # Class weighting
        self.class_weights = self._compute_class_weights(
            class_weights, class_counts, class_weighting, effective_beta, class_weight_cap
        )
        
        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # Classification metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        
        # Best validation metrics
        self.val_acc_best = MeanMetric()
        self.val_f1_best = MeanMetric()

    def _compute_class_weights(
        self,
        class_weights: Optional[Dict[int, float]],
        class_counts: Optional[Dict[int, int]],
        class_weighting: str,
        effective_beta: float,
        class_weight_cap: float,
    ) -> Optional[torch.Tensor]:
        """Compute class weights for handling class imbalance."""
        if class_weights is not None:
            # Manual class weights provided
            weights = torch.ones(self.num_classes)
            for class_id, weight in class_weights.items():
                if 0 <= class_id < self.num_classes:
                    weights[class_id] = weight
            return weights
        
        if class_weighting == "none" or class_counts is None:
            return None
        
        # Convert class counts to tensor
        counts = torch.zeros(self.num_classes)
        for class_id, count in class_counts.items():
            if 0 <= class_id < self.num_classes:
                counts[class_id] = count
        
        if class_weighting == "inverse":
            # Inverse frequency weighting
            weights = 1.0 / (counts + 1e-8)
        elif class_weighting == "effective":
            # Effective number weighting
            effective_nums = 1.0 - torch.pow(effective_beta, counts)
            weights = (1.0 - effective_beta) / effective_nums
        else:
            raise ValueError(f"Unknown class_weighting: {class_weighting}")
        
        # Cap weights and normalize
        weights = torch.clamp(weights, max=class_weight_cap)
        weights = weights / weights.mean()  # Normalize
        
        log.info(f"Computed class weights: {weights.tolist()}")
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step."""
        patterns = batch["pattern"]
        targets = batch["phase_id"]
        
        # Forward pass
        logits = self.forward(patterns)
        
        # Apply logit adjustment if enabled
        if self.logit_adjustment and self.class_weights is not None:
            logits = logits - self.tau * torch.log(self.class_weights[None, :].to(logits.device))
        
        # Compute loss
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, targets, weight=self.class_weights.to(logits.device))
        else:
            loss = F.cross_entropy(logits, targets)
        
        # Apply reliability weighting if available
        if self.use_reliability and "reliability" in batch:
            reliability = batch["reliability"]
            # Transform reliability: r' = max(epsilon, r^power)
            reliability_weight = torch.clamp(
                torch.pow(reliability, self.reliability_power),
                min=self.reliability_epsilon
            )
            loss = loss * reliability_weight.mean()
        
        predictions = torch.argmax(logits, dim=1)
        
        return loss, predictions, targets

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        
        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        
        return loss

    def on_validation_epoch_end(self) -> None:
        """Validation epoch end."""
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        
        self.val_acc_best(val_acc)
        self.val_f1_best(val_f1)
        
        # Log metrics
        self.log("val/loss", self.val_loss, prog_bar=True)
        self.log("val/acc", val_acc, prog_bar=True)
        self.log("val/f1", val_f1, prog_bar=False)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=False)

    def on_test_epoch_end(self) -> None:
        """Test epoch end."""
        self.log("test/loss", self.test_loss.compute())
        self.log("test/acc", self.test_acc.compute())
        self.log("test/f1", self.test_f1.compute())

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        if self.scheduler_type == "none":
            return {"optimizer": optimizer}
        
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=self.learning_rate * 0.01,
            )
            
        elif self.scheduler_type == "cosine_warmup":
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
            
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - self.warmup_epochs,
                eta_min=self.learning_rate * 0.01,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs],
            )
            
        elif self.scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=10,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/acc",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
