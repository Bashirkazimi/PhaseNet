"""Neural network architectures for phase (material) classification from diffraction patterns."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class DiffractionCNNEncoder(nn.Module):
    """Custom CNN encoder for diffraction patterns."""

    def __init__(
        self,
        input_channels: int = 1,
        num_features: int = 512,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_features),
        )
        
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class PretrainedEncoder(nn.Module):
    """Pretrained model encoder using timm."""

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        num_features: int = 512,
        freeze_backbone: bool = False,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for pretrained models. Install with: pip install timm")
        
        # Create backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            in_chans=input_channels,
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 224, 224)
            backbone_features = self.backbone(dummy_input).shape[1]
        
        # Optional feature adapter
        if backbone_features != num_features:
            self.feature_adapter = nn.Linear(backbone_features, num_features)
        else:
            self.feature_adapter = nn.Identity()
        
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.feature_adapter(features)


class PhaseClassifier(nn.Module):
    """Complete phase classification network with encoder and classifier head."""

    def __init__(
        self,
        encoder_type: str = "pretrained",
        num_classes: int = 3,
        encoder_config: Optional[dict] = None,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        
        encoder_config = encoder_config or {}
        
        # Create encoder
        if encoder_type == "custom_cnn":
            self.encoder = DiffractionCNNEncoder(**encoder_config)
        elif encoder_type == "pretrained":
            self.encoder = PretrainedEncoder(**encoder_config)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C, H, W) -> (B, num_classes)"""
        features = self.encoder(x)
        return self.classifier(features)


def create_phase_network(
    network_type: str = "resnet50",
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs,
) -> PhaseClassifier:
    """Factory function for creating phase classification networks."""
    
    if network_type == "custom_cnn":
        return PhaseClassifier(
            encoder_type="custom_cnn",
            num_classes=num_classes,
            encoder_config={
                "input_channels": kwargs.get("input_channels", 1),
                "num_features": kwargs.get("num_features", 512),
                "dropout_rate": kwargs.get("dropout_rate", 0.2),
            },
            hidden_dim=kwargs.get("hidden_dim", 256),
            dropout_rate=kwargs.get("head_dropout", 0.1),
        )
    else:
        # Pretrained model
        return PhaseClassifier(
            encoder_type="pretrained",
            num_classes=num_classes,
            encoder_config={
                "model_name": network_type,
                "pretrained": pretrained,
                "num_features": kwargs.get("num_features", 512),
                "freeze_backbone": freeze_backbone,
                "input_channels": kwargs.get("input_channels", 1),
            },
            hidden_dim=kwargs.get("hidden_dim", 256),
            dropout_rate=kwargs.get("head_dropout", 0.1),
        )
