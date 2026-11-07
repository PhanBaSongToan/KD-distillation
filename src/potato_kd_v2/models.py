"""Paper-faithful models plus the distillation adapters used by the improved trainer.

The deployment networks (``HybridStudent`` backbone and ``TeacherResNet18``) are
unchanged from Hoang et al. 2025: four stages of 4/6/8/10 hybrid dense inverted
residual blocks, growth rate 16, expansion ratio 3, widths 128/224/352/512.
Only *training-time* adapters (feature projectors) are added on top.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, groups: int = 1, activate: bool = True):
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class HybridDenseInvertedBlock(nn.Module):
    """DenseNet-style concatenation around a MobileNetV2-style block."""

    def __init__(self, in_channels: int, growth_rate: int = 16,
                 expansion_ratio: int = 3):
        super().__init__()
        bottleneck_channels = 4 * growth_rate
        expanded_channels = expansion_ratio * bottleneck_channels

        self.bottleneck = ConvBNAct(in_channels, bottleneck_channels, 1)
        self.main = nn.Sequential(
            ConvBNAct(bottleneck_channels, expanded_channels, 1),
            ConvBNAct(expanded_channels, expanded_channels, 3, groups=expanded_channels),
            ConvBNAct(expanded_channels, growth_rate, 1, activate=False),
        )
        self.shortcut = ConvBNAct(bottleneck_channels, growth_rate, 1, activate=False)

    def forward(self, x: Tensor) -> Tensor:
        x_bot = self.bottleneck(x)
        residual = self.main(x_bot) + self.shortcut(x_bot)
        return torch.cat([x, residual], dim=1)


def make_stage(in_channels: int, depth: int, growth_rate: int = 16) -> nn.Sequential:
    blocks, channels = [], in_channels
    for _ in range(depth):
        blocks.append(HybridDenseInvertedBlock(channels, growth_rate))
        channels += growth_rate
    return nn.Sequential(*blocks)


class HybridStudent(nn.Module):
    """Student described in Hoang et al. with main and auxiliary classifiers.

    ``growth_rate`` is exposed so capacity can be scaled as a controlled
    ablation (16 -> the paper's 1.05 M model; 24 -> ~1.6 M). Channel widths are
    derived exactly as in the paper: each stage adds ``depth * growth_rate``
    channels to its input width, so g=16 reproduces 128/224/352/512.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.0,
                 growth_rate: int = 16, depths=(4, 6, 8, 10)):
        super().__init__()
        g = int(growth_rate)
        w0 = 64
        w1 = w0 + depths[0] * g
        w2 = w1 + depths[1] * g
        w3 = w2 + depths[2] * g
        w4 = w3 + depths[3] * g
        self.widths = (w0, w1, w2, w3, w4)
        self.growth_rate = g

        self.stem = nn.Sequential(
            ConvBNAct(3, w0, 7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = make_stage(w0, depths[0], g)
        self.down1 = nn.AvgPool2d(2, 2)
        self.stage2 = make_stage(w1, depths[1], g)
        self.down2 = nn.AvgPool2d(2, 2)
        self.stage3 = make_stage(w2, depths[2], g)
        self.down3 = nn.AvgPool2d(2, 2)
        self.stage4 = make_stage(w3, depths[3], g)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.aux_head = nn.Linear(w3, num_classes)
        self.main_head = nn.Linear(w4, num_classes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        stage3_feature = x                  # [B, 352, 14, 14]
        x = self.down3(x)
        aux_feature = x                     # [B, 352, 7, 7]
        x = self.stage4(x)                  # [B, 512, 7, 7]

        aux_vec = self.pool(aux_feature).flatten(1)
        main_vec = self.pool(x).flatten(1)
        return {
            "stage3_feature": stage3_feature,
            "aux_feature": aux_feature,
            "main_feature": x,
            "aux_vector": aux_vec,
            "main_vector": main_vec,
            "aux_logits": self.aux_head(aux_vec),
            "main_logits": self.main_head(self.dropout(main_vec)),
        }


class TeacherResNet18(nn.Module):
    """ResNet18 teacher with the paper's auxiliary (stage-3) branch."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3   # [B, 256, 14, 14]
        self.layer4 = backbone.layer4   # [B, 512, 7, 7]

        self.aux_align = nn.Sequential(
            ConvBNAct(256, 256, 3, stride=2, groups=256),
            ConvBNAct(256, 352, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.aux_head = nn.Linear(352, num_classes)
        self.main_head = nn.Linear(512, num_classes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        layer3 = self.layer3(x)
        main_feature = self.layer4(layer3)
        aux_feature = self.aux_align(layer3)

        aux_vec = self.pool(aux_feature).flatten(1)
        main_vec = self.pool(main_feature).flatten(1)
        return {
            "layer3_feature": layer3,       # [B, 256, 14, 14]
            "layer4_feature": main_feature,  # [B, 512, 7, 7]
            "aux_feature": aux_feature,
            "main_feature": main_feature,
            "aux_vector": aux_vec,
            "main_vector": main_vec,
            "aux_logits": self.aux_head(aux_vec),
            "main_logits": self.main_head(self.dropout(main_vec)),
        }


class DistillStudent(nn.Module):
    """Training wrapper: student backbone + 1x1 adapters for feature KD.

    The adapters are *not* part of the deployed student. They let the student
    match the teacher at two spatial levels that share a resolution:

    * stage-3 : student 352x14x14  <-  teacher layer3 256x14x14  (1x1, 256->352)
    * stage-4 : student 512x7x7    <-  teacher layer4 512x7x7    (direct)

    The baseline only matched global-average-pooled vectors, which throws away
    the spatial structure the paper's own Grad-CAM analysis says matters.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.0,
                 growth_rate: int = 16, depths=(4, 6, 8, 10)):
        super().__init__()
        self.backbone = HybridStudent(num_classes, dropout=dropout,
                                      growth_rate=growth_rate, depths=depths)
        w3 = self.backbone.widths[3]
        w4 = self.backbone.widths[4]
        # Maps the teacher's 256-channel layer-3 output into the student's
        # stage-3 space (both are 14x14 at 224 input).
        self.proj3 = nn.Sequential(
            nn.Conv2d(256, w3, 1, bias=False), nn.BatchNorm2d(w3),
        )
        # stage-4: the paper's student has 512 channels, identical to the
        # teacher's layer-4, so no adapter is needed at growth 16. Wider
        # variants (growth 24 -> 736) need a 1x1 projection.
        self.proj4 = (nn.Identity() if w4 == 512
                      else nn.Conv2d(512, w4, 1, bias=False))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.backbone(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
