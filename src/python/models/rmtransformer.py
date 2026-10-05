import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import List


class ConvRefineBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaxViTStage(nn.Module):
    """A stage that prefers torchvision MaxViT and falls back to a conv stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_grid_size: int,
        n_layers: int,
        head_dim: int,
        partition_size: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
    ):
        from torchvision.models.maxvit import MaxVitBlock
        super().__init__()


        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.stage = MaxVitBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
            norm_layer=norm_layer,
            activation_layer=nn.GELU,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            attention_dropout=attention_dropout,
            partition_size=partition_size,
            input_grid_size=(input_grid_size, input_grid_size),
            n_layers=n_layers,
            p_stochastic=[0.0] * n_layers,
        )
        self.uses_torchvision = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class RMTransformer(nn.Module):
    """RMTransformer: MaxViT encoder + CNN decoder for radio map reconstruction."""

    def __init__(self, config=None, **kwargs):
        super().__init__()

        self.input_channels = int(kwargs.get("input_channels", config.get("model.params.input_channels", 2) if config else 2))
        self.output_channels = int(kwargs.get("output_channels", config.get("model.params.output_channels", 1) if config else 1))

        image_size = int(kwargs.get("image_size", config.get("model.params.image_size", 256) if config else 256))
        stage_layers = kwargs.get("stage_layers", config.get("model.params.stage_layers", [1, 1, 1, 1]) if config else [1, 1, 1, 1])
        head_dim = int(kwargs.get("head_dim", config.get("model.params.head_dim", 32) if config else 32))
        partition_size = int(kwargs.get("partition_size", config.get("model.params.partition_size", 4) if config else 4))

        squeeze_ratio = float(kwargs.get("squeeze_ratio", config.get("model.params.squeeze_ratio", 0.25) if config else 0.25))
        expansion_ratio = float(kwargs.get("expansion_ratio", config.get("model.params.expansion_ratio", 4.0) if config else 4.0))
        mlp_ratio = int(kwargs.get("mlp_ratio", config.get("model.params.mlp_ratio", 4) if config else 4))
        mlp_dropout = float(kwargs.get("mlp_dropout", config.get("model.params.mlp_dropout", 0.0) if config else 0.0))
        attention_dropout = float(kwargs.get("attention_dropout", config.get("model.params.attention_dropout", 0.0) if config else 0.0))

        if len(stage_layers) != 4:
            raise ValueError("model.params.stage_layers must have exactly 4 elements")

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        stage_grid_sizes = [
            image_size // 2,
            image_size // 4,
            image_size // 8,
            image_size // 16,
        ]

        self.enc1 = MaxViTStage(
            in_channels=128,
            out_channels=128,
            input_grid_size=stage_grid_sizes[0],
            n_layers=int(stage_layers[0]),
            head_dim=head_dim,
            partition_size=partition_size,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            attention_dropout=attention_dropout,
        )
        self.enc2 = MaxViTStage(
            in_channels=128,
            out_channels=256,
            input_grid_size=stage_grid_sizes[1],
            n_layers=int(stage_layers[1]),
            head_dim=head_dim,
            partition_size=partition_size,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            attention_dropout=attention_dropout,
        )
        self.enc3 = MaxViTStage(
            in_channels=256,
            out_channels=512,
            input_grid_size=stage_grid_sizes[2],
            n_layers=int(stage_layers[2]),
            head_dim=head_dim,
            partition_size=partition_size,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            attention_dropout=attention_dropout,
        )
        self.enc4 = MaxViTStage(
            in_channels=512,
            out_channels=1024,
            input_grid_size=stage_grid_sizes[3],
            n_layers=int(stage_layers[3]),
            head_dim=head_dim,
            partition_size=partition_size,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            attention_dropout=attention_dropout,
        )

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.fuse4 = ConvRefineBlock(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.fuse3 = ConvRefineBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.fuse2 = ConvRefineBlock(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.fuse1 = ConvRefineBlock(128 + 128, 128)

        self.output_head = nn.Sequential(
            ConvRefineBlock(128 + self.input_channels, 64),
            nn.Conv2d(64, self.output_channels, kernel_size=1),
        )

    @staticmethod
    def _align_for_concat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != skip.shape[2:]:
            return F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_skip = x

        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d3 = self.up4(x4)
        d3 = self._align_for_concat(d3, x3)
        d3 = self.fuse4(torch.cat([d3, x3], dim=1))

        d2 = self.up3(d3)
        d2 = self._align_for_concat(d2, x2)
        d2 = self.fuse3(torch.cat([d2, x2], dim=1))

        d1 = self.up2(d2)
        d1 = self._align_for_concat(d1, x1)
        d1 = self.fuse2(torch.cat([d1, x1], dim=1))

        d0 = self.up1(d1)
        d0 = self._align_for_concat(d0, x0)
        d0 = self.fuse1(torch.cat([d0, x0], dim=1))

        d0 = F.interpolate(d0, size=input_skip.shape[2:], mode="bilinear", align_corners=False)
        out = self.output_head(torch.cat([d0, input_skip], dim=1))
        return torch.sigmoid(out)


Model = RMTransformer
