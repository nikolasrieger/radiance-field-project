import torch.nn as nn

from radiance_field_project.config.schema import ModelConfig, ModelType
from radiance_field_project.models.gaussian_head import SplatRefiner
from radiance_field_project.models.minipointnet import (
    MiniPointNet3DGSNetwork,
    MiniPointNet2DGSNetwork,
)


def build_model(model_config: ModelConfig) -> nn.Module:
    """Build the per-splat refiner the config asks for.

    model_type=pointnet2  → SplatRefiner (PointNet++ encoder, configurable widths)
    model_type=minipointnet → MiniPointNet{3DGS,2DGS}Network (simple per-point MLP)

    For minipointnet the encoder/decoder/init_scale/init_opacity fields are ignored.
    """
    if model_config.model_type == ModelType.pointnet2:
        return SplatRefiner(
            output_format=model_config.output_type.value,
            encoder=list(model_config.encoder),
            decoder=list(model_config.decoder),
            init_scale=model_config.init_scale,
            init_opacity=model_config.init_opacity,
        )

    if model_config.model_type == ModelType.minipointnet:
        if model_config.output_type.value == "3dgs":
            return MiniPointNet3DGSNetwork()
        return MiniPointNet2DGSNetwork()

    raise ValueError(f"Unknown model_type: {model_config.model_type}")
