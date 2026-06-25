import torch.nn as nn

from radiance_field_project.config.schema import ModelConfig, ModelType
from radiance_field_project.models.gaussian_head import SplatRefiner
from radiance_field_project.models.minipointnet import (
    MiniPointNet3DGSNetwork,
    MiniPointNet2DGSNetwork,
    MiniPointNet2DGSNetworkPP,
)


def build_model(model_config: ModelConfig,
                encoder_in_channel: int | None = None,
                splat_dim: int | None = None,
                encoder_in_dim: int | None = None) -> nn.Module:
    """Build the per-splat refiner the config asks for.

    model_type=pointnet2  → SplatRefiner (PointNet++ encoder, configurable widths)
    model_type=minipointnet → MiniPointNet{3DGS,2DGS}Network (simple per-point MLP)

    For minipointnet the encoder/decoder/init_scale/init_opacity fields are ignored.

    encoder_in_channel overrides SA1 input width for SplatRefiner (for loading
    old checkpoints trained with xyz-only encoder input).
    splat_dim overrides the output feature width for MiniPointNet2DGSNetwork.
    encoder_in_dim overrides the encoder input width for MiniPointNet2DGSNetwork
    independently of splat_dim (old checkpoints with xyz-only encoder).
    """
    if model_config.model_type == ModelType.pointnet2:
        return SplatRefiner(
            output_format=model_config.output_type.value,
            encoder=list(model_config.encoder),
            decoder=list(model_config.decoder),
            init_scale=model_config.init_scale,
            init_opacity=model_config.init_opacity,
            encoder_in_channel=encoder_in_channel,
        )

    if model_config.model_type == ModelType.minipointnet:
        if model_config.output_type.value == "3dgs":
            return MiniPointNet3DGSNetwork()
        kw = {"hidden_dim": model_config.hidden_dim}
        if splat_dim is not None:
            kw["splat_dim"] = splat_dim
        if encoder_in_dim is not None:
            kw["encoder_in_dim"] = encoder_in_dim
        return MiniPointNet2DGSNetwork(**kw)

    if model_config.model_type == ModelType.minipointnetpp:
        kw = {"hidden_dim": model_config.hidden_dim}
        if splat_dim is not None:
            kw["splat_dim"] = splat_dim
        if encoder_in_dim is not None:
            kw["encoder_in_dim"] = encoder_in_dim
        return MiniPointNet2DGSNetworkPP(**kw)

    raise ValueError(f"Unknown model_type: {model_config.model_type}")
