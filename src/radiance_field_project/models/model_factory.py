import torch.nn as nn

from radiance_field_project.config.schema import ModelConfig
from radiance_field_project.models.gaussian_head import SplatRefiner


def build_model(model_config: ModelConfig) -> nn.Module:
    """Build the per-splat refiner the config asks for.

    ``cfg.model.output_type`` selects between the 3dgs and 2dgs heads of
    ``SplatRefiner``; ``encoder`` and ``decoder`` set the PointNet++ SA
    output widths and the trunk MLP widths respectively.
    """
    return SplatRefiner(
        output_format=model_config.output_type.value,
        encoder=list(model_config.encoder),
        decoder=list(model_config.decoder),
        init_scale=model_config.init_scale,
        init_opacity=model_config.init_opacity,
    )
