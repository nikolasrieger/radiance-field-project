from dataclasses import dataclass, field
from enum import Enum
from typing import List


class WandbMode(Enum):
    online = "online"
    offline = "offline"
    disabled = "disabled"


class OutputType(Enum):
    gs3d = "3dgs"
    gs2d = "2dgs"


class Activation(Enum):
    relu = "relu"
    gelu = "gelu"
    silu = "silu"


class Optimizer(Enum):
    adamw = "adamw"
    adam = "adam"


class LossType(Enum):
    l2_masked = "l2_masked"
    l1_masked = "l1_masked"


@dataclass
class RunConfig:
    name: str = "unnamed"
    seed: int = 42
    output_dir: str = "runs/${run.name}"
    device: str = "cuda"
    wandb_project: str = "radiance-field-conversion"
    wandb_mode: WandbMode = WandbMode.online


@dataclass
class DataConfig:
    num_supervision_views: int = 10
    image_height: int = 800
    image_width: int = 800
    subset_lists_filename: str = "set_lists_all-categories.sqlite"
    batch_size: int = 1
    num_workers: int = 0
    load_masks: bool = True
    load_depths: bool = False
    load_gaussian_splats: bool = True
    # Empty list means "no pinning"; populate with sequence names to overfit a subset.
    pick_sequences: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    output_type: OutputType = OutputType.gs3d
    # Output channel width of each PointNet++ SA stage. Length must be 3
    # (the final stage is group_all=True and produces the global feature).
    encoder: List[int] = field(default_factory=lambda: [128, 256, 1024])
    # Linear widths of the per-point trunk MLP. The last value is the head
    # input dim. Variable length is supported.
    decoder: List[int] = field(default_factory=lambda: [512, 256, 128])
    init_scale: float = 0.01
    init_opacity: float = 0.1


@dataclass
class LossConfig:
    type: LossType = LossType.l2_masked
    lambda_ssim: float = 0.2
    # Weight on the rendered-vs-depth-derived normal L1 loss. Set to 0 to
    # disable normal supervision. Requires output_type=gs2d (only the 2dgs
    # rasterizer produces surface normals).
    lambda_normal: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 1000
    lr: float = 1e-4
    optimizer: Optimizer = Optimizer.adamw
    weight_decay: float = 0.0
    loss: LossConfig = field(default_factory=LossConfig)
    log_every: int = 10
    ckpt_every: int = 100
    grad_clip: float = 1.0


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
