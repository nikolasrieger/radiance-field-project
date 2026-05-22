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
    num_sequences: int = 1
    num_supervision_views : int = 10
    image_height: int = 800
    image_width: int = 800
    subset_lists_filename: str = "set_lists_all-categories.sqlite"

@dataclass
class ModelConfig:
    output_type:  OutputType = OutputType.gs3d
    in_dim: int = 14
    hidden_dims: List[int] = field(default_factory=lambda : [64, 32, 64])

@dataclass
class LossConfig:
    type: LossType = LossType.l2_masked

@dataclass
class TrainConfig:
    epochs: int = 1000
    lr: float = 1e-4
    optimizer: Optimizer = Optimizer.adamw
    weight_decay: float = 0.0
    loss: LossConfig = field(default_factory=LossConfig)
    log_every: int = 10
    ckpt_every: int = 100

@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    