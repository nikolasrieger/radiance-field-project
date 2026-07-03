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
    linprim = "linprim"


class Activation(Enum):
    relu = "relu"
    gelu = "gelu"
    silu = "silu"


class Optimizer(Enum):
    adamw = "adamw"
    adam = "adam"


class SchedulerType(Enum):
    none = "none"
    plateau = "plateau"


class LossType(Enum):
    l2_masked = "l2_masked"
    l1_masked = "l1_masked"
    l1 = "l1"


class NormalLossType(Enum):
    l1 = "l1"
    cosine = "cosine"


class NormalRegion(Enum):
    foreground = "foreground"  # depth-valid pixels inside the fg mask
    full = "full"              # all depth-valid pixels


class ModelType(Enum):
    pointnet2 = "pointnet2"
    minipointnet = "minipointnet"
    minipointnetpp = "minipointnetpp"


class TokenSampling(Enum):
    fps = "fps"
    random = "random"


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
    # Root directory of the uco3d dataset (must contain metadata.sqlite).
    # Overrides the UCO3D_DATASET_ROOT environment variable when set.
    dataset_root: str = ""
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
    # Skip sequences whose Gaussian splat count exceeds this. 0 = no limit.
    max_splats: int = 0
    # Subsample Gaussians to at most this many per forward pass. 0 = no limit.
    max_points: int = 0
    # Cap the number of sequences loaded into the Gaussian cache. 0 = all.
    num_sequences: int = 0
    # Number of held-out sequences (from the "val" split) to evaluate the
    # validation loss on after each epoch. 0 = validation disabled.
    num_val_sequences: int = 0
    # Number of frames per validation sequence. 0 = same as num_supervision_views.
    num_val_views: int = 0


@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.pointnet2
    output_type: OutputType = OutputType.gs3d
    # Output channel width of each PointNet++ SA stage. Length must be 3
    # (the final stage is group_all=True and produces the global feature).
    # Only used when model_type=pointnet2.
    encoder: List[int] = field(default_factory=lambda: [128, 256, 1024])
    # Linear widths of the per-point decoder MLP. The last value is the head
    # input dim. Variable length is supported. Only used when model_type=pointnet2.
    #   decoder_type="trunk": legacy per-point MLP trunk over (global feat ⊕ params)
    #   decoder_type="fp":    PointNet++ Feature Propagation upsampling decoder;
    #                         `decoder` supplies the FP-stage MLP widths.
    decoder: List[int] = field(default_factory=lambda: [512, 256, 128])
    # Decoder variant. "trunk" (default, legacy global-broadcast) or "fp"
    # (DA3-style multi-scale upsampling decoder).
    decoder_type: str = "trunk"
    # Split the FP-decoded features into a geometry branch (means/scale/quat)
    # and an appearance branch (sh0/opacity). Only meaningful with decoder_type="fp".
    dual_branch: bool = False
    # Number of encoder tokens subsampled from the input cloud before the SA
    # stack. The SA stage sizes scale with it (npoint = tokens/2, tokens/8).
    # Only used when model_type=pointnet2.
    num_tokens: int = 1024
    # How encoder tokens are picked: "fps" (farthest point sampling, slower but
    # uniform coverage) or "random" (uniform random, cheap at large budgets).
    token_sampling: TokenSampling = TokenSampling.fps
    # Concatenate the global SA feature onto every per-point FP feature and
    # fuse back to the head width before the prediction head, combining the
    # minipointnet-style global conditioning with the FP local features.
    # Requires decoder_type="fp".
    global_skip: bool = False
    init_scale: float = 0.01
    init_opacity: float = 0.1
    # Hidden dimension for minipointnet/minipointnetpp. Scales all layer widths
    # as fractions of this value (h/16 → h/4 → h/2 → h for encoder,
    # h/2 → h/4 → h/8 for decoder). Default 1024 = original architecture.
    hidden_dim: int = 1024


@dataclass
class LossConfig:
    type: LossType = LossType.l2_masked
    lambda_ssim: float = 0.2
    # Weight on the normal supervision loss. Set to 0 to disable. Requires
    # output_type=gs2d (only the 2dgs rasterizer produces surface normals).
    lambda_normal: float = 0.0
    # Weight on the depth supervision loss (L1 against GT depth map). Set to 0
    # to disable. Works with both output types.
    lambda_depth: float = 0.0
    # Which loss to use for normal supervision: "l1" or "cosine" (1 - cos sim).
    normal_loss_type: NormalLossType = NormalLossType.l1
    # Region over which the normal loss is computed: "foreground" (fg mask ∩
    # depth valid) or "full" (all depth-valid pixels).
    normal_region: NormalRegion = NormalRegion.foreground
    # Region over which the photometric loss is computed: "foreground" (fg
    # mask only) or "full" (whole image).
    recon_region: NormalRegion = NormalRegion.foreground


@dataclass
class SchedulerConfig:
    # "none" (constant lr) or "plateau" (ReduceLROnPlateau, keyed on absolute
    # epoch counts rather than a fraction of train.epochs — safe to pair with
    # a very high epochs value and check convergence manually now and then).
    type: SchedulerType = SchedulerType.none
    # Decay factor applied to lr on each plateau event: lr *= factor.
    factor: float = 0.5
    # Number of epochs with no improvement (beyond `threshold`) before decaying.
    patience: int = 50
    # Minimum relative improvement in train loss to reset patience.
    threshold: float = 1e-4
    # Floor below which lr is never decayed further.
    min_lr: float = 1e-6


@dataclass
class TrainConfig:
    epochs: int = 1000
    lr: float = 1e-4
    optimizer: Optimizer = Optimizer.adamw
    weight_decay: float = 0.0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    log_every: int = 10
    # Cap the number of sequences logged as images to wandb per log step.
    max_log_sequences: int = 5
    # Run a validation pass every N epochs (requires data.num_val_sequences > 0).
    val_every: int = 1
    ckpt_every: int = 100
    grad_clip: float = 1.0
    # Number of steps to profile with torch.profiler (0 = disabled).
    # Skips 1 step + 1 warmup, then records this many steps.
    profile_steps: int = 0
    # Randomly sample this many views per scene per step from the frame cache.
    # 0 = use all cached views (original behaviour).
    num_views_per_step: int = 0


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
