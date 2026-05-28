import argparse
from pathlib import Path
from omegaconf import OmegaConf,DictConfig
from radiance_field_project.config.schema import ExperimentConfig

def load_config(yaml_path: str | Path, overrides: list[str] | None = None):
    base = OmegaConf.structured(ExperimentConfig())
    file_cfg = OmegaConf.load(yaml_path)
    cfg = OmegaConf.merge(base, file_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg

def parse_cli() -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("overrides", nargs="*",
                        help="key=value pairs, e.g. train.lr=1e-4")
    args = parser.parse_args()
    return load_config(args.config, args.overrides)