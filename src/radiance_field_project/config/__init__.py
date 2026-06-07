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