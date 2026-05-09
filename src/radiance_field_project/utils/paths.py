# src/radiance_field_project/utils/paths.py
"""Project paths."""
import os
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root")


PROJECT_ROOT = _find_project_root()

_env_root = os.environ.get("UCO3D_DATASET_ROOT")
_root_path = Path(_env_root) if _env_root else PROJECT_ROOT / "data" / "uco3d_data_raw"
if not _root_path.is_absolute():
    _root_path = PROJECT_ROOT / _root_path
UCO3D_DATASET_ROOT = _root_path.resolve()


os.environ["UCO3D_DATASET_ROOT"] = str(UCO3D_DATASET_ROOT)

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"