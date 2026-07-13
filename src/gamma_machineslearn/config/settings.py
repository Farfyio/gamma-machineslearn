"""Configuration loading helpers for the project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    """Container for run-time configuration values."""

    random_seed: int = 42
    test_size: float = 0.2
    output_dir: str = "outputs"
    metadata: dict[str, Any] = field(default_factory=dict)



def load_config(path: str | Path) -> ProjectConfig:
    """Load configuration from a YAML file into a ProjectConfig object."""

    config_path = Path(path)
    content = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return ProjectConfig(**content)
