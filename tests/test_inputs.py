from __future__ import annotations

import pytest
import yaml

from pcalm.config import load_config
from pcalm.training import adam_learning_rate
from scripts.run_headline_grid import load_depth_state_lr_table, load_state_lr_table


@pytest.mark.parametrize("section", [None, "model", "method", "training"])
def test_unknown_config_keys_are_rejected(tmp_path, section):
    values = {"typo": 1}
    if section is not None:
        values = {section: values}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(TypeError, match="typo"):
        load_config(path)


def test_synthetic_fallback_config_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset: fashion_mnist\nsynthetic_fallback: true\n")
    with pytest.raises(TypeError, match="synthetic_fallback"):
        load_config(path)


@pytest.mark.parametrize("loader", [load_state_lr_table, load_depth_state_lr_table])
def test_missing_state_lr_table_is_rejected(tmp_path, loader):
    with pytest.raises(FileNotFoundError):
        loader(tmp_path / "missing.csv")


@pytest.mark.parametrize("explicit_lr", [None, 5e-4])
def test_unsupported_gamma0_is_rejected(explicit_lr):
    with pytest.raises(ValueError, match="gamma0=1"):
        adam_learning_rate(width=32, depth=32, eta0=1e-3, gamma0=2.0, explicit_lr=explicit_lr)
