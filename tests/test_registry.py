from __future__ import annotations

import pytest

from readback.models.registry import ModelSpec, build, load_specs


def test_load_specs_parses_kind_ref_and_options(tmp_path):
    config = tmp_path / "models.toml"
    config.write_text(
        '[models.parakeet-v2]\nkind = "parakeet"\nref = "nvidia/x"\nbatch_size = 256\n'
        '[models.rasr-v1]\nkind = "parakeet"\nref = "/ckpt.nemo"\nfrom_path = true\n'
    )
    specs = load_specs(config)
    assert specs["parakeet-v2"].kind == "parakeet"
    assert specs["parakeet-v2"].ref == "nvidia/x"
    assert specs["parakeet-v2"].options == {"batch_size": 256}
    assert specs["rasr-v1"].options == {"from_path": True}


def test_load_specs_empty_when_no_models(tmp_path):
    config = tmp_path / "models.toml"
    config.write_text("")
    assert load_specs(config) == {}


def test_build_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown model kind"):
        build(ModelSpec(name="x", kind="mystery", ref="ref"))
