from pathlib import Path

from gamma_machineslearn.config import ProjectConfig, load_config



def test_load_config_reads_yaml(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "random_seed: 7\ntest_size: 0.3\noutput_dir: results\nmetadata:\n  run: smoke\n",
        encoding="utf-8",
    )

    cfg = load_config(config_file)

    assert isinstance(cfg, ProjectConfig)
    assert cfg.random_seed == 7
    assert cfg.test_size == 0.3
    assert cfg.output_dir == "results"
    assert cfg.metadata == {"run": "smoke"}
