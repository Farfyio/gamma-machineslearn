import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gamma_machineslearn.config import ProjectConfig, load_config


class TestConfig(unittest.TestCase):
    def test_load_config_reads_yaml(self):
        with TemporaryDirectory() as tmp_dir:
            config_file = Path(tmp_dir) / "config.yaml"
            config_file.write_text(
                "random_seed: 7\ntest_size: 0.3\noutput_dir: results\nmetadata:\n  run: smoke\n",
                encoding="utf-8",
            )

            cfg = load_config(config_file)

        self.assertIsInstance(cfg, ProjectConfig)
        self.assertEqual(cfg.random_seed, 7)
        self.assertEqual(cfg.test_size, 0.3)
        self.assertEqual(cfg.output_dir, "results")
        self.assertEqual(cfg.metadata, {"run": "smoke"})


if __name__ == "__main__":
    unittest.main()
