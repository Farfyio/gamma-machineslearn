import unittest

from gamma_machineslearn.preprocessing import create_preprocessing_pipeline


class TestPreprocessing(unittest.TestCase):
    def test_create_preprocessing_pipeline_has_scaler_step(self):
        pipeline = create_preprocessing_pipeline()
        self.assertEqual([name for name, _ in pipeline.steps], ["scaler"])


if __name__ == "__main__":
    unittest.main()
