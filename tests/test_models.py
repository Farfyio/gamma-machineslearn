import unittest

from gamma_machineslearn.models import ModelFactory


class ModelFactoryTests(unittest.TestCase):
    def test_register_and_create_custom_model(self):
        factory = ModelFactory()
        factory.register("custom", lambda alpha=1: {"alpha": alpha})

        model = factory.create("custom", alpha=3)

        self.assertEqual(model, {"alpha": 3})

    def test_create_unknown_model_raises_helpful_error(self):
        factory = ModelFactory()

        with self.assertRaises(ValueError) as ctx:
            factory.create("does_not_exist")

        self.assertIn("Unknown model", str(ctx.exception))
        self.assertIn("one_class_svm", str(ctx.exception))
        self.assertIn("isolation_forest", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
