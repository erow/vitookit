import unittest
from vitookit.datasets.build_dataset import build_dataset

class TestBuildDataset(unittest.TestCase):
    def test_build_dataset(self):
        args = Args()  # Replace Args() with the actual arguments you want to pass
        is_train = True  # Replace True with the actual value you want to pass
        dataset, nb_classes = build_dataset(args, is_train)
        
        # Add your assertions here to validate the dataset and nb_classes
        
        # Example assertion:
        self.assertIsNotNone(dataset)
        self.assertEqual(nb_classes, 37)

if __name__ == '__main__':
    unittest.main()