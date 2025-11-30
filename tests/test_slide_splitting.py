
import unittest
import logging
import sys
import os
sys.path.append(os.getcwd())
import importlib
create_dataset = importlib.import_module("src.main_scripts.01_create_dataset")
split_slides = create_dataset.split_slides

class TestSlideSplitting(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.INFO)
        
    def test_random_split(self):
        # 100 slides, 70/15/15 split
        all_slides = [f"slide_{i}" for i in range(100)]
        config = {
            'dataset_creation': {
                'validation_slides': [],
                'test_slides': [],
                'train_val_test_split': [0.7, 0.15, 0.15]
            }
        }
        
        assignments = split_slides(all_slides, config, self.logger)
        
        train_count = sum(1 for v in assignments.values() if v == 'train')
        val_count = sum(1 for v in assignments.values() if v == 'val')
        test_count = sum(1 for v in assignments.values() if v == 'test')
        
        self.assertEqual(train_count, 70)
        self.assertEqual(val_count, 15)
        self.assertEqual(test_count, 15)
        
    def test_explicit_split(self):
        all_slides = ["slide_A", "slide_B", "slide_C", "slide_D", "slide_E"]
        config = {
            'dataset_creation': {
                'validation_slides': ["slide_B"],
                'test_slides': ["slide_C"],
                'train_val_test_split': [0.7, 0.15, 0.15] # Should be ignored
            }
        }
        
        assignments = split_slides(all_slides, config, self.logger)
        
        self.assertEqual(assignments["slide_B"], 'val')
        self.assertEqual(assignments["slide_C"], 'test')
        self.assertEqual(assignments["slide_A"], 'train')
        self.assertEqual(assignments["slide_D"], 'train')
        self.assertEqual(assignments["slide_E"], 'train')

if __name__ == '__main__':
    unittest.main()
