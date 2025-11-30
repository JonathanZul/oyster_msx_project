import unittest
import shutil
import tempfile
import time
from pathlib import Path
import sys
import importlib

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import modules with numeric prefixes using importlib
create_dataset = importlib.import_module("src.main_scripts.01_create_dataset")
setup_directories = create_dataset.setup_directories

train_yolo = importlib.import_module("src.main_scripts.02_train_yolo")
find_latest_best_model = train_yolo.find_latest_best_model

# Mock logger
class MockLogger:
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg, exc_info=False): pass
    def critical(self, msg, exc_info=False): pass

class TestHILFlow(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.logger = MockLogger()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_setup_directories_overwrite(self):
        """Test that append_mode=False wipes the directory."""
        # Create a dummy file in the directory
        (self.test_dir / "dummy.txt").touch()
        
        # Run with append_mode=False
        setup_directories(self.test_dir, False, self.logger)
        
        # Check if dummy file is gone
        self.assertFalse((self.test_dir / "dummy.txt").exists())
        # Check if structure is created
        self.assertTrue((self.test_dir / "images/train").exists())

    def test_setup_directories_append(self):
        """Test that append_mode=True preserves the directory."""
        # Create a dummy file
        (self.test_dir / "dummy.txt").touch()
        
        # Run with append_mode=True
        setup_directories(self.test_dir, True, self.logger)
        
        # Check if dummy file still exists
        self.assertTrue((self.test_dir / "dummy.txt").exists())
        # Check if structure is created (idempotency)
        self.assertTrue((self.test_dir / "images/train").exists())

    def test_find_latest_best_model(self):
        """Test finding the latest best.pt file."""
        model_dir = self.test_dir / "models"
        model_dir.mkdir()
        
        # Case 1: No models
        self.assertIsNone(find_latest_best_model(model_dir, self.logger))
        
        # Case 2: One model
        run1 = model_dir / "run1/weights"
        run1.mkdir(parents=True)
        (run1 / "best.pt").touch()
        
        latest = find_latest_best_model(model_dir, self.logger)
        self.assertEqual(latest, run1 / "best.pt")
        
        # Case 3: Two models, verify timestamp sorting
        time.sleep(1.1) # Ensure timestamp difference
        
        run2 = model_dir / "run2/weights"
        run2.mkdir(parents=True)
        (run2 / "best.pt").touch()
        
        latest = find_latest_best_model(model_dir, self.logger)
        self.assertEqual(latest, run2 / "best.pt")

if __name__ == '__main__':
    unittest.main()
