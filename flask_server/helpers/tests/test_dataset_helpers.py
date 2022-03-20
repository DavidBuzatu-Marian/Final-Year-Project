import unittest
from logging import error
import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../")

from dataset_helpers import *
from model_helpers import save_file


class TestDatasetHelpers(unittest.TestCase):
    def test_delete_data_from_path(self):
        file = open("./__init__.py", "r")
        file.filename = "__init__.py"
        save_file(file, "./test/test.py")
        file.close()
        delete_data_from_path("./test")

    def test_delete_model_from_path_exists(self):
        file = open("./__init__.py", "r")
        file.filename = "__init__.py"
        save_file(file, "./model.pth")
        file.close()
        delete_model_from_path('./model.pth')

    def test_delete_model_from_path_not_exists(self):
        with self.assertRaises(FileNotFoundError):
            delete_model_from_path('./inexistent.pth')
