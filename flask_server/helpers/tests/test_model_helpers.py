import unittest
from logging import error
import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../")

from model_helpers import *


class TestAppHelpers(unittest.TestCase):
    def test_probability_of_failure(self):
        failed = is_failing(0.01)
        self.assertIn(failed, [True, False])