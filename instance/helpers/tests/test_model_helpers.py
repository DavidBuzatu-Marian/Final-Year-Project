import unittest
from logging import error
import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../")

from helpers.model_helpers import *


class TestModelHelpers(unittest.TestCase):
    def test_probability_of_failure(self):
        failed = is_failing(0.01)
        self.assertIn(failed, [True, False])

    def test_probability_of_failure_0(self):
        failed = is_failing(0)
        self.assertEqual(False, failed)

    def test_probability_of_failure_100(self):
        failed = is_failing(100)
        self.assertEqual(True, failed)
