from controller_helpers import *
import unittest
import sys
from dotenv import load_dotenv
from flask_pymongo import PyMongo
from flask import Flask
import os
import random
from bson.objectid import ObjectId

sys.path.insert(0, "../../")
sys.path.insert(1, "../")


class TestControllerHelpers(unittest.TestCase):
    load_dotenv()
    app = Flask(__name__)
    app.config["MONGO_URI"] = os.getenv("MONGO_TEST_URI")
    mongo = PyMongo(app)
    test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")

    def test_write_to_train_log(self):
        test_log = list()
        dummy_data = ["Some text", "Some more text", "More text"]
        write_to_train_log(test_log, dummy_data)
        self.assertEquals(test_log, dummy_data)

    def test_write_to_train_log_empty(self):
        test_log = list()
        dummy_data = []
        write_to_train_log(test_log, dummy_data)
        self.assertEquals(test_log, dummy_data)
