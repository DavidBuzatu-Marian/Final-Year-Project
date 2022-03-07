from bson.objectid import ObjectId
import os
from flask import Flask
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import sys
import unittest


sys.path.insert(0, "../../")
sys.path.insert(1, "../")
from controller_hepers import *


class TestControllerHelpers(unittest.TestCase):
    load_dotenv()
    app = Flask(__name__)
    app.config["MONGO_URI"] = os.getenv("MONGO_TEST_URI")
    mongo = PyMongo(app)
    test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")
    test_environment_id = ObjectId("61febbb5d4289b4b0b4a48f5")

    def test_write_to_train_log(self):
        test_log = list()
        dummy_data = ["Some text", "Some more text", "More text"]
        write_to_train_log(test_log, dummy_data)
        self.assertEqual(test_log, dummy_data)

    def test_write_to_train_log_empty(self):
        test_log = list()
        dummy_data = []
        write_to_train_log(test_log, dummy_data)
        self.assertEqual(test_log, dummy_data)

    def test_check_contribution(self):
        instances = {"182.49.34.34", "182.49.34.35", "182.49.35.34"}
        instance_ip = "182.49.34.34"
        test_result = "Contributed to current training round"
        result = check_contribution(instance_ip, instances)
        self.assertEqual(test_result, result)

    def test_check_contribution_ip_not_in_set(self):
        instances = {"182.49.34.34", "182.49.34.35", "182.49.35.34"}
        instance_ip = "182.49.34.79"
        test_result = "Does not contribute to training process anymore"
        result = check_contribution(instance_ip, instances)
        self.assertEqual(test_result, result)

    def test_check_contribution_empty_set(self):
        instances = {}
        instance_ip = "182.49.34.79"
        test_result = "Does not contribute to training process anymore"
        result = check_contribution(instance_ip, instances)
        self.assertEqual(test_result, result)

    def test_process_training_results(self):
        test_iteration = 0
        instances = {"182.49.34.34", "182.49.34.35", "182.49.35.34"}
        initial_instances = instances
        data = process_training_results(test_iteration, instances, initial_instances)
        test_real_data = [
            "Iteration nr: %d" % (test_iteration),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.34.34", "Contributed to current training round"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.34.35", "Contributed to current training round"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.35.34", "Contributed to current training round")
        ]
        self.assertEqual(sorted(data), sorted(test_real_data))

    def test_process_training_results_less_instances(self):
        test_iteration = 0
        instances = {"182.49.34.34", "182.49.34.35", "182.49.35.34"}
        initial_instances = {"182.49.35.40", "182.49.34.34",
                             "182.49.34.35", "182.49.35.34", "12.49.35.34"}
        data = process_training_results(test_iteration, instances, initial_instances)
        test_real_data = [
            "Iteration nr: %d" % (test_iteration),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.35.40", "Does not contribute to training process anymore"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.34.34", "Contributed to current training round"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.34.35", "Contributed to current training round"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.35.34", "Contributed to current training round"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "12.49.35.34", "Does not contribute to training process anymore")
        ]
        self.assertEqual(sorted(data), sorted(test_real_data))

    def test_process_training_results_no_instances(self):
        test_iteration = 10
        instances = {}
        initial_instances = {"182.49.35.40", "182.49.34.34",
                             "182.49.34.35", "182.49.35.34", "12.49.35.34"}
        data = process_training_results(test_iteration, instances, initial_instances)
        test_real_data = [
            "Iteration nr: %d" % (test_iteration),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.35.40", "Does not contribute to training process anymore"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.34.34", "Does not contribute to training process anymore"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.34.35", "Does not contribute to training process anymore"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.35.34", "Does not contribute to training process anymore"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "12.49.35.34", "Does not contribute to training process anymore")
        ]
        self.assertEqual(sorted(data), sorted(test_real_data))

    def test_write_logs_to_database(self):
        test_iteration = 10
        test_real_data = [
            "Iteration nr: %d" % (test_iteration),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.35.40", "Does not contribute to training process anymore"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.34.34", "Contributed to current training round"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.34.35", "Contributed to current training round"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "182.49.35.34", "Contributed to current training round"),
            "Instance IP: %s , Training result: %s , Other: None" % (
                "12.49.35.34", "Does not contribute to training process anymore")
        ]
        insert_result = write_logs_to_database(
            self.mongo.db, test_real_data, self.test_user_id, self.test_environment_id)
        self.assertIsNotNone(insert_result)
