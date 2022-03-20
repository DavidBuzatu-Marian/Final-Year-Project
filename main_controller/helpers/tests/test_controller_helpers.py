from bson.objectid import ObjectId
import os
from flask import Flask
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import sys
import unittest

from environment_classes.target_environment import TargetEnvironment


sys.path.insert(0, "../../")
sys.path.insert(1, "../")
from helpers.controller_hepers import *


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
        instances_error = dict()
        data = process_training_results(test_iteration, instances,
                                        initial_instances, instances_error)
        test_real_data = [
            "Iteration nr: %d" % (test_iteration),
            "Instance IP: %s , Training result: %s , Errors: %s" % (
                "182.49.34.34", "Contributed to current training round", "No error detected"),
            "Instance IP: %s , Training result: %s , Errors: %s" % (
                "182.49.34.35", "Contributed to current training round", "No error detected"),
            "Instance IP: %s , Training result: %s , Errors: %s" % (
                "182.49.35.34", "Contributed to current training round", "No error detected")
        ]
        self.assertEqual(sorted(data), sorted(test_real_data))

    def test_process_training_results_less_instances(self):
        test_iteration = 0
        instances = {"182.49.34.34", "182.49.34.35", "182.49.35.34"}
        initial_instances = {"182.49.35.40", "182.49.34.34",
                             "182.49.34.35", "182.49.35.34", "12.49.35.34"}
        instances_error = {
            "182.49.35.40": "Some 500 error",
            "12.49.35.34": "Some 500 error"
        }
        data = process_training_results(test_iteration, instances,
                                        initial_instances, instances_error)
        test_real_data = [
            "Iteration nr: %d" % (test_iteration),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.35.40", "Does not contribute to training process anymore", "Some 500 error"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.34.34", "Contributed to current training round", "No error detected"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.34.35", "Contributed to current training round", "No error detected"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.35.34", "Contributed to current training round", "No error detected"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("12.49.35.34", "Does not contribute to training process anymore", "Some 500 error")]
        self.assertEqual(sorted(data), sorted(test_real_data))

    def test_process_training_results_no_instances(self):
        test_iteration = 10
        instances = {}
        initial_instances = {"182.49.35.40", "182.49.34.34",
                             "182.49.34.35", "182.49.35.34", "12.49.35.34"}
        instances_error = {
            "182.49.35.40": "Some 500 error",
            "182.49.34.34": "Some 500 error",
            "182.49.34.35": "Some 500 error",
            "182.49.35.34": "Some 500 error",
            "12.49.35.34": "Some 500 error"
        }
        data = process_training_results(test_iteration, instances,
                                        initial_instances, instances_error)
        test_real_data = [
            "Iteration nr: %d" % (test_iteration),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.35.40", "Does not contribute to training process anymore", "Some 500 error"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.34.34", "Does not contribute to training process anymore", "Some 500 error"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.34.35", "Does not contribute to training process anymore", "Some 500 error"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.35.34", "Does not contribute to training process anymore", "Some 500 error"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("12.49.35.34", "Does not contribute to training process anymore", "Some 500 error")]
        self.assertEqual(sorted(data), sorted(test_real_data))

    def test_write_logs_to_database(self):
        test_iteration = 10
        test_real_data = [
            "Iteration nr: %d" % (test_iteration),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.35.40", "Does not contribute to training process anymore", "Some 500 error"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.34.34", "Contributed to current training round", "No error detected"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.34.35", "Contributed to current training round", "No error detected"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("182.49.35.34", "Contributed to current training round", "No error detected"),
            "Instance IP: %s , Training result: %s , Errors: %s" %
            ("12.49.35.34", "Does not contribute to training process anymore", "Some 500 error")]
        environment = TargetEnvironment(
            self.test_user_id, self.test_environment_id
        )
        insert_result = write_logs_to_database(
            self.mongo.db, test_real_data, environment)
        self.assertIsNotNone(insert_result)

    def test_delete_model_from_path(self):
        with open("./models/test-model.pth", "wb+") as test_file:
            test_file.write(b"Some content")
        delete_model_from_path("./models/test-model.pth")
        assert not os.path.isfile('./models/test-model.pth')

    def test_delete_model_from_path_not_found(self):
        with self.assertRaises(FileNotFoundError, msg="Model not found:./models/test-model2.pth"):
            delete_model_from_path("./models/test-model2.pth")

    def get_training_iterations(self):
        test_json = {"training_iterations": 3}
        self.assertEqual(get_training_iterations(test_json), 3)

    def get_instance_training_parameters(self):
        test_json = {"environment_parameters": {}}
        self.assertEqual(get_instance_training_parameters(test_json), {})\


    def get_instance_training_parameters(self):
        test_json = {"environment_model_network_options": {[{"layer": {}}]}}
        self.assertEqual(get_instance_training_parameters(test_json), [{"layer": {}}])


def test_create_model_success_one_instance(response_mock):
    with response_mock([
        'POST http://{}:{}/model/create -> 200 :Created model'.format("192.1.1.1", os.getenv("ENVIRONMENTS_PORT")),
    ]):
        create_model(set(["192.1.1.1"]), {})


def test_create_model_success_multiple_instances(response_mock):
    with response_mock([
        'POST http://{}:{}/model/create -> 200 :Created model'.format("192.1.1.1", os.getenv("ENVIRONMENTS_PORT")),
        'POST http://{}:{}/model/create -> 200 :Created model'.format("192.1.1.2", os.getenv("ENVIRONMENTS_PORT")),
        'POST http://{}:{}/model/create -> 200 :Created model'.format("192.1.1.3", os.getenv("ENVIRONMENTS_PORT")),
        'POST http://{}:{}/model/create -> 200 :Created model'.format("191.1.1.1", os.getenv("ENVIRONMENTS_PORT")),
    ]):
        create_model(set(["192.1.1.1", "192.1.1.2", "192.1.1.3", "191.1.1.1"]), {})
