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


try:
    from environment_classes.environment import Environment
    from helpers.environment_helpers import *
    from environment_classes.target_environment import TargetEnvironment
except ImportError as exc:
    sys.stderr.write("Error: failed to import modules ({})".format(exc))


class TestEnvironmentHelpers(unittest.TestCase):
    load_dotenv()
    app = Flask(__name__)
    app.config["MONGO_URI"] = os.getenv("MONGO_TEST_URI")
    mongo = PyMongo(app)
    test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")

    def test_save_ips_for_user(self):
        test_ips = {
            "value": [
                "86.226.152.234",
                "91.141.197.126",
                "65.252.196.198",
                "225.252.227.246",
                "186.31.110.169",
                "51.81.67.248",
                "148.74.192.211",
                "91.29.51.150",
                "115.81.152.124",
                "109.97.229.225",
            ]
        }
        test_environment_id = save_environment_for_user(
            self.mongo.db,
            self.test_user_id,
            Environment(
                {
                    "nr_instances": 2,
                    "environment_options": [{"id": 0, "probability_failure": 0.1}],
                    "machine_type": "e2-low",
                    "machine_series": 'e2'
                }
            ),
        )
        environment = TargetEnvironment(
            self.test_user_id,
            test_environment_id)

        self.assertIsNotNone(test_environment_id)
        test_environment_update = save_ips_for_user(
            self.mongo.db, test_ips, environment
        )
        self.assertIsNotNone(test_environment_update)

    def test_create_environment_data_distribution(self):
        test_ips = {
            "value": [
                "86.226.152.234",
                "91.141.197.126",
                "65.252.196.198",
                "225.252.227.246",
                "186.31.110.169",
                "51.81.67.248",
                "148.74.192.211",
                "91.29.51.150",
                "115.81.152.124",
                "109.97.229.225",
            ]
        }
        test_environment_id = save_environment_for_user(
            self.mongo.db,
            self.test_user_id,
            Environment(
                {
                    "nr_instances": 1,
                    "environment_options": [{"id": 0, "probability_failure": 0.1}],
                    "machine_type": "e2-low",
                    "machine_series": "e2"
                }
            ),
        )
        environment = TargetEnvironment(
            self.test_user_id,
            test_environment_id)
        self.assertIsNotNone(test_environment_id)
        test_environment_data_distribution_insert = (
            create_environment_data_distribution_entry(
                self.mongo.db,
                test_ips,
                environment
            )
        )
        self.assertIsNotNone(test_environment_data_distribution_insert)

    def test_delete_environment_for_user(self):
        test_ips = {
            "value": [
                "86.226.152.234",
                "91.141.197.126",
                "65.252.196.198",
                "225.252.227.246",
                "186.31.110.169",
                "51.81.67.248",
                "148.74.192.211",
                "91.29.51.150",
                "115.81.152.124",
                "109.97.229.225",
            ]
        }
        test_environment_id = save_environment_for_user(
            self.mongo.db,
            self.test_user_id,
            Environment(
                {
                    "nr_instances": 1,
                    "environment_options": [{"id": 0, "probability_failure": 0.1}],
                    "machine_type": "e2-low",
                    "machine_series": "e2"
                }
            ),
        )
        environment = TargetEnvironment(
            self.test_user_id,
            test_environment_id)
        self.assertIsNotNone(test_environment_id)
        test_environment_update = save_ips_for_user(
            self.mongo.db, test_ips, environment
        )
        self.assertIsNotNone(test_environment_update)
        test_environment_delete = delete_environment_for_user(
            self.mongo.db, environment
        )
        self.assertIsNotNone(test_environment_delete)

    def test_get_environment(self):
        test_ips = {
            "value": [
                "86.226.152.234",
                "91.141.197.126",
                "65.252.196.198",
                "225.252.227.246",
                "186.31.110.169",
                "51.81.67.248",
                "148.74.192.211",
                "91.29.51.150",
                "115.81.152.124",
                "109.97.229.225",
            ]
        }

        test_environment_id = save_environment_for_user(
            self.mongo.db,
            self.test_user_id,
            Environment(
                {
                    "nr_instances": 1,
                    "environment_options": [{"id": 0, "probability_failure": 0.1}],
                    "machine_type": "e2-low",
                    "machine_series": "e2"
                }
            ),
        )
        environment = TargetEnvironment(
            self.test_user_id,
            test_environment_id)
        self.assertIsNotNone(test_environment_id)
        test_environment_update = save_ips_for_user(
            self.mongo.db, test_ips, environment
        )
        self.assertIsNotNone(test_environment_update)
        environment_ips = get_environment(
            self.mongo.db, environment
        )
        self.assertEqual(
            set(test_ips["value"]), set(environment_ips["environment_ips"])
        )

    def test_save_environment_test_data_distribution(self):
        test_ips = {
            "value": [
                "86.226.152.234",
                "91.141.197.126",
                "65.252.196.198",
                "225.252.227.246",
                "186.31.110.169",
                "51.81.67.248",
                "148.74.192.211",
                "91.29.51.150",
                "115.81.152.124",
                "109.97.229.225",
            ]
        }
        test_environment_id = save_environment_for_user(
            self.mongo.db,
            self.test_user_id,
            Environment(
                {
                    "nr_instances": 1,
                    "environment_options": [{"id": 0, "probability_failure": 0.1}],
                    "machine_type": "e2-low",
                    "machine_series": "e2"
                }
            ),
        )
        environment = TargetEnvironment(
            self.test_user_id,
            test_environment_id)

        self.assertIsNotNone(test_environment_id)
        test_environment_update = save_ips_for_user(
            self.mongo.db, test_ips, environment
        )
        self.assertIsNotNone(test_environment_update)
        environment_dataset_distribution = {
            "86.226.152.234": 20,
            "91.141.197.126": 15,
            "65.252.196.198": 10,
            "225.252.227.246": 15,
            "186.31.110.169": 35,
            "51.81.67.248": 15,
            "148.74.192.211": 5,
            "91.29.51.150": 7,
            "115.81.152.124": 15,
            "109.97.229.225": 1,
        }
        test_dataset_length = 100
        for environment_ip, distribution in environment_dataset_distribution.items():
            environment_dataset_distribution[environment_ip] = random.sample(
                range(1, test_dataset_length), distribution
            )
        test_distribution = save_environment_test_data_distribution(
            self.mongo.db,
            environment,
            environment_dataset_distribution,
        )
        self.assertIsNotNone(test_distribution)

    def test_get_environment_data_distribution(self):
        test_ips = {
            "value": [
                "86.226.152.234",
                "91.141.197.126",
                "65.252.196.198",
                "225.252.227.246",
                "186.31.110.169",
                "51.81.67.248",
                "148.74.192.211",
                "91.29.51.150",
                "115.81.152.124",
                "109.97.229.225",
            ]
        }
        test_environment_id = save_environment_for_user(
            self.mongo.db,
            self.test_user_id,
            Environment(
                {
                    "nr_instances": 1,
                    "environment_options": [{"id": 0, "probability_failure": 0.1}],
                    "machine_type": "e2-low",
                    "machine_series": "e2"
                }
            ),
        )

        environment = TargetEnvironment(
            self.test_user_id,
            test_environment_id)

        self.assertIsNotNone(test_environment_id)
        test_environment_update = save_ips_for_user(
            self.mongo.db, test_ips, environment
        )
        self.assertIsNotNone(test_environment_update)
        environment_dataset_distribution = {
            "86.226.152.234": 20,
            "91.141.197.126": 15,
            "65.252.196.198": 10,
            "225.252.227.246": 15,
            "186.31.110.169": 35,
            "51.81.67.248": 15,
            "148.74.192.211": 5,
            "91.29.51.150": 7,
            "115.81.152.124": 15,
            "109.97.229.225": 1,
        }
        test_dataset_length = 100
        for environment_ip, distribution in environment_dataset_distribution.items():
            environment_dataset_distribution[environment_ip] = random.sample(
                range(1, test_dataset_length), distribution
            )
        test_distribution = save_environment_test_data_distribution(
            self.mongo.db,
            environment,
            environment_dataset_distribution,
        )
        self.assertIsNotNone(test_distribution)
        test_dataset_distribution = get_environment_data_distribution(
            self.mongo.db, environment
        )
        self.assertEqual(
            test_dataset_distribution["distributions"], environment_dataset_distribution
        )

    def test_delete_data_distribution_for_user(self):
        test_ips = {
            "value": [
                "86.26.152.234",
                "91.141.197.126",
                "65.252.196.198",
                "225.252.227.246",
                "186.31.110.169",
                "51.81.67.248",
                "148.74.192.211",
            ]
        }
        test_environment_id = save_environment_for_user(
            self.mongo.db,
            self.test_user_id,
            Environment(
                {
                    "nr_instances": 1,
                    "environment_options": [{"id": 0, "probability_failure": 0.1}],
                    "machine_type": "e2-low",
                    "machine_series": "e2"
                }
            ),
        )
        environment = TargetEnvironment(
            self.test_user_id,
            test_environment_id)
        self.assertIsNotNone(test_environment_id)
        test_environment_update = save_ips_for_user(
            self.mongo.db, test_ips, environment
        )
        self.assertIsNotNone(test_environment_update)
        test_environment_delete = delete_environment_train_distribution(
            self.mongo.db, environment
        )
        self.assertIsNotNone(test_environment_delete)
        test_environment_delete = delete_environment_data_distribution(
            self.mongo.db, environment
        )
        self.assertIsNotNone(test_environment_delete)
