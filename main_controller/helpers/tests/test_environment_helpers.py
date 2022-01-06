import unittest
import sys
from dotenv import load_dotenv
from flask_pymongo import PyMongo
from flask import Flask
import os
import random

sys.path.insert(0, "../../")
sys.path.insert(1, "../")

from environment_helpers import *


class TestEnvironmentHelpers(unittest.TestCase):
    load_dotenv()
    app = Flask(__name__)
    app.config["MONGO_URI"] = os.getenv("MONGO_TEST_URI")
    mongo = PyMongo(app)
    test_user_id = 1

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

        test_environment_ip = save_ips_for_user(
            self.mongo.db, test_ips, self.test_user_id
        )
        self.assertIsNotNone(test_environment_ip)

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
        test_environment_ip = save_ips_for_user(
            self.mongo.db, test_ips, self.test_user_id
        )
        self.assertIsNotNone(test_environment_ip)
        delete_environment_for_user(
            self.mongo.db, test_environment_ip, self.test_user_id
        )

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

        test_environment_ip = save_ips_for_user(
            self.mongo.db, test_ips, self.test_user_id
        )
        self.assertIsNotNone(test_environment_ip)
        environment_ips = get_environment(
            self.mongo.db, str(test_environment_ip), self.test_user_id
        )
        self.assertEqual(
            set(test_ips["value"]), set(environment_ips["environment_ips"])
        )

    def test_save_environment_data_distribution(self):
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
        test_environment_ip = save_ips_for_user(
            self.mongo.db, test_ips, self.test_user_id
        )
        self.assertIsNotNone(test_environment_ip)
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
        test_distribution_id = save_environment_data_distribution(
            self.mongo.db,
            str(test_environment_ip),
            self.test_user_id,
            environment_dataset_distribution,
        )
        self.assertIsNotNone(test_distribution_id)

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
        test_environment_ip = save_ips_for_user(
            self.mongo.db, test_ips, self.test_user_id
        )
        self.assertIsNotNone(test_environment_ip)
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
        test_distribution_id = save_environment_data_distribution(
            self.mongo.db,
            str(test_environment_ip),
            self.test_user_id,
            environment_dataset_distribution,
        )
        self.assertIsNotNone(test_distribution_id)
        test_dataset_distribution = get_environment_data_distribution(
            self.mongo.db, str(test_environment_ip), self.test_user_id
        )
        self.assertEqual(
            test_dataset_distribution["distributions"], environment_dataset_distribution
        )
