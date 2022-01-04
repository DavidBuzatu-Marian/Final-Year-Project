import unittest
import sys
import asyncio
from aiohttp import ClientSession
from logging import error
from dotenv import load_dotenv
from flask_pymongo import PyMongo
from flask import Flask
import os

sys.path.insert(0, "../../")
sys.path.insert(1, "../")

from environment_helpers import *

# Reference used for testing async code:
# https://stackoverflow.com/a/46324983/11023871
# https://stackoverflow.com/a/23036785/11023871
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()

    return wrapper


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
