import sys
import asyncio

sys.path.insert(0, "../../")
sys.path.insert(1, "../")
from app import app
import os
from helpers.controller_hepers import *
from unittest.mock import patch, mock_open
from werkzeug.exceptions import HTTPException
from environment_classes.target_environment import TargetEnvironment
import pytest


def test_async_train_post_with_one_instance(response_mock):
    with response_mock(['POST http://{}:{}/model/train -> 200: This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything'.format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT")),
                        ]):
        test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")
        test_environment_id = ObjectId("61febbb5d4289b4b0b4a48d4")
        environment = TargetEnvironment(
            test_user_id,
            test_environment_id)

        instances_error = async_train_post(["192.1.1.0"], {
            "loss": {
                "loss_type": "CrossEntropyLoss",
                "parameters": {}
            },
            "optimizer": {
                "optimizer_type": "RMSprop",
                "parameters": {
                    "lr": 0.001,
                    "weight_decay": 0.00000001,
                    "momentum": 0.9
                }
            },
            "hyperparameters": {
                "epochs": 60,
                "batch_size": 4,
                "reshape": "4, 1, 96, 96",
                "normalize": True
            }
        }, environment)
        assert len(instances_error) == 0
        with open(
            "./models/model-{}-{}-{}.pth".format(environment.id, environment.user_id, "192.1.1.0"), "r"
        ) as instance_model_file:
            assert instance_model_file.read() == 'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything'


# def test_request_wrapper_get_to_instance_fail(response_mock):
#     with response_mock([
#         'GET http://{}:{}/instance/availability -> 500 :Server error'.format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT")),
#     ]):
#         with app.app_context():
#             with pytest.raises(HTTPException) as httperror:
#                 res = request_wrapper(lambda: get_to_instance(
#                     "http://{}:{}/instance/availability".format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT"))))
#                 assert not res.ok
#                 assert res.content == b'Getting from: 192.1.1.0 went wrong. Response: Server error'

#                 assert 500 == httperror.value.code
