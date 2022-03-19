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


class MockResponse():
    def __init__(self, data, status):
        self.status = status
        stream = asyncio.StreamReader()
        stream.feed_data(data)
        stream.feed_eof()
        self.content = stream

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


@pytest.mark.asyncio
def test_async_train_post_with_one_instance(mocker):

    test_json = {
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
    }
    mock_response = MockResponse(b'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything', 200)
    mocker.patch('aiohttp.ClientSession.post', return_value=mock_response)

    test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")
    test_environment_id = ObjectId("61febbb5d4289b4b0b4a48d4")
    environment = TargetEnvironment(
        test_user_id,
        test_environment_id)

    instances_error = async_train_post(
        ["192.1.1.0"],
        test_json, environment)
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
