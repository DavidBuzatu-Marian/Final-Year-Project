import sys
import asyncio

from mocks.mock_stream_response import MockStreamResponse
from mocks.mock_text_response import MockTextResponse

sys.path.insert(0, "../../")
sys.path.insert(1, "../")
from app import app
import os
from helpers.controller_hepers import *
from unittest.mock import patch, mock_open
from werkzeug.exceptions import HTTPException
from environment_classes.target_environment import TargetEnvironment
import pytest


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


def test_async_train_post_with_one_instance(mocker):
    mock_response = MockStreamResponse(b'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything', 200)
    mocker.patch('aiohttp.ClientSession.post', return_value=mock_response)

    test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")
    test_environment_id = ObjectId("61febbb5d4289b4b0b4a48d4")
    environment = TargetEnvironment(
        test_user_id,
        test_environment_id)
    instances = set(["192.1.1.0"])
    instances_error = async_train_post(
        instances,
        test_json, environment)
    assert len(instances_error) == 0
    with open(
        "./models/model-{}-{}-{}.pth".format(environment.id, environment.user_id, "192.1.1.0"), "r"
    ) as instance_model_file:
        assert instance_model_file.read() == 'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything'
    assert "192.1.1.0" in instances


def test_async_train_post_with_one_instance_fail(mocker):
    mock_response = MockTextResponse('Some 500 error message from instance', 500)
    mocker.patch('aiohttp.ClientSession.post', return_value=mock_response)

    test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")
    test_environment_id = ObjectId("61febbb5d4289b4b0b4a48d4")
    environment = TargetEnvironment(
        test_user_id,
        test_environment_id)
    instances = set(["192.1.1.0"])
    instances_error = async_train_post(
        instances,
        test_json, environment)

    assert len(instances_error) == 1
    assert instances_error["192.1.1.0"] == 'Some 500 error message from instance'
    assert "192.1.1.0" not in instances


def test_async_train_post_with_more_instances(mocker):
    mock_responses = [MockStreamResponse(b'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything', 200),
                      MockStreamResponse(b'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything', 200),
                      MockStreamResponse(b'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything', 200)]
    mocker.patch('aiohttp.ClientSession.post', side_effect=mock_responses)

    test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")
    test_environment_id = ObjectId("61febbb5d4289b4b0b4a48d4")
    environment = TargetEnvironment(
        test_user_id,
        test_environment_id)
    instances = set(["192.12.1.0", "192.1.1.12", "192.1.12.3"])
    instances_error = async_train_post(
        instances,
        test_json, environment)
    assert len(instances_error) == 0
    with open(
        "./models/model-{}-{}-{}.pth".format(environment.id, environment.user_id, "192.12.1.0"), "r"
    ) as instance_model_file:
        assert instance_model_file.read() == 'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything'
    with open(
        "./models/model-{}-{}-{}.pth".format(environment.id, environment.user_id, "192.1.1.12"), "r"
    ) as instance_model_file:
        assert instance_model_file.read() == 'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything'
    with open(
        "./models/model-{}-{}-{}.pth".format(environment.id, environment.user_id, "192.1.12.3"), "r"
    ) as instance_model_file:
        assert instance_model_file.read() == 'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything'
    assert "192.12.1.0" in instances
    assert "192.1.1.12" in instances
    assert "192.1.12.3" in instances


def test_async_train_post_with_more_instances_fail(mocker):
    mock_response = MockTextResponse('Some 500 error message from instance', 500)
    mocker.patch('aiohttp.ClientSession.post', return_value=mock_response)

    test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")
    test_environment_id = ObjectId("61febbb5d4289b4b0b4a48d4")
    environment = TargetEnvironment(
        test_user_id,
        test_environment_id)
    instances = set(["192.1.1.0", "192.1.1.1", "192.1.2.3"])
    instances_error = async_train_post(
        instances,
        test_json, environment)
    assert len(instances_error) == 3
    assert instances_error["192.1.1.0"] == 'Some 500 error message from instance'
    assert instances_error["192.1.1.1"] == 'Some 500 error message from instance'
    assert instances_error["192.1.2.3"] == 'Some 500 error message from instance'
    assert "192.1.1.0" not in instances
    assert "192.1.1.1" not in instances
    assert "192.1.2.3" not in instances


def test_async_train_post_with_more_instances_two_fail(mocker):
    mock_responses = [MockStreamResponse(b'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything', 200),
                      MockTextResponse('Some 500 error message from instance', 500),
                      MockTextResponse('Some 500 error message from instance', 500)]
    mocker.patch('aiohttp.ClientSession.post', side_effect=mock_responses)

    test_user_id = ObjectId("61febbb5d4289b4b0b4a48d5")
    test_environment_id = ObjectId("61febbb5d4289b4b0b4a48d4")
    environment = TargetEnvironment(
        test_user_id,
        test_environment_id)
    instances = set(["192.12.11.0", "192.1.11.12", "192.1.112.3"])
    instances_error = async_train_post(
        instances,
        test_json, environment)
    assert len(instances_error) == 2
    assert len(instances) == 1
    with open(
        "./models/model-{}-{}-{}.pth".format(environment.id, environment.user_id, min(instances)), "r"
    ) as instance_model_file:
        assert instance_model_file.read() == 'This will be a pytorch model file,\
                        but it is still bytes, thus it can be anything'
    for _, err in instances_error.items():
        assert err == 'Some 500 error message from instance'
