import sys

sys.path.insert(0, "../../")
sys.path.insert(1, "../")
from app import app
import os
from helpers.request_helpers import *
from unittest.mock import patch, mock_open
from werkzeug.exceptions import HTTPException
import pytest


def test_request_wrapper_post_json_to_instance(response_mock):
    with response_mock([
        'POST http://{}:{}/instance/probabilityoffailure -> 200 :Added probability of failure'.format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT")),
    ]):
        res = request_wrapper(lambda: post_json_to_instance("http://{}:{}/instance/probabilityoffailure".format(
            "192.1.1.0", os.getenv("ENVIRONMENTS_PORT")), {"probabilityOfFailure": 0.1}))
        assert res.ok
        assert res.content == b'Added probability of failure'


def test_request_wrapper_get_to_instance(response_mock):
    with response_mock([
        'GET http://{}:{}/instance/availability -> 200 :Available'.format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT")),
    ]):
        res = request_wrapper(lambda: get_to_instance(
            "http://{}:{}/instance/availability".format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT"))))
        assert res.ok
        assert res.content == b'Available'


def test_request_wrapper_get_to_instance_fail(response_mock):
    with response_mock([
        'GET http://{}:{}/instance/availability -> 500 :Server error'.format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT")),
    ]):
        with app.app_context():
            with pytest.raises(HTTPException) as httperror:
                res = request_wrapper(lambda: get_to_instance(
                    "http://{}:{}/instance/availability".format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT"))))
                assert not res.ok
                assert res.content == b'Getting from: 192.1.1.0 went wrong. Response: Server error'

                assert 500 == httperror.value.code


def test_request_wrapper_get_to_instance_timeout(mocker):
    mocker.patch('requests.get', side_effect=requests.exceptions.ConnectTimeout())
    with app.app_context():
        with pytest.raises(HTTPException) as httperror:
            res = request_wrapper(lambda: get_to_instance(
                "http://{}:{}/instance/availability".format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT"))))
            assert res.content == "Status code: 408\nMessage:A request to an instance timedout."

            assert 408 == httperror.value.code


def test_request_wrapper_get_to_instance_request_exception(mocker):
    mocker.patch('requests.get', side_effect=requests.exceptions.RequestException())
    with app.app_context():
        with pytest.raises(HTTPException) as httperror:
            res = request_wrapper(lambda: get_to_instance(
                "http://{}:{}/instance/availability".format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT"))))
            assert res.content == "Status code: 500\nMessage:A request failed due to an internal server error on instance"
            assert 500 == httperror.value.code


def test_request_wrapper_get_to_instance_request_exception_http_error(mocker):
    mocker.patch('requests.get', side_effect=requests.exceptions.HTTPError())
    with app.app_context():
        with pytest.raises(HTTPException) as httperror:
            res = request_wrapper(lambda: get_to_instance(
                "http://{}:{}/instance/availability".format("192.1.1.0", os.getenv("ENVIRONMENTS_PORT"))))
            assert res.content == "Status code: 500\nMessage:A request failed due to an internal server error"
            assert 500 == httperror.value.code


def test_request_wrapper_get_to_instance_value_error(mocker):
    mocker.patch('requests.get', return_value={"Some text"})
    with app.app_context():
        with pytest.raises(HTTPException) as httperror:
            res = request_wrapper(lambda: get_to_instance(123))
            assert res.content == "Status code: 500\nMessage:A request failed due to an internal server error (ValueError)"
            assert 500 == httperror.value.code
