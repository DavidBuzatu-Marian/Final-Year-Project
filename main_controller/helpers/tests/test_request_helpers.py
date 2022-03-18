import sys
import asyncio

sys.path.insert(0, "../../")
sys.path.insert(1, "../")
from app import app
import os
from helpers.request_helpers import *
from unittest.mock import patch, mock_open
from werkzeug.exceptions import HTTPException
import pytest
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
