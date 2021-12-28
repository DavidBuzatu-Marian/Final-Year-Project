import unittest
import sys
import asyncio
from aiohttp import ClientSession
import json
from logging import error

sys.path.insert(0, "../../")
sys.path.insert(1, "../")

from request_helpers import *

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


class TestRequestHelpers(unittest.TestCase):
    # Test to dummy server
    @async_test
    async def test_post_to_instance(self):
        async with ClientSession() as client_session:
            response = await post_to_instance(
                "https://jsonplaceholder.typicode.com/posts",
                client_session,
                json.dumps(
                    {
                        "title": "foo",
                        "body": "bar",
                        "userId": 1,
                    }
                ),
                {
                    "Content-type": "application/json; charset=UTF-8",
                },
            )
            self.assertEqual(response.status, 201)
