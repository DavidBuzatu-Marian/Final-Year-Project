import asyncio


class MockStreamResponse():
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
