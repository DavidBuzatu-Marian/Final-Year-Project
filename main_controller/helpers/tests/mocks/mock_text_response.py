
class MockTextResponse():
    def __init__(self, text, status):
        self.status = status
        self.text = text

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self
