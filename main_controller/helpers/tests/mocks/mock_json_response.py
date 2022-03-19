
class MockJSONResponse():
    def __init__(self, json, status):
        self.status = status
        self._json = json
        self.ok = True

    def json(self):
        return self._json

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self
