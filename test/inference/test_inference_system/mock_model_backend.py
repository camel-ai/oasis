# mock_model_backend.py
import asyncio
import random

class MockModelBackend:
    async def run(self, message):
        # Simulate processing time between 0.1 to 0.5 seconds
        await asyncio.sleep(random.uniform(0.1, 0.5))
        #await asyncio.sleep(0.5)
        # Simulate a response
        return MockResponse(f"Processed: {message}")

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockMessage:
    def __init__(self, content):
        self.content = content