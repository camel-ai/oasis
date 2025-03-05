# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
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
