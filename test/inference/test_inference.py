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
import asyncio
from unittest import mock

import pytest

from oasis.inference import InferencerManager
from oasis.social_platform import Channel


@pytest.mark.asyncio
async def test_manager_run_with_mocked_response():
    channel = Channel()

    # Setup the InferencerManager with the real channel
    manager = InferencerManager(
        channel=channel,
        model_type="llama-3",
        model_path="/path/to/model",
        stop_tokens=["\n"],
        server_url=[{
            "host": "localhost",
            "ports": [8000]
        }],
    )

    # Mocking the run method of model_backend to return a mocked response
    mock_response = mock.Mock()
    mock_response.choices = [
        mock.Mock(message=mock.Mock(content="Mock Response"))
    ]

    # Mocking channel.send_to as well
    with mock.patch.object(manager.threads[0].model_backend,
                           'run',
                           return_value=mock_response):

        openai_messages = [{
            "role": "assistant",
            "content": 'mock_message',
        }]

        # Run the manager asynchronously
        task = asyncio.create_task(manager.run())

        # Add a message to the receive_queue
        mes_id = await channel.write_to_receive_queue(openai_messages)
        mes_id, content = await channel.read_from_send_queue(mes_id)
        assert content == "Mock Response"

        await manager.stop()
        await task


@pytest.mark.asyncio
async def test_multiple_threads():
    # Create a Channel instance
    channel = Channel()

    # Set up multiple ports to simulate multiple threads
    server_url = [{
        "host": "localhost",
        "ports": [8000, 8001, 8002]
    }  # 3 ports
                  ]

    # Initialize InferencerManager with multiple threads
    manager = InferencerManager(
        channel=channel,
        model_type="llama-3",
        model_path="/path/to/model",
        stop_tokens=["\n"],
        server_url=server_url,
        threads_per_port=2,  # 2 threads per port
    )

    # Mock the response for multiple threads
    mock_response = mock.Mock()
    mock_response.choices = [
        mock.Mock(message=mock.Mock(content="Mock Response"))
    ]

    # Replace the model_backend.run method for all threads with the mock
    for thread in manager.threads:
        thread.model_backend.run = mock.Mock(return_value=mock_response)

    # Start the manager
    task = asyncio.create_task(manager.run())

    # Send multiple messages to the queue
    openai_messages = [{
        "role": "assistant",
        "content": f"mock_message_{i}"
    } for i in range(10)]

    # Write messages to the receive queue
    message_ids = []
    for message in openai_messages:
        message_id = await channel.write_to_receive_queue([message])
        message_ids.append(message_id)

    # Read results from the send queue
    results = []
    for message_id in message_ids:
        _, content = await channel.read_from_send_queue(message_id)
        results.append(content)

    # Validate the results
    assert len(results) == 10  # Ensure all messages are processed
    assert all(content == "Mock Response"
               for content in results)  # Ensure all responses are correct

    # Stop the manager
    await manager.stop()
    await task
