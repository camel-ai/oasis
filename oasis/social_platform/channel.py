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
import uuid


class AsyncSafeDict:
    r"""
    A class provides a dictionary interface protected
    by an asyncio Lock to ensure safe concurrent access
    in asynchronous environments.
    """

    def __init__(self):
        r"""
        Initialize the AsyncSafeDict with an empty dictionary
        and a new lock.
        """
        self.dict = {}
        self.lock = asyncio.Lock()

    async def put(self, key, value):
        r"""
        Safely insert or update a key-value pair in the dictionary.

        Args:
            key: The key to insert/update.
            value: The value to associate with the key.
        """
        async with self.lock:
            self.dict[key] = value

    async def get(self, key, default=None):
        r"""
        Safely retrieve a value from the dictionary.

        Args:
            key: The key to retrieve.
            default: Value to return if key is not found.

        Returns:
            The value associated with the key,
            or default if key doesn't exist.
        """
        async with self.lock:
            return self.dict.get(key, default)

    async def pop(self, key, default=None):
        r"""
        Safely remove a value from the dictionary.

        Args:
            key: The key to remove.
            default: Value to remove if key is not found.

        Returns:
            The value associated with the key,
            or default if key doesn't exist
        """
        async with self.lock:
            return self.dict.pop(key, default)

    async def keys(self):
        r"""
        Safely retrieve all keys from the dictionary.

        Returns:
            List: A list of all keys currently in the dictionary.
        """
        async with self.lock:
            return list(self.dict.keys())


class Channel:
    r"""
    A class provides asynchronous communication approaches
    for message passing.
    """

    def __init__(self):
        r"""
        Initialize the Channel with a received queue and sent dictionary.
        """
        self.receive_queue = asyncio.Queue()  # Used to store received messages
        # Using an asynchronous safe dictionary to store messages to be sent
        self.send_dict = AsyncSafeDict()

    async def receive_from(self):
        r"""
        Receive a message from the channel's received queue.

        Returns:
            tuple: Message from the received queue.
        """
        message = await self.receive_queue.get()
        return message

    async def send_to(self, message):
        r"""
        Send a message to the channel's send dictionary.

        Args:
            message: The message to send.
        """
        # message_id is the first element of the message
        message_id = message[0]
        await self.send_dict.put(message_id, message)

    async def write_to_receive_queue(self, action_info):
        r"""
        Write a new message to the message receiving queue
        with a generated UUID.

        Args:
            action_info: The message content to enqueue.

        Returns:
            str: The generated message UUID.
        """
        message_id = str(uuid.uuid4())
        await self.receive_queue.put((message_id, action_info))
        return message_id

    async def read_from_send_queue(self, message_id):
        r"""
        Continuously check for and retrieve a specific message
        from the message sent dictionary.

        Args:
            message_id: The UUID of the message to retrieve.

        Returns:
            str: The message content when found.
        """
        while True:
            if message_id in await self.send_dict.keys():
                # Attempting to retrieve the message
                message = await self.send_dict.pop(message_id, None)
                if message:
                    return message  # Return the found message
            # Temporarily suspend to avoid tight looping
            await asyncio.sleep(
                0.1)  # set a large one to reduce the workload of cpu
