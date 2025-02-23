# channel.py
import asyncio
import uuid

class AsyncSafeDict:
    def __init__(self):
        self.dict = {}
        self.lock = asyncio.Lock()

    async def put(self, key, value):
        async with self.lock:
            self.dict[key] = value

    async def get(self, key, default=None):
        async with self.lock:
            return self.dict.get(key, default)

    async def pop(self, key, default=None):
        async with self.lock:
            return self.dict.pop(key, default)

    async def keys(self):
        async with self.lock:
            return list(self.dict.keys())

class Channel:
    def __init__(self):
        self.receive_queue = asyncio.Queue()  # Used to store received messages
        self.send_dict = AsyncSafeDict()

    async def receive_from(self):
        message = await self.receive_queue.get()
        return message

    async def send_to(self, message):
        # message_id is the first element of the message
        message_id = message[0]
        await self.send_dict.put(message_id, message)

    async def write_to_receive_queue(self, action_info, agent_id):
        message_id = str(uuid.uuid4())
        await self.receive_queue.put((message_id, action_info, agent_id))
        return message_id

    async def read_from_send_queue(self, message_id):
        while True:
            if message_id in await self.send_dict.keys():
                # Attempting to retrieve the message
                message = await self.send_dict.pop(message_id, None)
                if message:
                    return message  # Return the found message
            # Temporarily suspend to avoid tight looping
            await asyncio.sleep(0.1)  # Adjust as needed