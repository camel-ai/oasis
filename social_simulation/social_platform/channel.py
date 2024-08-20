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
        self.receive_queue = asyncio.Queue()  # 用于存储接收的消息
        self.send_dict = AsyncSafeDict()  # 使用异步安全字典存储要发送的消息

    async def receive_from(self):
        message = await self.receive_queue.get()
        return message

    async def send_to(self, message):
        # message_id 是消息的第一个元素
        message_id = message[0]
        await self.send_dict.put(message_id, message)

    async def write_to_receive_queue(self, action_info):
        message_id = str(uuid.uuid4())
        await self.receive_queue.put((message_id, action_info))
        return message_id

    async def read_from_send_queue(self, message_id):
        while True:
            if message_id in await self.send_dict.keys():
                # 尝试获取消息
                message = await self.send_dict.pop(message_id, None)
                if message:
                    return message  # 返回找到的消息

            await asyncio.sleep(0.001)  # 暂时挂起，避免紧密循环
