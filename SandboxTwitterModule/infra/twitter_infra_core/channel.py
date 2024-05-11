import asyncio
import uuid


class Twitter_Channel:
    def __init__(self):
        # 初始化两个消息队列
        self.receive_queue = asyncio.Queue()  # 用于存储接收的消息
        self.send_queue = asyncio.Queue()     # 用于存储要发送的消息

    async def receive_from(self):
        # 从接收队列中获取一个消息并返回
        message = await self.receive_queue.get()
        return message

    async def send_to(self, message):
        # 将消息添加到发送队列中
        print(message)
        await self.send_queue.put(message)

    async def write_to_receive_queue(self, action_info):
        # 生成唯一的message_id
        message_id = str(uuid.uuid4())
        # 向receive_queue写入消息
        await self.receive_queue.put((message_id, action_info))
        return message_id

    async def read_from_send_queue(self, message_id):
        # 循环检查send_queue以寻找特定message_id的消息
        while True:
            if not self.send_queue.empty():
                current_message = await self.send_queue.get()
                if current_message[0] == message_id:
                    return current_message
                else:
                    # 如果不是所需的message_id，将消息放回队列（可能需要更优雅的处理方式）
                    await self.send_queue.put(current_message)
            await asyncio.sleep(0.1)  # 简单的防止紧密循环，实际应用中可能需要更复杂的逻辑
