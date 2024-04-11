import asyncio
import json
import random
from time import sleep


class User:
    def __init__(self, name, id):
        self.name: str = name
        self.id: int = id


class Platform:
    def __init__(self):
        self.users = {}
        self.count = 0

    def add_user(self, user: User):
        self.users[user.id] = user

    def remove_user(self, user: User):
        del self.users[user.id]

    async def recive(self, reader):
        while True:
            data = await reader.readline()
            data = data.decode()
            try:
                message = json.loads(data)
                id, content = message["id"], message["content"]
                id = int(id)
                if id in self.users:
                    user = self.users[id]
                    print("平台收到:", user.name, content)
                else:
                    self.add_user(User(name="Agent" + str(id), id=id))
                    print("New user created: ", "Agent" + str(id))
            except:
                pass
            await asyncio.sleep(0.01)

    async def send(self, writer):
        while True:
            ids = list(self.users.keys())
            if len(ids) > 1:
                self.count += 1
                receiver_id = random.choice(ids)
                data = {"id": receiver_id, "content": "平台第" + str(self.count) + "次发送信息."}
                message = json.dumps(data) + "\n"
                writer.write(message.encode())
                await writer.drain()
                print("平台第" + str(self.count) + "次发送信息.")
            await asyncio.sleep(0.02)


    async def handler(self, reader, writer):
        await asyncio.gather(self.recive(reader), self.send(writer))

            
    async def run(self, host='127.0.0.1', port=8000):
        while True:
            try:
                server = await asyncio.start_server(self.handler, host, port)

                addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
                print(f'Serving on {addrs}')

                async with server:
                    await server.serve_forever()
               
            except:
                print("Waiting for connection...")
                sleep(3)
                continue
