import asyncio
from communication.channel import Channel, Management


class A:
    def __init__(self, name):
        self.count = 0
        self.name = name

    async def task(self, channel):
         while True:
            self.count += 1
            await channel.send_to("B", f"{self.name} says hello {self.count}")
            message = await channel.receive_from("B")
            print(f"A received: {message}")

    async def task2(self, channel):
        while True:
            self.count += 1
            await channel.send_to("C", f"{self.name} says hello {self.count}")
            message = await channel.receive_from("C")
            print(f"A received: {message}")
            

class B:
    def __init__(self, name):
        self.count = 0
        self.name = name

    async def task(self, chanel):
        while True:
            message = await chanel.receive_from("A")
            print(f"B received: {message}")
            self.count += 1
            await chanel.send_to("A", f"{self.name} says hello {self.count}")
        

class C:
    def __init__(self, name):
        self.count = 0
        self.name = name

    async def task(self, chanel):
        while True:
            message = await chanel.receive_from("A")
            print(f"C received: {message}")
            self.count += 1
            await chanel.send_to("A", f"{self.name} says hello {self.count}")


async def main():
    manager = Management()
    a = A("A")
    b = B("B")
    c = C("C")
    chanel_a = manager.regester_channel(a.name)
    chanel_b = manager.regester_channel(b.name)
    chanel_c = manager.regester_channel(c.name)
    manager.regester_task(a.task, chanel=chanel_a)
    manager.regester_task(a.task2, chanel=chanel_a)
    manager.regester_task(b.task, chanel=chanel_b)
    manager.regester_task(c.task, chanel=chanel_c)
    await manager.run()

if __name__ == "__main__":
    asyncio.run(main())