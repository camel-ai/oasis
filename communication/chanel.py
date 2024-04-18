import asyncio

class Channel:
    def __init__(self, name):
        self.name = name
        self.flags = {}
        self.input_queues = {}
        self.output_queues = {}

    def connect(self, chanel):
        self.input_queues[chanel.name] = asyncio.Queue()
        chanel.input_queues[self.name] = asyncio.Queue()
        self.output_queues[chanel.name] = chanel.input_queues[self.name]
        chanel.output_queues[self.name] = self.input_queues[chanel.name]

    async def receive_from(self, name):
        return await self.input_queues[name].get()
    
    async def send_to(self, name, message):
        await self.output_queues[name].put(message)
    
    def empty(self, name):
        return self.input_queues[name].empty()
    

class Management:
    def __init__(self):
        self.tasks = []
        self.chanels = {}

    def regester_chanel(self, name):
        if name not in self.chanels:
            channel = Channel(name)
            self.chanels[name] = chanel
            for n in self.chanels:
                if n != name:
                    chanel.connect(self.chanels[n])
        else:
            chanel = self.chanels[name]
        
        return chanel

    def regester_task(self, func, **params):
        self.tasks.append(func(**params))

    def regester_tasks(self, funcs):
        for func in funcs:
            self.tasks.append(func)
    
    def regester_thread(self, func, **params):
        pass

    def regester_threads(self, funcs):
        pass

    def regester_process(self, func, **params):
        pass

    def regester_processes(self, funcs):
        pass

    async def run(self):
        await asyncio.gather(*self.tasks)



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