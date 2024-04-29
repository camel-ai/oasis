import asyncio
import threading

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
        self.threads = []
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
        self.threads.append(threading.Thread(target=func, kwargs=params))  

    def regester_threads(self, funcs):
        pass

    def regester_process(self, func, **params):
        pass

    def regester_processes(self, funcs):
        pass

    async def run(self):
        if len(self.threads) != 0:
            for thread in self.threads:
                thread.start()
        if len(self.tasks) != 0:
            await asyncio.gather(*self.tasks)


