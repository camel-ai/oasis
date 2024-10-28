import asyncio
import logging
import threading

from oasis.inference.inference_thread import InferenceThread, SharedMemory

inference_log = logging.getLogger(name="inference")
inference_log.setLevel("DEBUG")

file_handler = logging.FileHandler("inference.log")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
inference_log.addHandler(file_handler)


class InferencerManager:
    r"""InferencerManager class to manage multiple inference threads."""

    def __init__(
        self,
        channel,
        model_type,
        model_path,
        stop_tokens,
        server_url,
    ):
        self.count = 0
        self.channel = channel
        self.threads = []
        for url in server_url:
            host = url["host"]
            for port in url["ports"]:
                _url = f"http://{host}:{port}/v1"
                shared_memory = SharedMemory()
                thread = InferenceThread(
                    model_path=model_path,
                    server_url=_url,
                    stop_tokens=stop_tokens,
                    model_type=model_type,
                    temperature=0.0,
                    shared_memory=shared_memory,
                )
                self.threads.append(thread)

    async def run(self):
        for thread in self.threads:
            thread_ = threading.Thread(target=thread.run)
            thread_.start()
        while True:
            for thread in self.threads:
                if thread.shared_memory.Done:
                    await self.channel.send_to(
                        (thread.shared_memory.Message_ID,
                         thread.shared_memory.Response))
                    thread.shared_memory.Done = False
                    thread.shared_memory.Busy = False
                    thread.shared_memory.Working = False

                if not thread.shared_memory.Busy:
                    if self.channel.receive_queue.empty():
                        continue
                    message = await self.channel.receive_from()
                    thread.shared_memory.Message_ID = message[0]
                    thread.shared_memory.Message = message[1]
                    thread.shared_memory.Busy = True
                    self.count += 1
                    inference_log.info(f"Message {self.count} received")
            await asyncio.sleep(0.0001)

    async def stop(self):
        for thread in self.threads:
            thread.alive = False
