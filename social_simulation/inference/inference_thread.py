from time import sleep
import logging
from camel.configs import ChatGPTConfig, OpenSourceConfig

from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelType

thread_log = logging.getLogger(name='inference.thread')
thread_log.setLevel('DEBUG')

class SharedMemory:
    Message_ID = 0
    Message = None
    Response = None
    Busy = False
    Working = False
    Done = False

class InferenceThread:
    def __init__(
        self,
        model_path:
        str = "/mnt/hwfile/trustai/models/Meta-Llama-3-8B-Instruct",  # noqa
        server_url: str = "http://10.140.0.144:8000/v1",
        stop_tokens: list[str] = None,
        model_type: ModelType = ModelType.LLAMA_3,
        temperature: float = 0.0,
        shared_memory: SharedMemory = None
    ):
        self.count = 0
        self.server_url = server_url
        self.model_type = model_type
        api_params = ChatGPTConfig(
            temperature=temperature,
            stop=stop_tokens,
        )
        model_config = OpenSourceConfig(
            model_path=model_path,
            server_url=server_url,
            api_params=api_params,
        )
        self.model_backend: BaseModelBackend = ModelFactory.create(
            model_type, model_config.__dict__)
        if shared_memory is None:
            self.shared_memory = SharedMemory()
        else:
            self.shared_memory = shared_memory

    def run(self):
        while True:
            if self.shared_memory.Busy and not self.shared_memory.Working:
                self.shared_memory.Working = True
                response = self.model_backend.run(
                    self.shared_memory.Message)
                self.shared_memory.Response = response.choices[0].message.content
                self.shared_memory.Done = True
                self.count += 1
                thread_log.info(f"Thread {self.server_url}: {self.count} finished.")

            sleep(0.0001)