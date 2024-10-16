from time import sleep
import logging
from camel.configs import ChatGPTConfig, OpenSourceConfig

from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType, ModelType

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
        model_platform_type: ModelPlatformType = ModelPlatformType.OPEN_SOURCE,
        model_type: ModelType = ModelType.LLAMA_3,
        temperature: float = 0.5,
        shared_memory: SharedMemory = None
    ):
        self.alive = True
        self.count = 0
        self.server_url = server_url
        self.model_type = model_type
        # api_params = ChatGPTConfig(
        #     temperature=temperature,
        #     stop=stop_tokens,
        # )
        # model_config = OpenSourceConfig(
        #     model_path=model_path,
        #     server_url=server_url,
        #     api_params=api_params,
        # )
        # model_config_dict = model_config.as_dict()
        # print('model_config_dict:', model_config_dict)
        # del model_config_dict['api_params']['tools']
        # del model_config_dict['api_params']['tool_choice']
        # self.model_backend: BaseModelBackend = ModelFactory.create(
        #     model_platform=model_platform_type, 
        #     model_type=model_type, 
        #     model_config_dict=model_config_dict,
        # )
        # print('server_url:', server_url)
        self.model_backend: BaseModelBackend = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM, 
            model_type= "llama-3", 
            model_config_dict={"temperature": 0.0},
            url = 'vllm',
            api_key = server_url # because of CAMEL bug here, will fix when CAMEL upgrade.
        )
        # print('self.model_backend._url:', self.model_backend._url)
        if shared_memory is None:
            self.shared_memory = SharedMemory()
        else:
            self.shared_memory = shared_memory

    def run(self):
        while self.alive:
            if self.shared_memory.Busy and not self.shared_memory.Working:
                self.shared_memory.Working = True
                try:
                    response = self.model_backend.run(
                        self.shared_memory.Message)
                    self.shared_memory.Response = response.choices[0].message.content
                except Exception as e:
                    print('Receive Response Exception:', str(e))
                    self.shared_memory.Response = "No response."
                self.shared_memory.Done = True
                self.count += 1
                thread_log.info(f"Thread {self.server_url}: {self.count} finished.")

            sleep(0.0001)