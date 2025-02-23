# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import logging
from time import sleep

from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType

thread_log = logging.getLogger(name="inference.thread")
thread_log.setLevel("DEBUG")


class SharedMemory:
    Message_ID = 0
    Message = None
    Agent_ID = 0
    Response = None
    Busy = False
    Working = False
    Done = False


class InferenceThread:

    def __init__(
        self,
        model_path: str = "models/Meta-Llama-3-8B-Instruct",  # noqa
        server_url: str = "http://10.140.0.144:8000/v1",
        stop_tokens: list[str] = None,
        model_platform_type: ModelPlatformType = ModelPlatformType.VLLM,
        model_type: str = "llama-3",
        temperature: float = 0.5,
        shared_memory: SharedMemory = None,
    ):
        self.alive = True
        self.count = 0
        self.server_url = server_url
        # print('model_type in InferenceThread:', model_type)
        self.model_type = model_type

        # print('server_url:', server_url)
        print("self.model_type:", self.model_type)
        self.model_backend: BaseModelBackend = ModelFactory.create(
            model_platform=model_platform_type,
            model_type=self.model_type,
            model_config_dict={
                "temperature": temperature,
                "stop": stop_tokens
            },
            url="vllm",
            api_key=server_url,
            # because of CAMEL bugs here, will fix when CAMEL upgrade.
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
                    self.shared_memory.Response = response.choices[
                        0].message.content
                except Exception as e:
                    print("Receive Response Exception:", str(e))
                    self.shared_memory.Response = "No response."
                self.shared_memory.Done = True
                self.count += 1
                thread_log.info(
                    f"Thread {self.server_url}: {self.count} finished.")

            sleep(0.1)
