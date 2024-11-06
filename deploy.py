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
import subprocess
import threading
import time

import requests


def check_port_open(host, port):
    while True:
        url = f"http://{host}:{port}/health"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
            else:
                time.sleep(0.3)
        except Exception:
            time.sleep(0.3)


if __name__ == "__main__":
    host = "10.109.1.8"
    ports = [
        [8002, 8003, 8005],
        [8006, 8007, 8008],
        [8011, 8009, 8010],
        [8014, 8012, 8013],
        [8017, 8015, 8016],
        [8020, 8018, 8019],
        [8021, 8022, 8023],
        [8024, 8025, 8026],
    ]
    gpus = [0]

    all_ports = [port for i in gpus for port in ports[i]]
    print("All ports: ", all_ports, '\n\n')

    t = None
    for i in range(3):
        for j, gpu in enumerate(gpus):
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python -m "
                f"vllm.entrypoints.openai.api_server --model "
                f"'/ibex/user/yangz0h/open_source_llm/llama-3' "
                f"--served-model-name 'llama-3' "
                f"--host {host} --port {ports[j][i]} --gpu-memory-utilization "
                f"0.3 --disable-log-stats")
            t = threading.Thread(target=subprocess.run,
                                 args=(cmd, ),
                                 kwargs={"shell": True},
                                 daemon=True)
            t.start()
        check_port_open(host, ports[0][i])

    t.join()
