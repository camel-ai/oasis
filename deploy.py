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
    host = "0.0.0.0"  # 平台 IP 地址
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
    gpus = [0, 1, 2]  # GPU 设备编号

    ports = [port for row in ports for port in row]  # 转换为一维数组
    '''
    原脚本中会在一个GPU上部署3个API服务并对应3个端口，所以ports定义成了二维数组
    因为我们平台的GPU显存不够大，需要改成一个GPU上部署一个API服务，对应一个端口
    所以这里把ports数组按行转换为一维
    '''

    # all_ports = [port for i in gpus for port in ports[i]]
    all_ports = ports[:len(gpus)]  # 输出将会使用的端口
    print("All ports: ", all_ports, '\n\n')

    t = None
    for i, gpu in enumerate(gpus):
        cmd = (
            f"TF_ENABLE_DEPRECATION_WARNINGS=1 "  # 启用TensorFlow的弃用功能警告
            f"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 "  # 允许模型支持超过默认最大长度的上下文
            f"VLLM_ENFORCE_CUDA_GRAPH=1 "  # 强制使用CUDA Graph优化推理流程
            f"CUDA_VISIBLE_DEVICES={gpu} "  # 指定 GPU
            f"python3 -m "
            f"vllm.entrypoints.openai.api_server "
            f"--model '/models/Qwen2.5-7B-Instruct' "  # 模型路径
            f"--served-model-name 'Qwen2.5-7B' "  # 对外暴露的模型名称
            # f"--tensor-parallel-size 8 "  # GPU 并行数，单卡部署需关闭
            f"--host {host} --port {ports[i]} "  # 绑定IP和端口
            f"--gpu-memory-utilization 0.9 "  # 设置GPU显存利用率阈值
            f"--disable-log-stats")  # 关闭性能统计日志
        t = threading.Thread(target=subprocess.run,
                                args=(cmd, ),
                                kwargs={"shell": True},
                                daemon=True)
        t.start()
        check_port_open(host, ports[i])

    t.join()
