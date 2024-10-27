import subprocess
import threading
import time

import requests


def check_port_open(host, port):
    while True:
        url = f'http://{host}:{port}/health'
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
            else:
                time.sleep(0.3)
        except:
            time.sleep(0.3)


if __name__ == '__main__':
    host = '10.109.1.8'
    ports = [[8002, 8003, 8005], [8006, 8007, 8008], [8011, 8009, 8010],
             [8014, 8012, 8013], [8017, 8015, 8016], [8020, 8018, 8019],
             [8021, 8022, 8023], [8024, 8025, 8026]]
    gpus = [0]

    #print('check\n')
    #check_port_open('10.140.0.184', 8002)
    #print('\nport 8002 is open\n')
    t = None
    for i in range(3):
        for j in range(len(gpus)):
            t = threading.Thread(
                target=subprocess.run,
                args=
                (f"CUDA_VISIBLE_DEVICES={gpus[j]} python -m vllm.entrypoints.openai.api_server --model {'/ibex/user/yangz0h/open_source_llm/llama-3'} --served-model-name 'llama-3' --host {host}  --port {ports[j][i]}  --gpu-memory-utilization 0.3  --disable-log-stats",
                 ),
                kwargs={'shell': True},
                daemon=True)
            t.start()
        check_port_open(host, ports[0][i])

    t.join()
