import subprocess, requests, time, threading


def check_port_open(host, port):
    while True:
            url= f'http://{host}:{port}/health'
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    break
                else:
                    time.sleep(0.3)
            except:
                time.sleep(0.3)


if __name__ == '__main__':
    host = '10.140.1.173'
    ports = [[8021, 8001, 8027], [8003, 8028, 8025], [8006, 8007, 8008], [8009, 8023, 8011], [8012, 8013, 8014], [8015, 8016, 8017], [8018, 8019, 8020]]
    gpus = [1, 2, 5, 7]

    t = None
    for i in range(3):
        for j in range(len(gpus)):
            t = threading.Thread(target=subprocess.run, args=(f"CUDA_VISIBLE_DEVICES={gpus[j]} python -m vllm.entrypoints.openai.api_server --model {'LLM-Research/Meta-Llama-3-8B-Instruct'}  --host {host}  --port {ports[j][i]}  --gpu-memory-utilization 0.3  --disable-log-stats",), kwargs={'shell':True}, daemon=True)
            t.start()
        check_port_open(host, ports[0][i])

    t.join()