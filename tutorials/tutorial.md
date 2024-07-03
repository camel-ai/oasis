# Tutorial

## Setting Up the Environment

### Step 1: Setup virtual env

(by py venv or conda)

For Linux
```bash
conda create --name camel_ss python=3.10 # or 3.11
conda activate camel_ss
```

For win:

```bash
python3.11 -m venv camel_ss       # Create virtual environment (by python3.10 or 3.11)
camel_ss\Scripts\activate  # Activate virtual environment
```

For macOS:

```bash
python3.11 -m venv camel_ss       # Create virtual environment
source camel_ss/bin/activate  # Activate virtual environment
```


### Step 2. Install requirements.txt

```bash
python3.11 -m pip install -r requirements.txt 
```

## Setting up vllm-config or config.yaml files

### vllm-config(currently, will be obsolete soon)
- read the vllm doc https://docs.vllm.ai/en/stable/ (Optional)
- download llama3-8b-instruct weights to local folder.
- prepare a linux-gpu enviornment. (Recommended VRAM >= 24G)
- Please ensure that the IP address of the GPU server can be accessed by your network, such as within the school's internet.

```bash
ifconfig -a # get your ip address
python -m vllm.entrypoints.openai.api_server --model /your/path/to/llama3-8b-instruct # get your port number
export LLAMA3_SERVER_URL="http://{your ip address}:{your port number}/v1" # eg, http://10.160.2.154:8000/v1
export LLAMA3_MODEL_PATH="/your/path/to/llama3-8b-instruct"
```

### config.yaml(will be integrated well soon)

#### To set up experiment data and test data
```python
class Config(Singleton):
    ...

    test_db_filepath = "your/test/data"
    user_data_filepath = "your/experiment/data"
    host = 'localhost'
    download_dir = '/mnt/workspace/.cache/modelscope/hub/'
```

#### To set up multi-LLMs & multi-GPUs to run simulation
```python
    single_model_single_instance = {
        'flag': False,
        'model_num': 1,
        'model_type': ModelType.LLAMA_2,
        'model_path': 'llama2',
        'port': 8000,
    }

    single_model_multi_instance = {
        'flag': True,
        'model_type': ModelType.LLAMA_3,
        'model_name': 'LLM-Research/Meta-Llama-3-8B-Instruct',
        'model_path': '/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct',
        'ports': [8000],
    }

    multi_model_single_instance = {
        'flag': False,
        'models': {
            'llama3': {
                'model_type': ModelType.LLAMA_3,
                'model_name': 'LLM-Research/Meta-Llama-3-8B-Instruct',
                'model_path': '/mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct',
                'port': 8000,
            },
            'qwen2': {
                'model_type': ModelType.QWEN_2,
                'model_name': 'qwen/Qwen2-7B-Instruct',
                'model_path': '/mnt/workspace/.cache/modelscope/hub/qwen/Qwen2-7B-Instruct',
                'port': 8001,
            },
            'glm4': {
                'model_type': ModelType.GLM_4,
                'model_name': 'ZhipuAI/glm-4-9b-chat',
                'model_path': '/mnt/workspace/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat',
                'port': 8002,
            },
        }
    }
```

#### To set up the three-stage recommender system 
with all components being pluggable to accommodate experimental needs 
(will be integrated well soon)
[recsys-design](./static/recsys.jpg)

#### To set up sandbox environment tailored to your experimental setup
sandbox-env: various agent actions & corresponding APIs supported by infra
[multiple-actions](./static/mutliple-actions.png)


## Run the Main Program


```bash
bash run.sh
```
coming soon:
A GUI(graphical user interface) to display simulations


## to-dos
- [x] to support functioncall & upgrade camel-ai to 0.1.5.1
- [ ] to support Multi-GPUs & Multi-LLMs @zhiyu @yuxian
- [ ] integrate argparser & config files @zhiyu @ziyi
- [ ] develop a simple GUI as a program entry point and for visualizing the running process and results (preliminary analysis)
- [ ] Develop a GUI(graphical user interface) to display simulations



