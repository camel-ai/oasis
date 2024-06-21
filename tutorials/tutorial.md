# Tutorial

## Quickstart

### Step 1: Create and Activate a Virtual Environment

Setup virtual env(by py venv or conda)

For Linux
```bash
conda create --name camel_ss python=3.10
conda activate camel_ss
```

For win:

```bash
python -m venv camel_ss       # Create virtual environment
camel_ss\Scripts\activate  # Activate virtual environment
```

For macOS:

```bash
python -m venv camel_ss       # Create virtual environment
source camel_ss/bin/activate  # Activate virtual environment
```




### Step 2: Install Necessary Packages

```bash
pip install --upgrade pip     # Upgrade pip

# current camel do not support llama3, I make a few change and fix a few bugs.
git clone https://github.com/zhangzaibin/camel-llama3.git
cd camel-llama3
pip install -e .

# install packages
pip install -r requirements.txt 
pip install prance
pip install openapi-spec-validator
pip install slack_sdk
```

### Step 3: Set Vllm -> Set Config files(Multi-GPUs, Models, agent clusters & data)
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




### Step 4: Run the Main Program

```bash
bash run.sh
```

## to-dos
- [ ] to support functioncall & upgrade camel-ai to 0.1.5.1
- [ ] to support Multi-GPUs & Multi-LLMs @zhiyu @yuxian
- [ ] integrate argparser & config files @zhiyu @ziyi
- [ ] develop a simple GUI as a program entry point and for visualizing the running process and results (preliminary analysis)



