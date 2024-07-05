# Social Simulation

## 💪TODOs
- [ ] More evaluation metric
- [ ] Agent interaction visualization

## 🚴Tips
### large-scale agents simulation 
Large-scale agent simulations contain too many actions; therefore, we need to increase ``self.rec_update_time`` in ``twitter/twitter.py`` to trade-off. Approximately set it to **20** when simulating **80** agents. So when you simulate thousands of agents, set a larger ``self.rec_update_time``

### time feature simulation
Simulating time features requires crawling a large number of historical tweets from Twitter. Therefore, if you are simulating other scenarios and large-scale users, we recommend using the pre-defined activation level instead of time feature v1.2.


## 🏃Quickstart

### Step 1: Create and Activate a Virtual Environment


For Linux
```bash
conda create --name camel_ss python=3.10
conda activate camel_ss
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

### Step 3: Set Vllm
Deploying some open source models with vllm, using llama3 as an example.

- read the vllm doc https://docs.vllm.ai/en/stable/ (Optional)
- download llama3-8b-instruct weights to local folder.
- prepare a linux-gpu enviornment. (Recommended VRAM >= 24G)
- Please ensure that the IP address of the GPU server can be accessed by your network, such as within the school's internet.

```bash
ifconfig -a # get your ip address
python -m vllm.entrypoints.openai.api_server --model /your/path/to/llama3-8b-instruct # get your port number
```
<!-- ### Step 4: Parepare Twhin-bert
- [optional] prepare the gpu environment (highly recommended) (we test with CPU, and it also performs ok.)
- download the twhin weights from https://huggingface.co/Twitter/twhin-bert-base
- change the path in line 20, 21 of twitter/recsys.py -->

### Step 4: Modify the Configuration File

- Edit the yaml file in `scripts/` to configure your settings.
- Ensure that all the parameters in the yaml file are correctly set according to your requirements.

### Step 5: Run the Main Program

```bash
python scripts/twitter_simulation.py --config_path scripts/twitter.yaml
# or 
python scripts/reddit_simulation.py --config_path scripts/reddit.yaml
```
<!-- 
### Step 7: Run all the experiments in batch and visualize the results
```bash
bash run_all.sh
``` -->




<!-- 
## Quickstart

### Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### Installation

```bash
pip install --upgrade pip
pip install -e .
```

## Script

### Run Twitter Script

```bash
python scripts/twitter_simulation.py
# or
python scripts/twitter_simulation.py --config_path scripts/twitter.yaml
# You can also change `controllable_user` to `true` in the yaml file to run with controllable user
```

### Run Reddit Script

```bash
python3 scripts/reddit_simulation.py
# or
python scripts/reddit_simulation.py --config_path scripts/reddit.yaml
```

## Test

```bash
pytest test
``` -->
