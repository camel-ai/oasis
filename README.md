# Social Simulation

## 💪TODOs

- [ ] More evaluation metric
- [ ] Agent interaction visualization

## 🚴Tips

### Large-scale agents simulation

Large-scale agent simulations contain too many actions; therefore, we need to increase `self.rec_update_time` in `twitter/twitter.py` to trade-off. Approximately set it to **20** when simulating **80** agents. So when you simulate thousands of agents, set a larger `self.rec_update_time`

### Time feature simulation

Simulating time features requires crawling a large number of historical tweets from Twitter. Therefore, if you are simulating other scenarios and large-scale users, we recommend using the pre-defined activation level instead of time feature v1.2.

## 🏃Quickstart

### Step 1: Create and Activate a Virtual Environment

For Linux

```bash
conda create --name camel_ss python=3.10
conda activate camel_ss
# Or
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Necessary Packages

```bash
pip install --upgrade pip setuptools
pip install -e .
```

### Step 3: Setup vLLM

Deploying open source models with vLLM, using llama3 as an example:

- Read the vllm doc https://docs.vllm.ai/en/stable/ (Optional)
- Download llama3-8b-instruct weights to local folder.
- Prepare a linux-gpu enviornment. (Recommended VRAM >= 24G)
- Please ensure that the IP address of the GPU server can be accessed by your network, such as within the school's internet.

```bash
ifconfig -a # get your ip address
python -m vllm.entrypoints.openai.api_server --model /your/path/to/llama3-8b-instruct # get your port number
```

### Step 4: Modify the Configuration File

- Edit the yaml file in `scripts/` to configure your settings.
- Ensure that all the parameters in the yaml file are correctly set according to your requirements.

### Step 5: Run the Main Program

```bash
python scripts/twitter_simulation.py --config_path scripts/twitter_openai.yaml
# or
python scripts/twitter_simulation.py --config_path scripts/twitter_opensource.yaml
# or
python scripts/reddit_simulation.py --config_path scripts/reddit.yaml
# or
python scripts/twitter_simulation_large.py --config_path scripts/twitter_simulation_large.yaml
```
