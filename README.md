[![Python Version][python-image]][python-url]
[![PyTest Status][pytest-image]][pytest-url]
[![Star][star-image]][star-url]
[![Slack][slack-image]][slack-url]
[![Discord][discord-image]][discord-url]
[![Wechat][wechat-image]][wechat-url]
[![Twitter][twitter-image]][twitter-url]

______________________________________________________________________

# OASIS: Open Agents Social Interaction Simulations on One Million Agents

<p align="center">
  <img src='assets/logo.jpg' width=400>
</p>

<p align="center">
  <a href="https://github.com/camel-ai/camel#community">Community</a> |
  <a href="https://github.com/camel-ai/social-simulation/tree/main/scripts">Examples</a> |
  <a href="https://huggingface.co/datasets/oasis-agent/oasis-dataset">Dataset</a> |
  <a href="https://github.com/camel-ai/camel#citation">Citation</a> |
  <a href="https://github.com/camel-ai/camel#contributing-to-camel-">Contributing</a> |
  <a href="https://www.camel-ai.org/">CAMEL-AI</a>
</p>

## 📝 Overview

<p align="center">
  <img src='assets/intro-new-logo.png' width=900>
</p>
🏝️ OASIS is a scalable, open-source social media simulator that integrates large language models with rule-based agents to realistically mimic the behavior of up to one million users on platforms like Twitter and Reddit. It's designed to facilitate the study of complex social phenomena such as information spread, group polarization, and herd behavior, offering a versatile tool for exploring diverse social dynamics and user interactions in digital environments.

### Key Features

📈 Scalability: OASIS supports simulations of up to one million agents, enabling studies of social media dynamics at a scale comparable to real-world platforms.

📲 ️Dynamic Environments: Adapts to real-time changes in social networks and content, mirroring the fluid dynamics of platforms like Twitter and Reddit for authentic simulation experiences.

👍🏼 Diverse Action Spaces: Agents can perform 21 actions, such as following, commenting, and reposting, allowing for rich, multi-faceted interactions.

🔥 Integrated Recommendation Systems: Features interest-based and hot-score-based recommendation algorithms, simulating how users discover content and interact within social media platforms.

## 🔧 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/camel-ai/social-simulation.git

cd social-simulation
```

### Step 2: Create and Activate a Virtual Environment

Please choose one of the following methods to set up your environment. You only need to follow one of these methods.

- Option 1: Using Conda (Linux & macOS & windows)

```bash
conda create --name oasis python=3.10
conda activate oasis
```

- Option 2: Using venv (Linux & macOS)

```bash
python -m venv oasis-venv
source oasis-venv/bin/activate
```

- Option 3: Using venv (Windows)

```bash
python -m venv oasis-venv
oasis-venv\Scripts\activate
```

### Step 3: Install Necessary Packages

```bash
pip install --upgrade pip setuptools

pip install -e .  # This will install dependencies as specified in pyproject.toml
```

## 🏃Quickstart (For OpenAI Models)

### Step 1: Set Up Environment Variables

First, you need to add your OpenAI API key to the system's environment variables. You can obtain your OpenAI API key from [here](https://platform.openai.com/api-keys). Note that the method for doing this will vary depending on your operating system and the shell you are using.

- For Bash shell (Linux, macOS, Git Bash on Windows):\*\*

```bash
# Export your OpenAI API key
export OPENAI_API_KEY=<insert your OpenAI API key>
OPENAI_API_BASE_URL=<inert your OpenAI API BASE URL>  #(Should you utilize an OpenAI proxy service, kindly specify this)
```

- For Windows Command Prompt:\*\*

```cmd
REM export your OpenAI API key
set OPENAI_API_KEY=<insert your OpenAI API key>
set OPENAI_API_BASE_URL=<inert your OpenAI API BASE URL>  #(Should you utilize an OpenAI proxy service, kindly specify this)
```

- For Windows PowerShell:\*\*

```powershell
# Export your OpenAI API key
$env:OPENAI_API_KEY="<insert your OpenAI API key>"
$env:OPENAI_API_BASE_URL="<inert your OpenAI API BASE URL>"  #(Should you utilize an OpenAI proxy service, kindly specify this)
```

Replace `<insert your OpenAI API key>` with your actual OpenAI API key in each case. Make sure there are no spaces around the `=` sign.

### Step 2: Modify the Configuration File (Optional)

If adjustments to the settings are necessary, you can specify the parameters in the `scripts/reddit_gpt_example/gpt_example.yaml` file. Explanations for each parameter are provided in the comments within the YAML file.

To import your own user and post data, please refer to the JSON file format located in the `/data/reddit/` directory of this repository. Then, update the `user_path` and `pair_path` in the YAML file to point to your data files.

### Step 3: Run the Main Program

```bash
# For Reddit
python scripts/reddit_gpt_example/reddit_simulation_gpt.py --config_path scripts/reddit_gpt_example/gpt_example.yaml
# For Twitter
```

Note: without modifying the Configuration File, running this script requires approximately 14 API requests to call gpt-4, and the cost incurred is minimal. (October 29, 2024)

## 📘 Comprehensive Guide (For Open Source Models)

We assume that users are conducting large-scale experiments on a Slurm workload manager cluster. Below, we provide the commands for running experiments with open-source models on the Slurm cluster. The steps for running these experiments on a local machine are similar.

### Step 1: Donwload Open Source Models

Taking the download of LLaMA-3 from Hugging Face as an example:

```bash
pip install huggingface_hub

huggingface-cli download --resume-download "meta-llama/Meta-Llama-3-8B-Instruct" --local-dir "YOUR_LOCAL_MODEL_DIRECTORY" --local-dir-use-symlinks False --resume-download --token "YOUR_HUGGING_FACE_TOKEN"
```

Note: Please replace "YOUR_LOCAL_MODEL_DIRECTORY" with your actual directory path where you wish to save the model and "YOUR_HUGGING_FACE_TOKEN" with your Hugging Face token. Obtain your token at https://huggingface.co/settings/tokens.

### Step 2: Request GPUs and Get Information

Ensure that the GPU memory you're requesting is sufficient for deploying the open-source model you've downloaded. Taking the application for an A100 GPU as an example:

```bash
salloc --ntasks=1 --mem=100G --time=11:00:00 --gres=gpu:a100:1
```

Next, obtain and record the information of that node. Please ensure that the IP address of the GPU server can be accessed by your network, such as within the school's internet.

```bash
srun --ntasks=1 --mem=100G --time=11:00:00 --gres=gpu:a100:1 bash -c 'ifconfig -a'
"""
Example output:
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
       inet 10.109.1.8  netmask 255.255.255.0  broadcast 192.168.1.255
       ether 02:42:ac:11:00:02  txqueuelen 0  (Ethernet)
       RX packets 100  bytes 123456 (123.4 KB)
       RX errors 0  dropped 0  overruns 0  frame 0
       TX packets 100  bytes 654321 (654.3 KB)
       TX errors 0  dropped 0 overruns 0  carrier 0  collisions
"""

srun --ntasks=1 --mem=100G --time=11:00:00 --gres=gpu:a100:1 bash -c 'echo $CUDA_VISIBLE_DEVICES'
"""
Example output: 0
"""
```

Document the IP address associated with the eth0 interface, which, in this example, is `10.109.1.8`. Additionally, note the identifier of the available GPU, which in this case is `0`.

### Step 3: Deploying vLLM

Based on the IP address and GPU identifier obtained from step 2, and the model path and name from step 1, modify the `hosts`, `gpus` variables, and the `'YOUR_LOCAL_MODEL_DIRECTORY'`, `'YOUR_LOCAL_MODEL_NAME strings'` in the `deploy.py` file. For example:

```python
if __name__ == "__main__":
    host = "10.109.1.8"  # input your IP address
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
    gpus = [0]  # input your $CUDA_VISIBLE_DEVICES

    all_ports = [port for i in gpus for port in ports[i]]
    print("All ports: ", all_ports, '\n\n')

    t = None
    for i in range(3):
        for j, gpu in enumerate(gpus):
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} python -m "
                f"vllm.entrypoints.openai.api_server --model "
                f"'YOUR_LOCAL_MODEL_DIRECTORY' "  # input the path where you downloaded your model
                f"--served-model-name 'YOUR_LOCAL_MODEL_NAME' "  # input the name of the model you downloaded
                f"--host {host} --port {ports[j][i]} --gpu-memory-utilization "
                f"0.3 --disable-log-stats")
            t = threading.Thread(target=subprocess.run,
                                 args=(cmd, ),
                                 kwargs={"shell": True},
                                 daemon=True)
            t.start()
        check_port_open(host, ports[0][i])
    t.join()
```

Next, run the `deploy.py` script. Then you will see an output, which contains a list of all ports.

```bash
srun --ntasks=1 --time=11:00:00 --gres=gpu:a100:1 bash -c 'python deploy.py'
"""
Example output:
All ports:  [8002, 8003, 8005]

More other output about vllm...
"""
```

### Step 4: Modify the Configuration File

Before the simulation begins, you need to enter your model name, model path, host, and ports into the corresponding yaml file in the experiment script such as `scripts\reddit_simulation_align_with_human\business_3600.yaml`. An example of what to write is:

```yaml
inference:
  model_type: 'YOUR_LOCAL_MODEL_NAME'  # input the name of the model you downloaded (eg. 'llama-3')
  model_path: 'YOUR_LOCAL_MODEL_DIRECTORY'  # input the path where you downloaded your model
  stop_tokens: ["<|eot_id|>", "<|end_of_text|>"]
  server_url:
    - host: "10.109.1.8"
      ports: [8002, 8003, 8005]  # Input the list of all ports obtained in step 3
```

Additionally, you can modify other settings related to data and experimental details in the yaml file. For instructions on this part, refer to `scripts\reddit_gpt_example\gpt_example.yaml`.

### Step 5: Run the Main Program

You need to open a new terminal and then run:

```bash
# For Reddit

# Align with human
python scripts/reddit_simulation_align_with_human/reddit_simulation_align_with_human.py --config_path scripts/reddit_simulation_align_with_human/business_3600.yaml

# Agent's reaction to counterfactual content
python scripts/reddit_simulation_counterfactual/reddit_simulation_counterfactual.py --config_path scripts/reddit_simulation_counterfactual/control_100.yaml

# For Twitter(X)

# Information spreading
# one case in align_with_real_world, The ‘user_char’ field in the dataset we have open-sourced has been replaced with  ‘description’ to ensure privacy protection.
python scripts/twitter_simulation/twitter_simulation_large.py --config_path scripts/twitter_simulation/align_with_real_world/yaml_200/sub1/False_Business_0.yaml

# Group Polarization
python scripts/twitter_simulation/group_polarization/twitter_simulation_group_polar.py --config_path scripts/twitter_simulation/group_polarization/group_polarization.yaml

# For One Million Simulation
python scripts/twitter_simulation_1M_agents/twitter_simulation_1m.py --config_path scripts/twitter_simulation_1M_agents/twitter_1m.yaml
```

## 💡Tips

### For Twitter Simluation:

- Customizing temporal feature

When simulating on generated users, you can customizing temporal feature in `social_simulation/social_agent/agents_generator.py` by modifying `profile['other_info']['active_threshold']`. For example, you can set it to all 1 if you believe that the generated users should be active the entire time.

### For Reddit Simluation:

- Reddit recommendation system

The Reddit recommendation system is highly time-sensitive. Currently, one time step in the `reddit_simulation_xxx.py`simulates approximately two hours in the agent world, so essentially, new posts are recommended at every time step. To ensure that all posts made by controllable users can be seen by other agents, it is recommended that `the number of agents` × `activate_prob` > `max_rec_post_len` > `round_post_num`.

## 🚢 More Tutorials

To discover how to create profiles for large-scale users, as well as how to visualize and analyze social simulation data once your experiment concludes, please refer to [More Tutorials](tutorials/tutorial.md) for detailed guidance.

## 📢 News

<!-- - Public release of our dataset on Hugging Face (November 05, 2024) -->

- Initial release of OASIS github repository (November 15, 2024)

## 🔗 Citation

To be supplemented after the release on arXiv.

## 🙌 Acknowledgment

We would like to thank Douglas for designing the logo of our project.

## 🖺 License

The source code is licensed under Apache 2.0.

## 🥂 Contributing to OASIS 🏝️

We greatly appreciate your interest in contributing to our open-source initiative. To ensure a smooth collaboration and the success of contributions, we adhere to a set of contributing guidelines similar to those established by CAMEL. For a comprehensive understanding of the steps involved in contributing to our project, please refer to the CAMEL contributing guidelines [here](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md). 🤝🚀

An essential part of contributing involves not only submitting new features with accompanying tests (and, ideally, examples) but also ensuring that these contributions pass our automated pytest suite. This approach helps us maintain the project's quality and reliability by verifying compatibility and functionality.

## 📬 Contact

If you're keen on exploring new research opportunities or discoveries with our platform and wish to dive deeper or suggest new features, we're here to talk. Feel free to get in touch for more details at camel.ai.team@gmail.com.

[discord-image]: https://img.shields.io/badge/Discord-CAMEL--AI-7289da?logo=discord&logoColor=white&color=7289da
[discord-url]: https://discord.gg/CNcNpquyDc
[pytest-image]: https://github.com/camel-ai/camel/actions/workflows/pytest_package.yml/badge.svg
[pytest-url]: https://github.com/camel-ai/social-simulation/actions/workflows/pytest_package.yml
[python-image]: https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg
[python-url]: https://docs.python.org/3.10/
[slack-image]: https://img.shields.io/badge/Slack-CAMEL--AI-blueviolet?logo=slack
[slack-url]: https://join.slack.com/t/camel-kwr1314/shared_invite/zt-1vy8u9lbo-ZQmhIAyWSEfSwLCl2r2eKA
[star-image]: https://img.shields.io/github/stars/camel-ai/social-simulation?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/camel-ai/social-simulation/stargazers
[twitter-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social&color=brightgreen&logo=twitter
[twitter-url]: https://twitter.com/CamelAIOrg
[wechat-image]: https://img.shields.io/badge/WeChat-CamelAIOrg-brightgreen?logo=wechat&logoColor=white
[wechat-url]: https://ghli.org/camel/wechat.png
