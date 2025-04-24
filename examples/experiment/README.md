# Welcome to 🏝️OASIS

This section is a legacy tutorial for reproducing the experimental results in the paper.

## 🏃Installation (For OpenAI Models)

### Step 1: Set Up Environment Variables

First, you need to add your OpenAI API key to the system's environment variables. You can obtain your OpenAI API key from [here](https://platform.openai.com/api-keys). Note that the method for doing this will vary depending on your operating system and the shell you are using.

- For Bash shell (Linux, macOS, Git Bash on Windows):\*\*

```bash
# Export your OpenAI API key
export OPENAI_API_KEY=<insert your OpenAI API key>
export OPENAI_API_BASE_URL=<insert your OpenAI API BASE URL>  #(Should you utilize an OpenAI proxy service, kindly specify this)
```

- For Windows Command Prompt:\*\*

```cmd
REM export your OpenAI API key
set OPENAI_API_KEY=<insert your OpenAI API key>
set OPENAI_API_BASE_URL=<insert your OpenAI API BASE URL>  #(Should you utilize an OpenAI proxy service, kindly specify this)
```

- For Windows PowerShell:\*\*

```powershell
# Export your OpenAI API key
$env:OPENAI_API_KEY="<insert your OpenAI API key>"
$env:OPENAI_API_BASE_URL="<insert your OpenAI API BASE URL>"  #(Should you utilize an OpenAI proxy service, kindly specify this)
```

Replace `<insert your OpenAI API key>` with your actual OpenAI API key in each case. Make sure there are no spaces around the `=` sign.

### Step 2: Modify the Configuration File (Optional)

If adjustments to the settings are necessary, you can specify the parameters in the `examples/experiment/reddit_gpt_example/gpt_example.yaml` file. Explanations for each parameter are provided in the comments within the YAML file.

To import your own user and post data, please refer to the JSON file format located in the `/data/reddit/` directory of this repository. Then, update the `user_path` and `pair_path` in the YAML file to point to your data files.

### Step 3: Run the Main Program

```bash
# For Reddit
python examples/experiment/reddit_gpt_example/reddit_simulation_gpt.py --config_path examples/experiment/reddit_gpt_example/gpt_example.yaml

# For Reddit with Electronic Mall
python examples/experiment/reddit_emall_demo/emall_simulation.py --config_path examples/experiment/reddit_emall_demo/emall.yaml

# For Twitter
python examples/experiment/twitter_gpt_example/twitter_simulation.py --config_path examples/experiment/twitter_gpt_example/gpt_example.yaml
```

Note: without modifying the Configuration File, running the Reddit script requires only 36 agents operating at an activation probability of 0.1 for 2 time steps, the entire process approximately requires 7.2 agent inferences, and approximately 14 API requests to call GPT-4. The Twitter script has about 111 agents operating at an activation probability of roughly 0.1 for 3 time steps, i.e., 33.3 agent inferences, using GPT-3.5-turbo. I hope this is a cost you can bear. For running larger scale agent simulations, it is recommended to read the next section on experimenting with open-source models.

<br>

## 📘 Comprehensive Guide (For Open Source Models)

We assume that users are conducting large-scale experiments on a Slurm workload manager cluster. Below, we provide the commands for running experiments with open-source models on the Slurm cluster. The steps for running these experiments on a local machine are similar.

### Step 1: Download Open Source Models

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

Before the simulation begins, you need to enter your model name, model path, host, and ports into the corresponding yaml file in the experiment script such as `examples\experiment\reddit_simulation_align_with_human\business_3600.yaml`. An example of what to write is:

```yaml
inference:
  model_type: 'YOUR_LOCAL_MODEL_NAME'  # input the name of the model you downloaded (eg. 'llama-3')
  model_path: 'YOUR_LOCAL_MODEL_DIRECTORY'  # input the path where you downloaded your model
  stop_tokens: ["<|eot_id|>", "<|end_of_text|>"]
  server_url:
    - host: "10.109.1.8"
      ports: [8002, 8003, 8005]  # Input the list of all ports obtained in step 3
```

Additionally, you can modify other settings related to data and experimental details in the yaml file. For instructions on this part, refer to `examples\experiment\reddit_gpt_example\gpt_example.yaml`.

### Step 5: Run the Main Program

You need to open a new terminal and then run:

```bash
# For Reddit

# Align with human
python examples/experiment/reddit_simulation_align_with_human/reddit_simulation_align_with_human.py --config_path examples/experiment/reddit_simulation_align_with_human/business_3600.yaml

# Agent's reaction to counterfactual content
python examples/experiment/reddit_simulation_counterfactual/reddit_simulation_counterfactual.py --config_path examples/experiment/reddit_simulation_counterfactual/control_100.yaml

# For Twitter(X)

# Information spreading
# one case in align_with_real_world, The 'user_char' field in the dataset we have open-sourced has been replaced with  'description' to ensure privacy protection.
python examples/experiment/twitter_simulation/twitter_simulation_large.py --config_path examples/experiment/twitter_simulation/align_with_real_world/yaml_200/sub1/False_Business_0.yaml

# Group Polarization
python examples/experiment/twitter_simulation/group_polarization/twitter_simulation_group_polar.py --config_path examples/experiment/twitter_simulation/group_polarization/group_polarization.yaml

# For One Million Simulation
python examples/experiment/twitter_simulation_1M_agents/twitter_simulation_1m.py --config_path examples/experiment/twitter_simulation_1M_agents/twitter_1m.yaml
```

<br>

## 💡Tips

### For Twitter Simulation:

- Customizing temporal feature

When simulating on generated users, you can customize the temporal feature in `social_simulation/social_agent/agents_generator.py` by modifying `profile['other_info']['active_threshold']`. For example, you can set it to all 1 if you believe that the generated users should be active the entire time.

### For Reddit Simulation:

- Reddit recommendation system

The Reddit recommendation system is highly time-sensitive. Currently, one time step in the `reddit_simulation_xxx.py`simulates approximately two hours in the agent world, so essentially, new posts are recommended at every time step. To ensure that all posts made by controllable users can be seen by other agents, it is recommended that `the number of agents` × `activate_prob` > `max_rec_post_len` > `round_post_num`.

<br>

## 🚢 More Tutorials

To discover how to create profiles for large-scale users, as well as how to visualize and analyze social simulation data once your experiment concludes, please refer to [Addtional Tutorials](user_generation_visualization.md) for detailed guidance.
