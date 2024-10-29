# 🚢 More Tutorials

## 👨‍👨‍👧‍👦 User Generation

### 🔵 Twitter User Profile Generation

...

### 🟠 Reddit User Profile Generation

We utilized the findings from a survey on the distribution of real Reddit users (https://explodingtopics.com/blog/reddit-users) as a reference to randomly generate demographic information for our users. Leveraging this data, we then employed GPT to create more detailed descriptions.

- Step 1:

Modify your OpenAI API key at the top of the `generator\reddit\user_generate.py` file.

```bash
client = OpenAI(api_key='sk-xxx')
```

- Step 2:

Change the number of users you want to generate and the path where the user data will be saved at the bottom of the `generator\reddit\user_generate.py` file.

```bash
if __name__ == "__main__":
    N = 10000  # Target user number
    user_data = generate_user_data(N)
    output_path = 'user_data_10000.json'
    save_user_data(user_data, output_path)
    print(f"Generated {N} user profiles and saved to {output_path}")
```

- Step 3:

Run the Script:

```bash
python generator/reddit/user_generate.py
```

## 📊 Data Visualization

### 🟠 Reddit Sore Analysis

- Step 1:

After running python `scripts/reddit_simulation_align_with_human/reddit_simulation_align_with_human.py`, a database file and a JSON file with the same name will be generated. Modify `visualization/reddit_simulation_align_with_human/code/analysis_all.py` here to their respective paths and the common filename (excluding the extension).

```bash
if __name__ == "__main__":
    folder_path = ("visualization/reddit_simulation_align_with_human"
                   "/experiment_results")
    exp_name = "business_3600"
    db_path = folder_path + f"/{exp_name}.db"
    exp_info_file_path = folder_path + f"/{exp_name}.json"
    analysis_score.main(exp_info_file_path, db_path, exp_name, folder_path)
```

- Step 2:

```bash
pip install matplotlib
```

- Step 3:

Run the Script:

```bash
python visualization/reddit_simulation_align_with_human/code/analysis_all.py
```

Then, you will see the scores of the three groups—down-treated, control, up-treated at the end of the experiment, like this

<p align="center">
  <img src='../visualization/reddit_simulation_align_with_human/experiment_results/score_business_3600.png' width=400>
</p>

### 🟠 Reddit Counterfactual Content Analysis

- Step 1:

First, you need to add your OpenAI API key to the system's environment variables.

- Step 2:

After running python `scripts/reddit_simulation_counterfactual/reddit_simulation_counterfactual.py`, 3 database files will be generated. Modify `visualization/reddit_simulation_counterfactual/code/analysis_couterfact.py` here to their respective paths.

```bash
db_files = [
    'couterfact_up_100.db',
    'couterfact_cnotrol_100.db',
    'couterfact_down_100.db'
]
```

- Step 3:

```bash
pip install aiohttp
```

- Step 4:

Run the Script:

```bash
python visualization/reddit_simulation_counterfactual/code/analysis_couterfact.py
```

Then, you will see the disagree scores of the three groups—down-treated, control, up-treated in each timestep of the experiment, like this

<p align="center">
  <img src='../visualization/reddit_simulation_counterfactual/result/example.png' width=400>
</p>
