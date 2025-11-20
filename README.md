<div align="center">
  <a href="https://www.camel-ai.org/">
    <img src="assets/banner.png" alt=banner>
  </a>
</div>

</br>

<div align="center">

<h1> OASIS: Open Agent Social Interaction Simulations with One Million Agents
</h1>

[![Documentation][docs-image]][docs-url]
[![Discord][discord-image]][discord-url]
[![X][x-image]][x-url]
[![Reddit][reddit-image]][reddit-url]
[![Wechat][wechat-image]][wechat-url]
[![Wechat][oasis-image]][oasis-url]
[![Hugging Face][huggingface-image]][huggingface-url]
[![Star][star-image]][star-url]
[![Package License][package-license-image]][package-license-url]

<h4 align="center">

[Community](https://github.com/camel-ai/camel#community) |
[Paper](https://arxiv.org/abs/2411.11581) |
[Examples](https://github.com/camel-ai/oasis/tree/main/scripts) |
[Dataset](https://huggingface.co/datasets/oasis-agent/oasis-dataset) |
[Citation](https://github.com/camel-ai/oasis#-citation) |
[Contributing](https://github.com/camel-ai/oasis#-contributing-to-oasis) |
[CAMEL-AI](https://www.camel-ai.org/)

</h4>

</div>

<br>

<p align="left">
  <img src='assets/intro.png'>

🏝️ OASIS is a scalable, open-source social media simulator that incorporates large language model agents to realistically mimic the behavior of up to one million users on platforms like Twitter and Reddit. It's designed to facilitate the study of complex social phenomena such as information spread, group polarization, and herd behavior, offering a versatile tool for exploring diverse social dynamics and user interactions in digital environments.

</p>

<br>

<div align="center">
🌟 Star OASIS on GitHub and be instantly notified of new releases.
</div>

<br>

<div align="center">
    <img src="assets/star.gif" alt="Star" width="196" height="52">
  </a>
</div>

<br>

## ✨ Key Features

### 📈 Scalability

OASIS supports simulations of up to ***one million agents***, enabling studies of social media dynamics at a scale comparable to real-world platforms.

### 📲 Dynamic Environments

Adapts to real-time changes in social networks and content, mirroring the fluid dynamics of platforms like **Twitter** and **Reddit** for authentic simulation experiences.

### 👍🏼 Diverse Action Spaces

Agents can perform **23 actions**, such as following, commenting, and reposting, allowing for rich, multi-faceted interactions.

### 🔥 Integrated Recommendation Systems

Features **interest-based** and **hot-score-based recommendation algorithms**, simulating how users discover content and interact within social media platforms.

<br>

## 📺 Demo Video

### Introducing OASIS: Open Agent Social Interaction Simulations with One Million Agents

https://github.com/user-attachments/assets/3bd2553c-d25d-4d8c-a739-1af51354b15a

<br>

For more showcaes:

- Can 1,000,000 AI agents simulate social media?
  [→Watch demo](https://www.youtube.com/watch?v=lprGHqkApus&t=2s)

<br>

## 🎯 Usecase

<div align="left">
    <img src="assets/research_simulation.png" alt=usecase1>
    <img src="assets/interaction.png" alt=usecase2>
   <a href="http://www.matrix.eigent.ai">
    <img src="assets/content_creation.png" alt=usecase3>
   </a>
    <img src="assets/prediction.png" alt=usecase4>
</div>

## ⚙️ Quick Start

Get started with OASIS in minutes! Choose between Reddit or Twitter simulations below.

### Reddit Simulation Quick Start

1. **Install the OASIS package:**

Installing OASIS is a breeze thanks to its availability on PyPI. Simply open your terminal and run:

```bash
pip install camel-oasis
```

2. **Set up your OpenAI API key:**

```bash
# For Bash shell (Linux, macOS, Git Bash on Windows):
export OPENAI_API_KEY=<insert your OpenAI API key>

# For Windows Command Prompt:
set OPENAI_API_KEY=<insert your OpenAI API key>
```

3. **Prepare the agent profile file:**

Create the profile you want to assign to the agent. As an example, you can download [user_data_36.json](https://github.com/camel-ai/oasis/blob/main/data/reddit/user_data_36.json) and place it in your local `./data/reddit` folder.

4. **Run the following Python code:**

```python
import asyncio
import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import (ActionType, LLMAction, ManualAction,
                   generate_reddit_agent_graph)


async def main():
    # Define the model for the agents
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Define the available actions for the agents
    available_actions = [
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_COMMENT,
        ActionType.DISLIKE_COMMENT,
        ActionType.SEARCH_POSTS,
        ActionType.SEARCH_USER,
        ActionType.TREND,
        ActionType.REFRESH,
        ActionType.DO_NOTHING,
        ActionType.FOLLOW,
        ActionType.MUTE,
    ]

    agent_graph = await generate_reddit_agent_graph(
        profile_path="./data/reddit/user_data_36.json",
        model=openai_model,
        available_actions=available_actions,
    )

    # Define the path to the database
    db_path = "./data/reddit_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )

    # Run the environment
    await env.reset()

    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = [
        ManualAction(action_type=ActionType.CREATE_POST,
                     action_args={"content": "Hello, world!"}),
        ManualAction(action_type=ActionType.CREATE_COMMENT,
                     action_args={
                         "post_id": "1",
                         "content": "Welcome to the OASIS World!"
                     })
    ]
    actions_1[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_COMMENT,
        action_args={
            "post_id": "1",
            "content": "I like the OASIS world."
        })
    await env.step(actions_1)

    actions_2 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents()
    }

    # Perform the actions
    await env.step(actions_2)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### Twitter Simulation Quick Start

For a Twitter simulation with just a few agents:

1. **Install the OASIS package** (if not already installed):

```bash
pip install camel-oasis
```

2. **Set up your OpenAI API key:**

```bash
# For Bash shell (Linux, macOS, Git Bash on Windows):
export OPENAI_API_KEY=<insert your OpenAI API key>

# For Windows Command Prompt:
set OPENAI_API_KEY=<insert your OpenAI API key>
```

3. **Prepare the agent profile file:**

Download [False_Business_0.csv](https://github.com/camel-ai/oasis/blob/main/data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv) and place it in your local `./data/twitter_dataset/anonymous_topic_200_1h/` folder.

4. **Run the following Python code:**

```python
import asyncio
import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import ActionType, LLMAction, ManualAction, generate_twitter_agent_graph


async def main():
    # Define the model for the agents
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Define the available actions for the agents
    available_actions = ActionType.get_default_twitter_actions()

    agent_graph = await generate_twitter_agent_graph(
        profile_path="./data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv",
        model=openai_model,
        available_actions=available_actions,
    )

    # Define the path to the database
    db_path = "./data/twitter_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
    )

    # Run the environment
    await env.reset()

    # Agent 0 creates an initial post
    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Just joined OASIS! Excited to explore this platform."}
    )
    await env.step(actions_1)

    # Activate a few agents (agents 1, 2, 3, 4) to interact using LLM
    actions_2 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents([1, 2, 3, 4])
    }
    await env.step(actions_2)

    # Agent 1 creates a response post
    actions_3 = {}
    actions_3[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Welcome to OASIS! It's great to have you here."}
    )
    await env.step(actions_3)

    # All agents respond with LLM-driven actions
    actions_4 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents()
    }
    await env.step(actions_4)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
```

<br>

> \[!TIP\]
> For more detailed instructions and additional configuration options, check out the [documentation](https://docs.oasis.camel-ai.org/).

### More Tutorials

To discover how to create profiles for large-scale users, as well as how to visualize and analyze social simulation data once your experiment concludes, please refer to [More Tutorials](examples/experiment/user_generation_visualization.md) for detailed guidance.

<div align="center">
  <img src="assets/tutorial.png" alt="Tutorial Overview">
</div>

## 🧪 MVP: "Needle in the Hashtag" Dataset Generation

Use this minimal pipeline to generate a small synthetic dataset with agents emitting inline label tokens like `<LBL:INCEL_SLANG>`, `<LBL:MISINFO_CLAIM>`, and `<LBL:SUPPORTIVE>`, powered by Gemini 2.5 Flash‑Lite.

### Prerequisites
- Python 3.10 or 3.11 with Poetry
- `.env` containing your Gemini key:

```bash
echo "GEMINI_API_KEY=your-gemini-key" > .env
```

### Configure the MVP
Edit `configs/mvp_master.yaml`:
- `personas`: set counts per persona (default: 5/5/5 = 15 agents)
- `simulation.steps`: number of steps (default: 8)
- `simulation.action_mix`: interaction mix (e.g., higher `create_comment` for threads)
- `simulation.gemini_model`: `gemini-2.5-flash-lite`
- `simulation.skip_imputation`: true to keep raw `<LBL:...>` tokens

### Run the simulation
Creates `data/mvp/oasis_mvp_gemini.db` with posts, comments, likes, follows.

```bash
poetry install
poetry run python3 scripts/run_mvp_gemini.py --config ./configs/mvp_master.yaml | cat
```

### Build the dataset
- Raw (preserve `<LBL:...>` tokens):

```bash
poetry run python3 scripts/build_dataset.py \
  --db ./data/mvp/oasis_mvp_gemini.db \
  --out ./data/mvp/posts_mvp_raw.jsonl \
  --static-bank ./data/label_tokens_static_bank.yaml \
  --skip-imputation | cat
```

- Imputed (replace `<LBL:...>` from a static phrase bank):

```bash
poetry run python3 scripts/build_dataset.py \
  --db ./data/mvp/oasis_mvp_gemini.db \
  --out ./data/mvp/posts_mvp.jsonl \
  --static-bank ./data/label_tokens_static_bank.yaml | cat
```

Optional validation/visualization:

```bash
poetry run python3 scripts/mvp_validate.py --file ./data/mvp/posts_mvp.jsonl | cat
poetry run python3 scripts/visualize_mvp.py --db ./data/mvp/oasis_mvp_gemini.db --out ./data/mvp/posts_mvp_raw.html | cat
```

### How to modify behavior
- **Agents**: change counts in `personas` (keep total ~10–20 for MVP)
- **Steps**: set `simulation.steps`
- **Action mix**: tune `simulation.action_mix` to bias threads (e.g., increase `create_comment`)
- **Model and temperature**: `simulation.gemini_model`, `simulation.temperature`
- **Imputation**: set `simulation.skip_imputation: true` (or use CLI `--skip-imputation`)
- **Thread context**: replies automatically receive the original post + all existing replies as context

Notes:
- Safety settings are disabled in the Gemini client for red‑teaming.
- All actions are available to agents by default in this MVP.

### Graph + Recsys calibration (10k agents, TwHIN‑BERT)

For larger‑scale experiments (e.g., 10k agents) with TwHIN‑BERT, we use a calibrated SBM + PA + triadic‑closure graph and a seed‑only runner:

- **Personas (500 → 10k)**:
  - Generated deterministically via `scripts/generate_personas.py` using `configs/mvp_master.yaml`:
    - `incel: 3334`, `misinfo: 3333`, `benign: 3333`
    - `seed: 314159`
- **Graph generator**: `scripts/build_graph.py`
  - Uses a simple SBM + preferential attachment + triadic closure, with classes taken from `primary_label` in `data/personas_mvp.csv`.
  - Recommended 10k parameters (also stored under `graph` in `configs/mvp_master.yaml`):

```bash
poetry run python3 scripts/generate_personas.py --config ./configs/mvp_master.yaml | cat

poetry run python3 scripts/build_graph.py \
  --personas ./data/personas_mvp.csv \
  --out ./data/edges_mvp_10k_tc_assort.csv \
  --theta-b 0.0005 \
  --rho 0.6 \
  --alpha 6.0 \
  --avg-out-degree 10 \
  --seed 314159 \
  --tc-prob 0.02 \
  --tc-max-per-node 5 \
  --tc-homophily 0.4 \
  --tc-assort-beta 2.0 \
  --class-col primary_label | cat
```

- **Empirical structure at 10k** (from `scripts/graph_metrics.py`):
  - Strong within‑class communities (homophily by `primary_label`)
  - Slightly disassortative by degree (hubs broadcast to many smaller accounts)
  - Non‑trivial clustering from triadic closure (local “pods” inside each class)
  - Heavy‑tailed but not extreme degree distribution (visible hubs per class without a single super‑hub)

- **Seed‑only TwHIN‑BERT calibration**:
  - We initialize the DB, seed follows and a small pool of initial posts, refresh TwHIN‑BERT once, write provenance, and then exit:

```bash
MVP_EDGES_CSV="./data/edges_mvp_10k_tc_assort.csv" \
poetry run python3 scripts/run_mvp_gemini.py \
  --config ./configs/mvp_master.yaml \
  --seed-only | cat
```

  - This:
    - Seeds ~10k users and ~1e5 follow edges into the SQLite DB and `AgentGraph`
    - Creates a small number of initial posts so follow feeds are non‑empty
    - Runs TwHIN‑BERT to populate the `rec` table once (using the current `platform` config in `configs/mvp_master.yaml`)
    - Writes `dataset_provenance.json` next to the DB, including graph params and `edges_*.csv` SHA‑256 hash

You can tune the **recsys amplification** via the `platform` block in `configs/mvp_master.yaml`:
- `following_post_count`: number of posts drawn from follows per refresh
- `max_rec_post_len`: number of recommended posts injected per refresh
- `refresh_rec_post_count`: size of the candidate pool per refresh
Increasing `max_rec_post_len` and reducing `following_post_count` shifts feeds toward TwHIN‑BERT‑recommended content (including cross‑class posts), while higher homophily in the graph keeps most organic follow edges within class.

## 🚀 Production Simulation + Automated Reporting

Run a production Twitter‑like simulation backed by your chosen LLM (default example uses xAI Grok) and automatically generate JSONL exports and visualizations in one step.

### Prerequisites
- Python 3.10 or 3.11, Poetry installed
- Install deps:

```bash
poetry install | cat
```

- Set your LLM key (example for xAI Grok):

```bash
export XAI_API_KEY=<your-xai-key>
```

### One-step run (simulation + report)
This will create a new SQLite DB and then produce JSONL exports and HTML/PNG visuals automatically.

```bash
cd /Users/jordanmoshcovitis/Documents/GitHub/oasis
poetry run python3 scripts/run_production_sim.py \
  --manifest configs/mvp_master.yaml \
  --personas-csv ./data/mvp/personas_20.csv \
  --db ./data/mvp/oasis_mvp_grok_20.db \
  --steps 5 \
  --edges-csv ./data/mvp/edges_20.csv \
  --warmup-steps 1 \
  --unique-db \
  --report | cat
```

### End-to-end production pipeline (20-agent master config)

Use this sequence whenever you change `configs/mvp_master.yaml` and want a fresh dataset:

- **1. Configure the master manifest (`configs/mvp_master.yaml`)**
  - **`personas` block**: set `incel` / `misinfo` / `benign` counts, `seed`, and `personas_csv` (e.g. `./data/mvp/personas_20.csv`).
  - **`graph` block**: set `class_col`, `theta_b`, `rho`, `alpha`, triadic-closure params, `rewire_ratio`, `seed`, and `edges_csv` (e.g. `./data/mvp/edges_20.csv`).

- **2. Generate personas for this manifest**

  ```bash
  poetry run python3 scripts/generate_personas.py \
    --config ./configs/mvp_master.yaml | cat
  # writes personas to the personas_csv path in the manifest (default: ./data/mvp/personas_20.csv)
  ```

- **3. Generate the follow graph for this personas file**

  ```bash
  poetry run python3 scripts/build_graph.py \
    --config ./configs/mvp_master.yaml | cat
  # writes edges to the edges_csv path in the manifest (default: ./data/mvp/edges_20.csv)
  ```

- **4. Run the production simulation + report (command above)**
  - `run_production_sim.py` will:
    - Build `ExtendedSocialAgent` instances from the personas CSV.
    - Create a Twitter-like `Platform` with TwHIN-BERT recsys.
    - Seed follows from the edges CSV (if provided and table is empty).
    - Run `warmup_steps` + `steps` of all-agents `LLMAction`.
    - Optionally invoke `scripts/report_production.py` to export JSONL + HTML/PNGs.

### Safety checks and failure modes

`scripts/run_production_sim.py` performs conservative validation before any LLM calls:

- **Personas vs manifest**
  - Fails if the personas CSV:
    - does not exist, or
    - has zero rows.
  - Derives an **expected total agent count** from:
    - `population` block (for manifests like `data/manifest_mvp.yaml`), or
    - `personas` counts (`incel`, `misinfo`, `benign`) in `configs/mvp_master.yaml`.
  - If `CSV rows != expected_total`, it exits with:

    > `[production] Personas CSV row count does not match manifest population/persona counts. ... Regenerate personas for this manifest.`

- **Edges vs personas**
  - If `--edges-csv` (or `PROD_EDGES_CSV`) is set:
    - Fails if the edges CSV does not exist.
    - Scans all `follower_id` / `followee_id` and computes `max_id`; if `max_id >= persona_rows` it exits with:

      > `[production] Edges CSV references user_id outside the personas index range. ... Regenerate the graph after updating personas.`
  - If `--edges-csv` is omitted, the simulation runs **without** an initial follow graph (no seeding).

These checks ensure you **cannot** accidentally:
- run a 20-row personas CSV with a 10k manifest (or vice versa), or
- pair an edges file built for a different population size with the current personas.

What it produces (defaults):
- DB: under `./data/mvp/` (name includes a timestamp when `--unique-db` is used)
- Sidecar: `<db_dir>/sidecar.jsonl`
- Report output directory: `<db_dir>/production/`
- JSONL (content + labels): `production_export.jsonl`
- JSONL (actions: trace + likes/comments/posts/follows): `production_actions.jsonl`
- PNGs: `action_timeline.png`, `interaction_network.png`
- HTML: `production_report.html` (summary) and `production_threads.html` (threaded view with persona badges + label chips)

### Customize report output paths
Add these optional flags to the same command:

```bash
--report-out-dir ./data/reports/run_YYYYMMDD_HHMMSS \
--report-export-jsonl ./data/reports/run_YYYYMMDD_HHMMSS/export.jsonl \
--report-export-actions ./data/reports/run_YYYYMMDD_HHMMSS/actions.jsonl \
--report-threads-html ./data/reports/run_YYYYMMDD_HHMMSS/threads.html
```

### Run report separately (optional)
If you ran the simulation without `--report`, you can generate outputs later:

```bash
DB=$(ls -t data/mvp/oasis_mvp_*.db | head -1)
poetry run python3 scripts/report_production.py \
  --db "$DB" \
  --sidecar "$(dirname "$DB")/sidecar.jsonl" \
  --out-dir data/production \
  --export-jsonl data/production/production_export.jsonl \
  --export-actions data/production/production_actions.jsonl \
  --threads-html data/production/production_threads.html | cat
```

## 📢 News

### Upcoming Features & Contributions

> We welcome community contributions! Join us in building these exciting features.

- [Support Multi Modal Platform](https://github.com/camel-ai/oasis/issues/47)

<!-- - Public release of our dataset on Hugging Face (November 05, 2024) -->

### Latest Updates

📢 Add the report post action to mark inappropriate content. - 📆 June 8, 2025

- Add features for creating group chats, sending messages in group chats, and leaving group chats. - 📆 June 6, 2025
- Support Interview Action for asking agents specific questions and getting answers. - 📆 June 2, 2025
- Support customization of each agent's models, tools, and prompts; refactor the interface to follow the PettingZoo style. - 📆 May 22, 2025
- Refactor into the OASIS environment, publish camel-oasis on PyPI, and release the documentation. - 📆 April 24, 2025
- Support OPENAI Embedding model for Twhin-Bert Recommendation System. - 📆 March 25, 2025
  ...
- Slightly refactoring the database to add Quote Action and modify Repost Action - 📆 January 13, 2025
- Added the demo video and oasis's star history in the README - 📆 January 5, 2025
- Introduced an Electronic Mall on the Reddit platform - 📆 December 5, 2024
- OASIS initially released on arXiv - 📆 November 19, 2024
- OASIS GitHub repository initially launched - 📆 November 19, 2024

## 🔎 Follow-up Research

- [MultiAgent4Collusion](https://github.com/renqibing/MultiAgent4Collusion): multi-agent collusion simulation framework in social systems
- More to come...

If your research is based on OASIS, we'd be happy to feature your work here—feel free to reach out or submit a pull request to add it to the [README](https://github.com/camel-ai/oasis/blob/main/README.md)!

## 🥂 Contributing to OASIS🏝️

> We greatly appreciate your interest in contributing to our open-source initiative. To ensure a smooth collaboration and the success of contributions, we adhere to a set of contributing guidelines similar to those established by CAMEL. For a comprehensive understanding of the steps involved in contributing to our project, please refer to the OASIS [contributing guidelines](https://github.com/camel-ai/oasis/blob/master/CONTRIBUTING.md). 🤝🚀
>
> An essential part of contributing involves not only submitting new features with accompanying tests (and, ideally, examples) but also ensuring that these contributions pass our automated pytest suite. This approach helps us maintain the project's quality and reliability by verifying compatibility and functionality.

## 📬 Community & Contact

If you're keen on exploring new research opportunities or discoveries with our platform and wish to dive deeper or suggest new features, we're here to talk. Feel free to get in touch for more details at camel.ai.team@gmail.com.

<br>

- Join us ([*Discord*](https://discord.camel-ai.org/) or [*WeChat*](https://ghli.org/camel/wechat.png)) in pushing the boundaries of finding the scaling laws of agents.

- Join WechatGroup for further discussions!

<div align="">
  <img src="assets/wechatgroup.png" alt="WeChat Group QR Code" width="600">
</div>

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=camel-ai/oasis&type=Date)](https://star-history.com/#camel-ai/oasis&Date)

## 🔗 Citation

```
@misc{yang2024oasisopenagentsocial,
      title={OASIS: Open Agent Social Interaction Simulations with One Million Agents},
      author={Ziyi Yang and Zaibin Zhang and Zirui Zheng and Yuxian Jiang and Ziyue Gan and Zhiyu Wang and Zijian Ling and Jinsong Chen and Martz Ma and Bowen Dong and Prateek Gupta and Shuyue Hu and Zhenfei Yin and Guohao Li and Xu Jia and Lijun Wang and Bernard Ghanem and Huchuan Lu and Chaochao Lu and Wanli Ouyang and Yu Qiao and Philip Torr and Jing Shao},
      year={2024},
      eprint={2411.11581},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.11581},
}
```

## 🙌 Acknowledgment

We would like to thank Douglas for designing the logo of our project.

## 🖺 License

The source code is licensed under Apache 2.0.

[discord-image]: https://img.shields.io/discord/1082486657678311454?logo=discord&labelColor=%20%235462eb&logoColor=%20%23f5f5f5&color=%20%235462eb
[discord-url]: https://discord.camel-ai.org/
[docs-image]: https://img.shields.io/badge/Documentation-EB3ECC
[docs-url]: https://docs.oasis.camel-ai.org/
[huggingface-image]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CAMEL--AI-ffc107?color=ffc107&logoColor=white
[huggingface-url]: https://huggingface.co/camel-ai
[oasis-image]: https://img.shields.io/badge/WeChat-OASISProject-brightgreen?logo=wechat&logoColor=white
[oasis-url]: ./assets/wechatgroup.png
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/camel-ai/oasis/blob/main/licenses/LICENSE
[reddit-image]: https://img.shields.io/reddit/subreddit-subscribers/CamelAI?style=plastic&logo=reddit&label=r%2FCAMEL&labelColor=white
[reddit-url]: https://www.reddit.com/r/CamelAI/
[star-image]: https://img.shields.io/github/stars/camel-ai/oasis?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/camel-ai/oasis/stargazers
[wechat-image]: https://img.shields.io/badge/WeChat-CamelAIOrg-brightgreen?logo=wechat&logoColor=white
[wechat-url]: ./assets/wechat.JPGwechat.jpg
[x-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social
[x-url]: https://x.com/CamelAIOrg
