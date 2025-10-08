______________________________________________________________________

# OASIS: Open Agent Social Interaction Simulations with One Million Agents

<p align="center">
  <img src='assets/logo.jpg' width=400>
</p>

## 📝 Overview

<p align="center">
  <img src='assets/intro-new-logo.png' width=900>
</p>
🏝️ OASIS is a scalable, open-source social media simulator that integrates large language models with rule-based agents to realistically mimic the behavior of up to one million users on platforms like Twitter and Reddit. It's designed to facilitate the study of complex social phenomena such as information spread, group polarization, and herd behavior, offering a versatile tool for exploring diverse social dynamics and user interactions in digital environments.

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

### Can 1,000,000 AI agents simulate social media?

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

<br>

> \[!TIP\]
> For more detailed instructions and additional configuration options, check out the documentation.

### More Tutorials

To discover how to create profiles for large-scale users, as well as how to visualize and analyze social simulation data once your experiment concludes, please refer to [More Tutorials](examples/experiment/user_generation_visualization.md) for detailed guidance.

<div align="center">
  <img src="assets/tutorial.png" alt="Tutorial Overview">
</div>
