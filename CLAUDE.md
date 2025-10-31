# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OASIS (Open Agent Social Interaction Simulations) is a scalable social media simulator that uses LLM agents to simulate up to one million users on platforms like Twitter and Reddit. It facilitates studying complex social phenomena such as information spread, group polarization, and herd behavior.

**Tech Stack:**
- Python 3.10-3.11 (Poetry for dependency management)
- CAMEL-AI framework for LLM agent capabilities
- SQLite3 for simulation state storage
- Async/await architecture throughout
- pytest for testing, pre-commit hooks for code quality

## Development Commands

### Environment Setup
```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Install pre-commit hooks (auto-formats and lints on commit)
pre-commit install
```

### Testing
```bash
# Run all tests (requires OPENAI_API_KEY environment variable)
pytest test

# Run specific test file
pytest test/agent/test_agent_graph.py

# Run with coverage report
pytest --cov --cov-report=html
# or for comprehensive coverage:
coverage erase
coverage run --source=. -m pytest .
coverage html
# View at htmlcov/index.html
```

### Code Quality
```bash
# Run all pre-commit checks
pre-commit run --all-files

# Update dependencies after modifying pyproject.toml
poetry lock

# Add license headers to Python files
python licenses/update_license.py . licenses/license_template.txt
```

### Documentation
```bash
# Build docs locally (requires npm install -g mintlify)
cd docs
mintlify dev
```

## Architecture

### Core Components

**1. Environment Layer** (`oasis/environment/`)
- `OasisEnv`: Main simulation environment following PettingZoo-style interface
- `LLMAction`: Actions decided by LLM agents
- `ManualAction`: Explicitly programmed actions
- Key methods: `reset()`, `step(actions)`, `close()`
- Uses asyncio semaphore (default: 128) to limit concurrent LLM requests

**2. Social Agent Layer** (`oasis/social_agent/`)
- `SocialAgent`: Extends CAMEL's ChatAgent with social media capabilities
- `AgentGraph`: Manages relationships between agents
- `agents_generator.py`: Functions to generate agent graphs from profiles
  - `generate_twitter_agent_graph(profile_path, model, available_actions)`
  - `generate_reddit_agent_graph(profile_path, model, available_actions)`
- Each agent has:
  - User profile (personality, interests, demographics)
  - Available actions (subset of 23 possible actions)
  - LLM model (can be customized per agent)
  - Custom prompts and tools

**3. Platform Layer** (`oasis/social_platform/`)
- `Platform`: Core platform logic, database operations, recommendation systems
- `Channel`: Message passing between agents and platform
- `database.py`: SQLite3 operations for posts, comments, users, follows, etc.
- `recsys.py`: Recommendation algorithms
  - Twitter: TwHIN-BERT based personalized recommendations
  - Reddit: Hot score based recommendations
  - Random recommendations
- Platform configuration:
  - `recsys_type`: "twhin-bert" | "reddit" | "random"
  - `refresh_rec_post_count`: Posts returned per refresh
  - `max_rec_post_len`: Max posts in recommendation feed
  - `show_score`: Display combined score vs separate likes/dislikes
  - `allow_self_rating`: Allow users to rate own content

**4. Clock System** (`oasis/clock/`)
- Simulates accelerated time for experiments
- Default time magnification: 60x

### Data Flow

1. **Initialization**: `oasis.make(agent_graph, platform, database_path)` creates environment
2. **Reset**: `await env.reset()` initializes database and agents
3. **Step Execution**: `await env.step(actions_dict)` processes actions
   - Actions dict maps agents to LLMAction/ManualAction
   - Platform processes actions via database operations
   - Recommendation system updates
   - Agents receive observations via Channel
4. **Cleanup**: `await env.close()` closes database connections

### Action Types

23 available actions including:
- Content: CREATE_POST, CREATE_COMMENT, QUOTE, REPOST
- Engagement: LIKE_POST, DISLIKE_POST, LIKE_COMMENT, DISLIKE_COMMENT
- Discovery: REFRESH, TREND, SEARCH_POSTS, SEARCH_USER
- Social: FOLLOW, UNFOLLOW, MUTE, UNMUTE
- Group: CREATE_GROUP_CHAT, SEND_GROUP_CHAT_MESSAGE, LEAVE_GROUP_CHAT
- Moderation: REPORT_POST
- Special: DO_NOTHING, INTERVIEW (for asking agents questions)

Each platform has default action sets via `ActionType.get_default_twitter_actions()` and similar.

## Key Patterns

### Running Simulations

All simulations are async. Basic pattern:
```python
async def main():
    # 1. Create model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # 2. Generate agent graph from profiles
    agent_graph = await generate_twitter_agent_graph(
        profile_path="./data/twitter_dataset/...",
        model=model,
        available_actions=available_actions,
    )

    # 3. Create environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path="./data/simulation.db",
    )

    # 4. Run simulation
    await env.reset()

    # Manual actions for initial content
    actions = {
        env.agent_graph.get_agent(0): ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": "Hello!"}
        )
    }
    await env.step(actions)

    # LLM-driven actions
    actions = {agent: LLMAction() for _, agent in env.agent_graph.get_agents()}
    await env.step(actions)

    await env.close()

asyncio.run(main())
```

### Agent Customization

- **Per-agent models**: Pass list of models to `generate_*_agent_graph()`
- **Custom prompts**: Use `user_info_template` parameter in agent creation
- **Custom tools**: Add tools via `tools` parameter to SocialAgent
- **Custom platforms**: Create Platform instance with custom config instead of DefaultPlatformType

### Database Inspection

```python
from oasis import print_db_contents
print_db_contents("./data/simulation.db")
```

## Code Style Guidelines

**From CONTRIBUTING.md:**
- Use `logger` instead of `print` for output
- Avoid abbreviations in naming (use `message_window_size` not `msg_win_sz`)
- Follow Google Python Style Guide
- All code formatted with yapf, isort, and ruff (enforced by pre-commit)
- Keep docstrings under 79 characters per line
- Use raw docstrings: `r"""Description..."""`
- Document all parameters with type hints

**Docstring format:**
```python
def example_function(param1: int, param2: str = "default") -> bool:
    r"""Brief description of function.

    Args:
        param1 (int): Description of param1.
        param2 (str, optional): Description of param2.
            (default: :obj:`"default"`)

    Returns:
        bool: Description of return value.
    """
```

## Testing Structure

- `test/agent/`: Agent and agent graph tests
- `test/infra/database/`: Database operation tests
- `test/infra/recsys/`: Recommendation system tests
- Tests require `OPENAI_API_KEY` environment variable
- Use `conftest.py` for shared test fixtures

## Common Gotchas

1. **Python version**: Must be 3.10-3.11 (not 3.12+)
2. **Database cleanup**: Delete old .db files before new simulations or use `if os.path.exists(db_path): os.remove(db_path)`
3. **Async everywhere**: All simulation code must be async/await
4. **API keys**: Set `OPENAI_API_KEY` env var for testing
5. **Profile paths**: Agent profiles are CSV (Twitter) or JSON (Reddit) files
6. **Action validation**: Not all actions work on all platforms - use platform-specific defaults
7. **Logging**: Logs automatically saved to `./log/` directory with timestamps

## Important Files

- `oasis/__init__.py`: Main exports and public API
- `oasis/environment/env.py`: Core simulation loop
- `oasis/social_agent/agent.py`: Agent implementation
- `oasis/social_platform/platform.py`: Platform logic
- `oasis/social_platform/typing.py`: Enums and type definitions
- `examples/`: Working examples for different scenarios
- `generator/`: Tools for generating synthetic user profiles
- `visualization/`: Analysis and visualization scripts