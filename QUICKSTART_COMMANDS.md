# MVP: Needle in the Hashtag — Quick Start (Gemini)

This section covers the minimal dataset-generation pipeline using Gemini 2.5 Flash‑Lite.

## Prerequisites

- Python 3.10 or 3.11, Poetry
- Gemini API key in `.env`:

```bash
echo "GEMINI_API_KEY=your-gemini-key" > .env
```

## Install

```bash
poetry install | cat
```

## Run Simulation (Twitter platform)

```bash
poetry run python3 scripts/run_mvp_gemini.py --config ./configs/mvp_master.yaml | cat
```

Outputs DB at `data/mvp/oasis_mvp_gemini.db`.

## Build Dataset

Raw (keep <LBL:...> tokens):

```bash
poetry run python3 scripts/build_dataset.py \
  --db ./data/mvp/oasis_mvp_gemini.db \
  --out ./data/mvp/posts_mvp_raw.jsonl \
  --static-bank ./data/label_tokens_static_bank.yaml \
  --skip-imputation | cat
```

Imputed (replace tokens deterministically):

```bash
poetry run python3 scripts/build_dataset.py \
  --db ./data/mvp/oasis_mvp_gemini.db \
  --out ./data/mvp/posts_mvp.jsonl \
  --static-bank ./data/label_tokens_static_bank.yaml | cat
```

Optional:

```bash
poetry run python3 scripts/mvp_validate.py --file ./data/mvp/posts_mvp.jsonl | cat
poetry run python3 scripts/visualize_mvp.py --db ./data/mvp/oasis_mvp_gemini.db --out ./data/mvp/posts_mvp_raw.html | cat
```

## Modify Behavior (configs/mvp_master.yaml)

- `personas`: set agent counts (default 5/5/5)
- `simulation.steps`: default 8
- `simulation.action_mix`: increase `create_comment` for denser threads
- `simulation.gemini_model`: `gemini-2.5-flash-lite`
- `simulation.skip_imputation`: true to preserve label tokens

---

# OASIS Twitter Simulation - Quick Start Commands

This guide provides step-by-step commands to run a Twitter simulation with 111 AI agents and visualize the results.

## Prerequisites

- Python 3.10 or 3.11
- Poetry installed
- OpenAI API key

## Setup Commands

### 1. Install Dependencies

```bash
cd /path/to/oasis
poetry install
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**Or** export directly in your shell:

**For Mac/Linux:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**For Windows Command Prompt:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**For Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

### 3. Verify Data Files Exist

Check that the agent profile CSV exists:

```bash
ls -la ./data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv
```

## Running the Simulation

### Option A: If using `.env` file (Mac/Linux)

```bash
cd /path/to/oasis
export $(cat .env | xargs)
poetry run python3 examples/twitter_quick_start.py
```

### Option B: If using `.env` file (Windows)

```cmd
cd \path\to\oasis
for /f "tokens=*" %i in (.env) do set %i
poetry run python examples/twitter_quick_start.py
```

### Option C: Inline with API key (Mac/Linux)

```bash
cd /path/to/oasis
OPENAI_API_KEY="your-api-key-here" poetry run python3 examples/twitter_quick_start.py
```

### Option D: Inline with API key (Windows)

```cmd
cd \path\to\oasis
set OPENAI_API_KEY=your-api-key-here && poetry run python examples/twitter_quick_start.py
```

## What Happens During the Simulation

The script will:
1. Load 111 AI agents from the CSV file
2. Create a simulation database at `./data/twitter_simulation.db`
3. Run multiple steps:
   - Agent 0 creates an initial post
   - Agents 1-4 interact using LLM
   - Agent 1 creates a welcome post
   - All 111 agents perform LLM-driven actions (like, repost, quote, create posts)
4. Save all interactions to the database

**Expected runtime:** 2-5 minutes (depends on OpenAI API response time)

## Visualizing the Results

After the simulation completes, run the visualization script:

### Mac/Linux

```bash
cd /path/to/oasis
export $(cat .env | xargs)
poetry run python3 examples/visualize_simulation.py
```

### Windows

```cmd
cd \path\to\oasis
for /f "tokens=*" %i in (.env) do set %i
poetry run python examples/visualize_simulation.py
```

## Output Files

The visualization script generates:

1. **Action Timeline Chart**: `./data/action_timeline.png`
   - Bar chart showing distribution of agent actions

2. **Interaction Network Graph**: `./data/interaction_network.png`
   - Network visualization of agent interactions
   - Node size = activity level
   - Edge width = interaction strength

3. **Console Output**: 
   - Trace table sample
   - Action summary statistics
   - Network statistics

## Viewing the Database Directly

### View all tables in the database

```bash
sqlite3 ./data/twitter_simulation.db "SELECT name FROM sqlite_master WHERE type='table';"
```

### View trace table (first 20 rows)

```bash
sqlite3 ./data/twitter_simulation.db "SELECT * FROM trace LIMIT 20;"
```

### View action summary

```bash
sqlite3 ./data/twitter_simulation.db "SELECT action, COUNT(*) as count FROM trace WHERE action != 'sign_up' GROUP BY action ORDER BY count DESC;"
```

### View all posts

```bash
sqlite3 ./data/twitter_simulation.db "SELECT user_id, content, created_at FROM post LIMIT 10;"
```

## Complete Workflow (Mac/Linux)

Here's the complete sequence from start to finish:

```bash
# 1. Navigate to project
cd /path/to/oasis

# 2. Install dependencies (first time only)
poetry install

# 3. Set up API key in .env file (first time only)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 4. Run the simulation
export $(cat .env | xargs)
poetry run python3 examples/twitter_quick_start.py

# 5. Visualize the results
poetry run python3 examples/visualize_simulation.py

# 6. View the generated visualizations
open ./data/action_timeline.png
open ./data/interaction_network.png
```

## Complete Workflow (Windows)

```cmd
REM 1. Navigate to project
cd \path\to\oasis

REM 2. Install dependencies (first time only)
poetry install

REM 3. Set up API key in .env file (first time only)
echo OPENAI_API_KEY=your-api-key-here > .env

REM 4. Load environment variables
for /f "tokens=*" %i in (.env) do set %i

REM 5. Run the simulation
poetry run python examples/twitter_quick_start.py

REM 6. Visualize the results
poetry run python examples/visualize_simulation.py

REM 7. View the generated visualizations
start ./data/action_timeline.png
start ./data/interaction_network.png
```

## Troubleshooting

### Missing API Key Error
```
ValueError: Missing or empty required API keys in environment variables: OPENAI_API_KEY
```

**Solution:** Ensure your `.env` file exists and contains your API key, or export it directly:
```bash
export OPENAI_API_KEY="your-actual-key"
```

### Module Not Found Error
```
ModuleNotFoundError: No module named 'camel'
```

**Solution:** Install dependencies:
```bash
poetry install
```

### Matplotlib Missing (for visualization)
```
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution:** Add matplotlib as a dev dependency:
```bash
poetry add matplotlib --group dev
```

### CSV File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: './data/twitter_dataset/...'
```

**Solution:** Ensure you're running from the project root directory or adjust the path.

## Customization

### Use Different Agent Profiles

Edit line 35 in `twitter_quick_start.py`:
```python
profile_path="./data/twitter_dataset/your_custom_file.csv"
```

### Change Number of Simulation Steps

Modify the actions in `twitter_quick_start.py`:
- Add more `await env.step(actions)` calls
- Change which agents are activated in each step

### Adjust Visualization Parameters

In `visualize_simulation.py`, you can:
- Change `figsize=(16, 12)` for different graph sizes
- Modify `min_degree=1` to filter nodes
- Customize colors, layouts, and styling

## Next Steps

- Explore other example scripts in `examples/`
- Check the documentation: https://docs.oasis.camel-ai.org/
- Try scaling up to more agents
- Customize agent behaviors and actions
- Analyze different social phenomena (misinformation, polarization, etc.)

## Support

For issues or questions:
- GitHub Issues: https://github.com/camel-ai/oasis/issues
- Documentation: https://docs.oasis.camel-ai.org/
- Discord: https://discord.camel-ai.org/

