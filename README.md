# Social Simulation

Social Simulation.

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
```
