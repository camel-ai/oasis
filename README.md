# Pipeline

## Quickstart

### Step 1: Create and Activate a Virtual Environment
For win:
```bash
python -m venv camel_ss       # Create virtual environment
camel_ss\Scripts\activate  # Activate virtual environment
```
For macOS:
```bash
python -m venv camel_ss       # Create virtual environment
source camel_ss/bin/activate  # Activate virtual environment
```

### Step 2: Install Necessary Packages

```bash
pip install --upgrade pip     # Upgrade pip
pip install -r requirements.txt # Install packages from requirements file
```

### Step 3: Run the Main Program

```bash
python main.py                # Run the program
# You will be prompted to input 'AgentCount' which is the size of the AI-Society/Sandbox
```


## to-do

- [x] waiting for the origin info-server features to PR&Merge 
- [ ] Agent Generator 
- [ ] Agent running 


## pytest

For win:
```bash
pytest test\test_infra  
```

For Mac OS X:
```bash
pytest
```
