# Pipeline

## Quickstart

### Step 1: Create and Activate a Virtual Environment
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

### Step 4: Show Results from the Database

```bash
python showDB.py  # Display the database content, including Users, Accounts, Tweets, Traces, Rec
```


## to-do

- [ ] waiting for the origin info-server features to PR&Merge 
- [ ] Agent Generator 
- [ ] Agent running 


## pytest

pytest test\test_infra