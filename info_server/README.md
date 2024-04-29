# Installation

```commandline
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Pre-commit
Run pre-commit before commiting
```commandline
pre-commit run --files [your_changed_file_name]
```

# DB Creation Example
Run
```commandline
python3 core/create_database.py
```
A twitter database example will be created in the `db` folder.

# Test
```commandline
pytest test
```