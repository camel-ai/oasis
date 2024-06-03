# A Friendly Readme
A Friendly Tutorial for Setting Up Your LLM and Agent Society Simulator,(especially if you're not a master of shell/brew/pip or coding itself)

## before-Quickstart 

### English edition

We assume that you "can find and open the terminal". Please first briefly check the following items:
- homebrew
- pip
- python
- `./.zshrc`
- `./bash_profile`

If you have only heard of some of the concepts above, or even have no idea what they are, please jump to the file below,
where I will tell you how to set up an AI/simulator experimental environment and install some common tools.
[check_files](check_files-en.md)

### 中文版

我们默认你“能找到并打开terminal”, 请你先简单检查一下这些内容
- homebrew
- pip
- python
- `./.zshrc`
- `./bash_profile`

如果你只是听说过上面的某些概念，甚至完全不知道这都是什么，请先跳转到下面的文件，
我将告诉你如何配置一个 AI/simulator 的实验环境，以及安装一些常用的工具
[check_files](check_files-cn.md)


## Quickstart

### Step 1: Create and Activate a Virtual Environment
For win:
```bash
python -m venv env       # Create virtual environment
env\Scripts\activate  # Activate virtual environment
```
For macOS:
```bash
python -m venv env       # Create virtual environment
source env/bin/activate  # Activate virtual environment
```

You can also create a different virtual environment name like `camel_ss`
BUT you'd better add `camel_ss/` into your `.gitignore` file to avoid accidentally committing your virtual environment files to the repository.


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

## pytest

For win:
```bash
pytest test
```

For Mac OS X:
```bash
pytest
```
