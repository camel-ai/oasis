# check_files

This is the English version.

## Introduction

The main contents are as follows:
I am trying to write a tutorial to enable you to configure everything using the command line because the command line is a more useful (and more universal) method than GUI. Using a GUI might also require some background knowledge to match the content in the tutorial with the GUI patterns, whereas the command line only needs `copy` & `paste`.

### 1. Homebrew (brew)

#### 1.1 Check if Homebrew is installed
Homebrew is a package manager for macOS, used to install and manage software packages. To check if Homebrew is installed and its version, use the following command:
```bash
brew --version
```
If Homebrew is not installed, this command will return an error message.

#### How to install Homebrew

If you find that Homebrew is not installed on your system, you can install it by following these steps:

1. Open your terminal.
2. Paste the following command into the terminal and press enter:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   This command will download and run the Homebrew installation script.

3. During the installation process, you may be prompted to enter your system password to install necessary tools.

4. After the installation is complete, you may need to add Homebrew to your PATH environment variable. The installation script usually provides the corresponding instructions.

5. After installation, reopen the terminal window and run `brew --version` to confirm if Homebrew is installed successfully.

By following these steps, you should be able to successfully install Homebrew and start using it to manage software packages on macOS.

### 2. Install Python and pip

#### 2.1 Check if Python and pip are installed
First, check if the specified versions of Python and pip are already installed. You can confirm using the following commands:
(PS: Currently, `camel-ai` and other packages do not support Python>=3.12, and some pip packages do not support Python<=3.9)

```bash
python3.10 --version
python3.11 --version
pip3 --version
```

These commands will display the installed versions of Python and pip.

#### 2.2 Install Python using Homebrew
If you need to install Python 3.10 or 3.11, you can use Homebrew for installation. Homebrew will also install the pip that matches with Python. Follow these steps to install:

1. **Install Python 3.10**:
   ```bash
   brew install python@3.10
   ```
2. **Install Python 3.11**:
   ```bash
   brew install python@3.11
   ```

After installation, you might need to add Python's path to your Shell configuration file to ensure that you can directly use the `python3.10` or `python3.11` commands. For example, for users using Zsh:

```bash
echo 'export PATH="/usr/local/opt/python@3.10/bin:$PATH"' >> ~/.zshrc
echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Use `python3.10 --version` and `python3.11 --version` to confirm if Python is correctly installed. Also, `pip3 --version` can be used to check the installation and version of pip.

#### 2.3 Update pip
To ensure you are using the latest version of pip, you can update it with the following command:

```bash
pip3 install --upgrade pip
```

By following these steps, you can install the specified versions of Python on macOS and ensure pip is also installed and updated.

### 3. Configure environment variables and aliases

#### 3.1 Edit the Shell configuration file
To simplify daily operations, you can set aliases and environment variables in your Shell configuration file:

- **If using Zsh**:
  ```bash
  nano ~/.zshrc
  ```
- **If using Bash**:
  ```bash
  nano ~/.bash_profile
  ```
If using the `nano` command mentioned above, save and exit the editor (`Ctrl + X`, then `Y`, and finally press `Enter`).

Or you can also directly use the open command,

- **If using Zsh**:
  ```bash
  open ~/.zshrc
  ```
- **If using Bash**:
  ```bash
  open ~/.bash_profile
  ```
Remember to save after modifications (saving is not the same as activating, see [3.3-Apply New Configuration](./check_files.md/#33-apply-new-configuration) for activation).

#### 3.2 Add configurations
Add the following content to the file to facilitate quicker use of Python and pip, and configure other potentially needed

 environment variables:

```bash
# Set aliases
alias py3.10='python3.10'
alias py3.11='python3.11'
# Or you might prefer to use `python` to launch a commonly used version of Python, like python3.10
alias python='python3.10'
alias pip='pip3'

# For example, configure some APIs, replace 'your_api_key_here' as needed
export SOME_API_KEY='your_api_key_here'
# Configure OpenAI API KEY
export OPENAI_API_KEY='your_api_key_here'
```

#### 3.3 Apply new configuration
Reload your configuration file to make the changes effective:

- **If using Zsh**:
  ```bash
  source ~/.zshrc
  ```
- **If using Bash**:
  ```bash
  source ~/.bash_profile
  ```

After completing these steps, your system will be configured to use the specified versions of Python and pip, as well as necessary environment variables and aliases, helping you use command-line tools more effectively.

### 4. Download this project's GitHub repository

Use `git` to clone the project repository. First, ensure that you have installed `git`. If not, you can install it via Homebrew:
```bash
brew install git
```

Then clone the project repository
```bash
git clone https://github.com/camel-ai/social-simulation/tree/main
cd social-simulation/
```