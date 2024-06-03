# check_files

这里是中文版

## 简介

主要是下面这些内容
我试图写一份教程，让你能够使用 命令行配置好一切
因为命令行是一个比GUI更好用（而且更通用）的方式，使用GUI可以还需要一些背景知识让你把教程中的内容和GUI的图案对应起来
而命令行只需要 `copy` & `paste`


### 1. Homebrew (brew)

#### 1.1 检查 Homebrew 是否安装
Homebrew 是 macOS 的包管理器，用于安装和管理软件包。要检查 Homebrew 是否已安装以及其版本，可以使用以下命令：
```bash
brew --version
```
如果 Homebrew 未安装，此命令会返回错误信息。

#### 如何安装 Homebrew

如果你发现 Homebrew 未安装在你的系统上，你可以按照以下步骤进行安装：

1. 打开你的终端。
2. 粘贴以下命令到终端中，并按回车执行：
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   这个命令会下载并运行 Homebrew 的安装脚本。

3. 安装过程中，可能会提示你输入系统密码，以便安装必要的工具。

4. 完成安装后，你可能需要添加 Homebrew 到你的 PATH 环境变量中。安装脚本通常会提供相应的指示。

5. 安装完成后，重新打开终端窗口，然后运行 `brew --version` 来确认 Homebrew 是否安装成功。

通过以上步骤，你应该能够成功安装 Homebrew，并开始使用它来管理 macOS 上的软件包。



### 2. 安装 Python 和 pip

#### 2.1 检查 Python 和 pip 是否已安装
首先，我们检查是否已经安装了指定版本的 Python 和 pip。可以使用以下命令来确认：
(PS: 目前`camel-ai`等pkg不支持 python>=3.12，其他一些pip pkg 不支持 python<=3.9)

```bash
python3.10 --version
python3.11 --version
pip3 --version
```

这些命令会显示已安装的 Python 和 pip 版本。

#### 2.2 使用 Homebrew 安装 Python
如果你需要安装 Python 3.10 或 3.11，可以使用 Homebrew 来进行安装。Homebrew 同时也会安装与 Python 匹配的 pip。按照下面的步骤来进行安装：

1. **安装 Python 3.10**：
   ```bash
   brew install python@3.10
   ```
2. **安装 Python 3.11**：
   ```bash
   brew install python@3.11
   ```

安装完成后，你可能需要在你的 Shell 配置文件中添加 Python 的路径，以确保可以直接使用 `python3.10` 或 `python3.11` 命令。例如，对于使用 Zsh 的用户：

```bash
echo 'export PATH="/usr/local/opt/python@3.10/bin:$PATH"' >> ~/.zshrc
echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

使用 `python3.10 --version` 和 `python3.11 --version` 来确认是否正确安装了 Python。同时，`pip3 --version` 可以用来检查 pip 的安装和版本。

#### 2.3 更新 pip
为了确保使用最新版本的 pip，可以用以下命令更新它：

```bash
pip3 install --upgrade pip
```

通过这些步骤，你可以在 macOS 上安装指定版本的 Python，并确保 pip 也得到安装和更新。

### 3. 配置环境变量和别名

#### 3.1 编辑 Shell 配置文件
为了简化日常操作，你可以在 Shell 配置文件中设置别名和环境变量：

- **如果使用 Zsh**：
  ```bash
  nano ~/.zshrc
  ```
- **如果使用 Bash**：
  ```bash
  nano ~/.bash_profile
  ```
如果用的是上面的 `nano` 命令，保存并退出编辑器（`Ctrl + X`, 然后 `Y`, 最后按 `Enter`）。

或者也可以直接使用 open 命令，

- **如果使用 Zsh**：
  ```bash
  open ~/.zshrc
  ```
- **如果使用 Bash**：
  ```bash
  open ~/.bash_profile
  ```
修改以后记得保存(save)（保存不等于激活，激活请见 [3.3-应用新配置](./check_files.md/#33-应用新配置)

#### 3.2 添加配置
在文件中添加以下内容，以便更快捷地使用 Python 和 pip，以及配置其他可能需要的环境变量：

```bash
# 设置别名
alias py3.10='python3.10'
alias py3.11='python3.11'
# 或者你可以习惯用`python` 启动某个常用版本的 python, 比如python3.10
alias python='python3.10'
alias pip='pip3'

# 例如，配置某些 API，根据需要替换 'your_api_key_here'
export SOME_API_KEY='your_api_key_here'
# 配置OpenAI API KEY
export OPENAI_API_KEY='your_api_key_here'
```

#### 3.3 应用新配置
重新加载你的配置文件，以使更改生效：

- **如果使用 Zsh**：
  ```bash
  source ~/.zshrc
  ```
- **如果使用 Bash**：
  ```bash
  source ~/.bash_profile
  ```

完成这些步骤后，你的系统就配置好了使用指定版本的 Python 和 pip，以及必要的环境变量和别名，帮助你更有效地使用命令行工具。




### 4. 下载本项目的GitHub 仓库

使用 `git` 克隆项目仓库。首先，确保你已经安装了 `git`。如果没有，可以通过Homebrew安装：
```bash
brew install git
```

然后克隆项目仓库
```bash
git clone https://github.com/camel-ai/social-simulation/tree/main
cd social-simulation/
```

