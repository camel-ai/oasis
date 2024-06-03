#!/bin/bash

# 检测并处理 Homebrew 的安装
function check_brew {
    echo "---------检测并处理 Homebrew 的安装--------"
    if type brew > /dev/null 2>&1; then
        echo "Homebrew 已安装，版本信息如下："
        brew --version
    else
        echo "Homebrew 未安装。是否现在安装？(yes/no)"
        read install_brew
        if [ "$install_brew" = "yes" ]; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
            source ~/.zshrc
            echo "Homebrew 安装完成。"
        fi
    fi
}

# 检测并处理 Python 和 pip 的安装与版本
function check_python {
    echo "---------检测并处理 Python 和 pip 的安装与版本--------"
    local needed_version1="3.10"
    local needed_version2="3.11"
    local pythons_installed=$(compgen -c python | grep -E '^python3\.1[01](\.[0-9]+)?$' | sort -u)

    echo "目前推荐使用 Python $needed_version1 or $needed_version2"
    if [[ ! -z "$pythons_installed" ]]; then
        echo "本地安装的 Python 版本包括："
        for py in $pythons_installed; do
            $py --version
        done
    else
        echo "未检测到已安装的 Python 版本。"
    fi

    echo "是否尝试安装/升级 Python？(yes/no)"
    read update_python
    if [ "$update_python" = "yes" ]; then
        echo "请选择要安装的 Python 版本：$needed_version1 或 $needed_version2 (输入版本号)"
        read python_version
        if [[ "$python_version" == "$needed_version1" || "$python_version" == "$needed_version2" ]]; then
            if type brew > /dev/null 2>&1; then
                brew install python@$python_version
                echo 'export PATH="/usr/local/opt/python@'$python_version'/bin:$PATH"' >> ~/.zshrc
                source ~/.zshrc
                echo "Python $python_version 安装/升级完成。"
            else
                echo "Homebrew 未安装，无法安装 Python。"
            fi
        else
            echo "输入的版本号不正确。"
        fi
    fi

    # 检查 pip 是否安装，并尝试升级
    if type pip3 > /dev/null 2>&1; then
        echo "pip 已安装，版本信息如下："
        pip3 --version
        echo "是否升级 pip？(yes/no)"
        read update_pip
        if [ "$update_pip" = "yes" ]; then
            pip3 install --upgrade pip
            echo "pip 升级完成。"
        fi
    else
        echo "pip 未安装。正在安装最新版本的 pip..."
        if type brew > /dev/null 2>&1; then
            brew install python@3.10  # 这将自动安装与 Python 匹配的 pip
            echo "pip 安装完成。"
        else
            echo "Homebrew 未安装，无法安装 pip。"
        fi
    fi
}

# 执行检查
check_brew
check_python

echo "基础环境配置检查完成。"
