source camel-ss/Scripts/activate
# 设置Hugging Face访问令牌环境变量
export HF_HOME=hf_home
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_METRICS_CACHE=$HF_HOME
export HF_MODULES_CACHE=$HF_HOME
export HF_TRANSFORMERS_CACHE=$HF_HOME
export HF_TOKEN="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"

pip install huggingface-hub


# 设置Hugging Face令牌和模型下载目录
HF_TOKEN="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"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
LOCAL_DIR="hf_home"


# 使用 huggingface-cli 下载模型
huggingface-cli download --resume-download "$MODEL_NAME" --local-dir "$LOCAL_DIR" --local-dir-use-symlinks False --resume-download --token "$HF_TOKEN"
