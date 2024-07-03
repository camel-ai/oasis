from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


# 函数：处理每个批次
def process_batch(model, tokenizer, batch_texts):
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output

def generate_tweet_vector(model, tokenizer, texts, batch_size):
    # 循环处理所有消息
    # 如果消息列表过大，采用batch的方式去处理。
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_outputs = process_batch(model, tokenizer, batch_texts)
        all_outputs.append(batch_outputs)
    all_outputs_tensor = torch.cat(all_outputs, dim=0) # num_tweets x dimension 
    return all_outputs_tensor.cpu()

if __name__ == "__main__":
    # 输入的字符串列表（假设有上万条消息）
    texts = ["I'm using TwHIN-BERT! #TwHIN-BERT #NLP"] * 10000  # 这里用相同的消息重复10000次作为示例
    # 定义批量大小
    batch_size = 100
    all_outputs_tensor = generate_tweet_vector(texts, batch_size)
    print(all_outputs_tensor.shape)
