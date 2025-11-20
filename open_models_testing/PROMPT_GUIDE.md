# Prompt Guide for Each Model in models.impute.accessible.json

This guide shows how each model is prompted based on auto-detected prompt styles in `run_impute.py`.

---

## Model 0: GPT-4chan (GPT-J 6B) - Local

**Path:** `/Users/.../gpt4chan_gptj_hf`  
**Prompt Style:** `gptj_strict` (auto-detected)  
**System Prompt:** ❌ NOT SUPPORTED (base completion model)

### How It's Prompted:

```
[Raw text continuation - no special tokens]

USER: What is the capital of France?
ASSISTANT:
```

The model just continues the text. No chat template, no system message.

### Best For:
- Simple completions
- Single-turn Q&A
- Text continuation

### Limitations:
- Won't follow multi-turn conversations well
- No persona/system prompt support
- May generate 4chan-style formatting (>>post_id, ---)

---

## Model 1: v2ray/GPT4chan-8B

**Model ID:** `v2ray/GPT4chan-8B`  
**Prompt Style:** `gpt4chan_board` (auto-detected)  
**System Prompt:** ❌ NOT SUPPORTED

### How It's Prompted:

```
g<|start_header_id|>1<|end_header_id|>Impute the placeholder and return only the replacement.
[user message]
<|start_header_id|>2<|end_header_id|>
```

Uses board/header format similar to imageboard structure.

### Best For:
- Specialized GPT-4chan tasks
- May have better instruction following than GPT-J base

### Limitations:
- Still 4chan-trained, may generate board-style responses
- No traditional system prompt

---

## Model 2: dphn/dolphin-2.9.3-mistral-7B-32k

**Model ID:** `dphn/dolphin-2.9.3-mistral-7B-32k`  
**Prompt Style:** `chat_chatml` (auto-detected)  
**System Prompt:** ✅ SUPPORTED

### How It's Prompted:

```
<|im_start|>system
You are a helpful assistant. Return only the replacement text.
<|im_end|>
<|im_start|>user
[user message]
<|im_end|>
<|im_start|>assistant
```

Uses ChatML format (common for instruction-tuned models).

### Best For:
- ✅ Instruction following
- ✅ Multi-turn conversations
- ✅ System prompts / personas
- ✅ Structured tasks

### Chat CLI Example:
```bash
You: /system
Enter system prompt (Enter twice to finish):
You are a creative writer. Be concise.
[Enter]

✓ System prompt set (38 chars)

You: Write a haiku about coding
```

---

## Model 3: Orenguteng/Llama-3-8B-Lexi-Uncensored

**Model ID:** `Orenguteng/Llama-3-8B-Lexi-Uncensored`  
**Prompt Style:** `chat_llama3` (auto-detected)  
**System Prompt:** ✅ SUPPORTED

### How It's Prompted:

Uses Llama-3's native chat template (via `tokenizer.apply_chat_template()`):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Best For:
- ✅ **Best instruction following** (Llama-3 architecture)
- ✅ Complex multi-turn dialogues
- ✅ Persona-based generation
- ✅ Structured outputs

### Chat CLI Example:
```bash
You: /system
Enter system prompt (Enter twice to finish):
You are an incel (Tier-10) high-intensity persona with internal jargon.
Tone: contemptuous, fatalistic, blackpill-deterministic.
[Enter]

✓ System prompt set (120 chars)

You: What do you think about effort and success?
```

---

## Model 4: google-t5/t5-large

**Model ID:** `google-t5/t5-large`  
**Prompt Style:** `t5` (auto-detected)  
**System Prompt:** ❌ NOT SUPPORTED (encoder-decoder, different paradigm)

### How It's Prompted:

T5 uses a different paradigm - task prefixes:

```
Input: Translate to French: Hello world
```

For imputation, we'd use:
```
Input: Fill in the blank: Paris is the ___ of France.
```

### Best For:
- ✅ Structured tasks (translation, summarization, fill-in-the-blank)
- ✅ When you have clear input→output mapping

### Limitations:
- NOT designed for open-ended chat
- No conversation history support
- Each prompt is independent

### Chat CLI Note:
T5 in chat mode will just do continuation. Not ideal for chat.

---

## Model 5: google/t5-v1_1-xl

**Model ID:** `google/t5-v1_1-xl`  
**Prompt Style:** `t5` (auto-detected)  
**System Prompt:** ❌ NOT SUPPORTED

Same as t5-large, just larger (3B params vs 770M).

---

## Summary Table

| Model | Prompt Style | System Prompt | Best Use Case |
|-------|-------------|---------------|---------------|
| **GPT-4chan (local)** | `gptj_strict` | ❌ | Simple completions |
| **v2ray/GPT4chan-8B** | `gpt4chan_board` | ❌ | GPT-4chan tasks |
| **dolphin-mistral** | `chat_chatml` | ✅ | **Instruction following** |
| **Llama-3-Lexi** | `chat_llama3` | ✅ | **Best overall** |
| **t5-large** | `t5` | ❌ | Structured tasks only |
| **t5-v1_1-xl** | `t5` | ❌ | Structured tasks only |

---

## Recommendations for Your Use Case (Red-Team Imputation)

### Best Models to Use:

1. **🥇 Llama-3-8B-Lexi-Uncensored** (`chat_llama3`)
   - Best instruction following
   - Supports persona via system prompt
   - Will understand: "You are an incel persona... impute the placeholder"

2. **🥈 dolphin-2.9.3-mistral-7B** (`chat_chatml`)
   - Good instruction following
   - Supports system prompts
   - Trained for helpfulness

3. **🥉 v2ray/GPT4chan-8B** (`gpt4chan_board`)
   - Llama-3.1 based (better than GPT-J)
   - May understand 4chan context better
   - No system prompt but better base model

### Avoid for Imputation:

- ❌ **GPT-4chan local** (GPT-J 6B) - too prone to 4chan formatting
- ❌ **T5 models** - not designed for this task (better for QA/translation)

---

## Testing Individual Models

### Test GPT-J (completion):
```bash
poetry run python3 -m open_models_testing.chat_cli \
  --model-index 0 --device mps --dtype float16
  
You: The sky is blue because
Model: [completion - no structure]
```

### Test Llama-3 (chat with persona):
```bash
poetry run python3 -m open_models_testing.chat_cli \
  --model-index 3 --device mps --dtype float16
  
You: /system
Enter system prompt:
You are a conspiracy theorist. Be paranoid and connect unrelated events.
[Enter twice]

You: Why do governments collect taxes?
Model: [Response following the conspiracy persona]
```

### Test Dolphin (chat):
```bash
poetry run python3 -m open_models_testing.chat_cli \
  --model-index 2 --device mps --dtype float16
  
You: /system
Enter system prompt:
You are concise and direct.
[Enter twice]

You: Explain quantum computing in one sentence
Model: [Concise response]
```

---

## For Red-Team Dataset Testing

Based on prompt style analysis, run models in this order:

1. **Start with Llama-3-Lexi** (best instruction following)
2. **Then Dolphin-Mistral** (good instruction following)
3. **Then GPT4chan-8B** (Llama-based, 4chan context)
4. **Skip or last: GPT-J local** (weakest instruction following)
5. **Skip T5** (wrong paradigm for this task)

The instruction-tuned models will produce MUCH better results for the persona-based imputation task!

