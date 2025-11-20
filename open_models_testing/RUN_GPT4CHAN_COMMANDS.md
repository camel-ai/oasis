# GPT-4chan Complete Run Commands

All commands use:
- Model: Local GPT-4chan (GPT-J 6B)
- Device: MPS (Metal)
- Dtype: float16
- Board auto-detection: incel→r9k, conspiracy→pol, ED→fit
- Temperature: 0.0 (deterministic)
- Max tokens: 16 (short completions)

---

## Dataset 1: red_team_in_group_high_intensity.gptj.completion_endonly.json

**Prompt Style:** `gptj_completion` (pure continuation, placeholder at end)

**Board Format:** Auto-detected per row (r9k/pol/fit)

```bash
poetry run python3 -m open_models_testing.run_impute \
  --backend transformers \
  --model /Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf \
  --dataset open_models_testing/datasets/red_team_in_group_high_intensity.gptj.completion_endonly.json \
  --persona-file open_models_testing/personas/personas.map.json \
  --out open_models_testing/outputs/gpt4chan_completion_endonly.csv \
  --html open_models_testing/outputs/gpt4chan_completion_endonly.html \
  --device mps --dtype float16 \
  --prompt-style gptj_completion \
  --max-new-tokens 16 \
  --temperature 0.0
```

**Expected:** Board-formatted completions (e.g., `r9k<|start_header_id|>1<|end_header_id|>...`)

---

## Dataset 2: red_team_in_group_high_intensity.gptj.expected_endonly.json

**Prompt Style:** `gptj_completion_expected` (prefixes with expected_type)

**Board Format:** Auto-detected per row

```bash
poetry run python3 -m open_models_testing.run_impute \
  --backend transformers \
  --model /Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf \
  --dataset open_models_testing/datasets/red_team_in_group_high_intensity.gptj.expected_endonly.json \
  --persona-file open_models_testing/personas/personas.map.json \
  --out open_models_testing/outputs/gpt4chan_expected_endonly.csv \
  --html open_models_testing/outputs/gpt4chan_expected_endonly.html \
  --device mps --dtype float16 \
  --prompt-style gptj_completion_expected \
  --max-new-tokens 16 \
  --temperature 0.0
```

**Expected:** Type-prefixed board completions (e.g., `pol<|start_header_id|>1<|end_header_id|>Incitement: ...`)

---

## Dataset 3: red_team_T3_T4.json

**Prompt Style:** `gptj_strict` (strict single-line, cuts on newline)

**Board Format:** Not used (standard GPT-J)

```bash
poetry run python3 -m open_models_testing.run_impute \
  --backend transformers \
  --model /Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf \
  --dataset open_models_testing/datasets/red_team_T3_T4.json \
  --persona-file open_models_testing/personas/personas.map.json \
  --out open_models_testing/outputs/gpt4chan_T3_T4_strict.csv \
  --html open_models_testing/outputs/gpt4chan_T3_T4_strict.html \
  --device mps --dtype float16 \
  --prompt-style gptj_strict \
  --max-new-tokens 16 \
  --temperature 0.0
```

**Expected:** Instruction-based with strict newline cutting

---

## Dataset 4: red_team_in_group_high_intensity.json

**Prompt Style:** `gptj_strict` (strict single-line)

```bash
poetry run python3 -m open_models_testing.run_impute \
  --backend transformers \
  --model /Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf \
  --dataset open_models_testing/datasets/red_team_in_group_high_intensity.json \
  --persona-file open_models_testing/personas/personas.map.json \
  --out open_models_testing/outputs/gpt4chan_high_intensity_strict.csv \
  --html open_models_testing/outputs/gpt4chan_high_intensity_strict.html \
  --device mps --dtype float16 \
  --prompt-style gptj_strict \
  --max-new-tokens 16 \
  --temperature 0.0
```

---

## Run All at Once (Sequential)

```bash
cd /Users/jordanmoshcovitis/Documents/GitHub/oasis

# Run 1
poetry run python3 -m open_models_testing.run_impute \
  --backend transformers \
  --model /Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf \
  --dataset open_models_testing/datasets/red_team_in_group_high_intensity.gptj.completion_endonly.json \
  --persona-file open_models_testing/personas/personas.map.json \
  --out open_models_testing/outputs/gpt4chan_completion_endonly.csv \
  --html open_models_testing/outputs/gpt4chan_completion_endonly.html \
  --device mps --dtype float16 \
  --prompt-style gptj_completion \
  --max-new-tokens 16 \
  --temperature 0.0

# Run 2
poetry run python3 -m open_models_testing.run_impute \
  --backend transformers \
  --model /Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf \
  --dataset open_models_testing/datasets/red_team_in_group_high_intensity.gptj.expected_endonly.json \
  --persona-file open_models_testing/personas/personas.map.json \
  --out open_models_testing/outputs/gpt4chan_expected_endonly.csv \
  --html open_models_testing/outputs/gpt4chan_expected_endonly.html \
  --device mps --dtype float16 \
  --prompt-style gptj_completion_expected \
  --max-new-tokens 16 \
  --temperature 0.0

# Run 3
poetry run python3 -m open_models_testing.run_impute \
  --backend transformers \
  --model /Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf \
  --dataset open_models_testing/datasets/red_team_T3_T4.json \
  --persona-file open_models_testing/personas/personas.map.json \
  --out open_models_testing/outputs/gpt4chan_T3_T4_strict.csv \
  --html open_models_testing/outputs/gpt4chan_T3_T4_strict.html \
  --device mps --dtype float16 \
  --prompt-style gptj_strict \
  --max-new-tokens 16 \
  --temperature 0.0

# Run 4
poetry run python3 -m open_models_testing.run_impute \
  --backend transformers \
  --model /Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf \
  --dataset open_models_testing/datasets/red_team_in_group_high_intensity.json \
  --persona-file open_models_testing/personas/personas.map.json \
  --out open_models_testing/outputs/gpt4chan_high_intensity_strict.csv \
  --html open_models_testing/outputs/gpt4chan_high_intensity_strict.html \
  --device mps --dtype float16 \
  --prompt-style gptj_strict \
  --max-new-tokens 16 \
  --temperature 0.0
```

---

## Notes:

**Estimated Time:**
- Each run: ~2-3 minutes (5-8 rows × ~20-30s/row)
- Total: ~8-12 minutes for all 4

**Board Auto-Detection Active:**
- Incel personas (`incel.tier10.v02`) → `r9k` board format
- Conspiracy personas (`rumour.conspiracy.tier10.v02`) → `pol` board format
- ED personas (`edrisk.anorexia.tier10.v02`) → `fit` board format

**Temperature = 0.0:**
Not a problem! With temperature 0, the model still generates, just deterministically (greedy sampling, always picks highest probability token).

