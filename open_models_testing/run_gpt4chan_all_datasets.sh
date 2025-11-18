#!/usr/bin/env bash
# Run GPT-4chan local model over all GPT-J optimized datasets with board auto-detection

set -euo pipefail

BASE_DIR="/Users/jordanmoshcovitis/Documents/GitHub/oasis"
MODEL_PATH="$BASE_DIR/open_models_testing/models/gpt4chan_gptj_hf"
PERSONA_FILE="$BASE_DIR/open_models_testing/personas/personas.map.json"
OUT_DIR="$BASE_DIR/open_models_testing/outputs"

# GPT-J specific datasets
DATASETS=(
  "red_team_in_group_high_intensity.gptj.completion_endonly.json:gptj_completion"
  "red_team_in_group_high_intensity.gptj.expected_endonly.json:gptj_completion_expected"
  "red_team_T3_T4.json:gptj_strict"
  "red_team_in_group_high_intensity.json:gptj_strict"
)

echo "======================================================================" | cat
echo "GPT-4CHAN MODEL - ALL DATASETS WITH BOARD AUTO-DETECTION" | cat
echo "======================================================================" | cat
echo "Model: GPT-4chan local (6B)" | cat
echo "Device: MPS (Metal)" | cat
echo "Dtype: float16" | cat
echo "Board detection: AUTO (incel→r9k, conspiracy→pol, ED→fit)" | cat
echo "======================================================================" | cat

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r dataset_file prompt_style <<< "$spec"
  
  dataset_path="$BASE_DIR/open_models_testing/datasets/$dataset_file"
  
  if [ ! -f "$dataset_path" ]; then
    echo "\n[SKIP] Dataset not found: $dataset_file" | cat
    continue
  fi
  
  # Create output filename
  output_base="${dataset_file%.json}"
  
  echo "\n>>> Dataset: $dataset_file" | cat
  echo "    Prompt style: $prompt_style" | cat
  echo "    Processing..." | cat
  
  poetry run python3 -m open_models_testing.run_impute \
    --backend transformers \
    --model "$MODEL_PATH" \
    --dataset "$dataset_path" \
    --persona-file "$PERSONA_FILE" \
    --out "$OUT_DIR/gpt4chan_${output_base}.csv" \
    --html "$OUT_DIR/gpt4chan_${output_base}.html" \
    --device mps \
    --dtype float16 \
    --prompt-style "$prompt_style" \
    --max-new-tokens 16 \
    --temperature 0.0 | cat
  
  echo "    ✓ Complete" | cat
done

echo "\n======================================================================" | cat
echo "✓ ALL DATASETS COMPLETE" | cat
echo "======================================================================" | cat
echo "Outputs in: $OUT_DIR/gpt4chan_*.csv" | cat

