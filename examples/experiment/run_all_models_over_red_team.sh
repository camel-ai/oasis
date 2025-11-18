#!/usr/bin/env bash
set -euo pipefail

# Run each model explicitly over both red-team datasets using the
# auto-mapped prompt styles in run_impute.py.

# ========== CONFIGURATION ==========
# Toggle device: "mps" for Apple Metal GPU, "cpu" for CPU-only
DEVICE="mps"  # Change to "mps" for GPU acceleration (requires sufficient RAM)

# dtype: "float32" for CPU (better stability), "float16" for MPS (saves memory)
DTYPE="float16"  # Change to "float16" if using MPS

# Note: 8B models need ~16-19 GB RAM. With <16 GB available, use CPU mode.
# ===================================

BASE_DIR="/Users/jordanmoshcovitis/Documents/GitHub/oasis"

DS_T3_T4="$BASE_DIR/open_models_testing/datasets/red_team_T3_T4.json"
DS_HIGH_INTENSITY="$BASE_DIR/open_models_testing/datasets/red_team_in_group_high_intensity.json"

OUT_DIR="$BASE_DIR/open_models_testing/outputs"
if [ ! -d "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
fi

# Declare models explicitly (backend|model_path_or_id)
MODELS=(
#   "transformers|/Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf"
  "transformers|v2ray/GPT4chan-8B"
  "transformers|dphn/dolphin-2.9.3-mistral-7B-32k"
  "transformers|Orenguteng/Llama-3-8B-Lexi-Uncensored"
#   "transformers|google-t5/t5-large"
#   "transformers|google/t5-v1_1-xl"
)

datasets=(
  "$DS_T3_T4"
  "$DS_HIGH_INTENSITY"
)

dataset_labels=(
  "red_team_T3_T4"
  "red_team_in_group_high_intensity"
)

sanitize() {
  # Sanitize model id into a filesystem-friendly label
  local s="$1"
  s="${s//\//__}"
  s="${s//[^A-Za-z0-9._-]/_}"
  echo "$s"
}

run_one() {
  local backend="$1"; shift
  local model_id="$1"; shift
  local dataset_path="$1"; shift
  local ds_label="$1"; shift

  local model_label
  model_label=$(sanitize "$model_id")

  # For local model paths, check existence
  if [[ "$model_id" == /* ]] && [ ! -e "$model_id" ]; then
    echo "[WARN] Skipping missing local model: $model_id" | cat
    return 0
  fi

  # Determine device/dtype for this model
  local device="$DEFAULT_DEVICE"
  local dtype="$DEFAULT_DTYPE"
  
  # Check for model-specific overrides
  for pattern in "${!MODEL_OVERRIDES[@]}"; do
    if [[ "$model_id" == *"$pattern"* ]]; then
      local override="${MODEL_OVERRIDES[$pattern]}"
      device="${override%%:*}"
      dtype="${override##*:}"
      break
    fi
  done

  echo "\n=== Running model: $model_id on $ds_label ===" | cat
  echo "    Device: $device | Dtype: $dtype" | cat
  poetry run python3 -m open_models_testing.run_impute \
    --backend "$backend" \
    --model "$model_id" \
    --dataset "$dataset_path" \
    --out "$OUT_DIR/${ds_label}.${model_label}.csv" \
    --html "$OUT_DIR/${ds_label}.${model_label}.html" \
    --device "$device" \
    --dtype "$dtype" \
    --temperature 0.0 \
    --max-new-tokens 32 | cat
}

for i in "${!datasets[@]}"; do
  ds="${datasets[$i]}"
  ds_label="${dataset_labels[$i]}"
  echo "\n>>> Dataset: $ds_label" | cat
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r backend model_id <<< "$spec"
    run_one "$backend" "$model_id" "$ds" "$ds_label"
  done
done

echo "\nDone. Outputs written to: $OUT_DIR" | cat


