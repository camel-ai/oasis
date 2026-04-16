# Short-Video Negative Feedback Experiment

This directory contains a paper-style experiment template for studying negative feedback in the TikTok-style short-video simulator.

## Research Question

Does stronger explicit negative feedback (`not_interested`) reduce the downstream reach of a low-quality video under the traffic-pool recommender?

## What The Experiment Does

- Creates a deterministic short-video population with 3 creators and 6 viewers
- Uploads three videos with different quality and audience fit
- Runs two conditions:
  - `baseline`: the low-quality comedy video receives limited negative feedback
  - `treatment`: the same video receives additional `not_interested` actions
- Saves two SQLite databases that can be inspected with the built-in observability and visualization tools

## Run It

```bash
python examples/experiment/short_video_negative_feedback/run_experiment.py
```

By default, outputs are written to:

```text
examples/experiment/short_video_negative_feedback/output
```

## Run Stronger Multi-Run Experiments

To improve statistical strength, run paired baseline/treatment experiments with
seeded perturbations:

```bash
python examples/experiment/short_video_negative_feedback/run_experiment.py \
  --output-dir examples/experiment/short_video_negative_feedback/output \
  --n-runs 30 \
  --seed-start 2026 \
  --watch-jitter 0.08 \
  --behavior-flip-prob 0.05 \
  --treatment-extra-neg-prob 0.85
```

This will additionally write:

- `multirun_metrics.csv` (per-run metrics)
- `multirun_summary.csv` (mean/std/95% CI and paired sign-flip p-values)
- `MULTIRUN_REPORT.md` (analysis notes)

Core OASIS short-video metrics now include:

- `retention_3s_rate`: fraction of watch events with watched seconds >= 3
- `negative_feedback_rate`: `total_negative_feedback / total_views`
- `creator_coverage`: share of active creators receiving at least one view

## Add Control Strategies (Four-Arm)

To add stronger controls, run with four conditions:

- `baseline`: weak negative feedback
- `treatment`: stronger targeted negative feedback
- `none`: disable negative feedback entirely
- `random`: inject random negative feedback at a fixed probability

```bash
python examples/experiment/short_video_negative_feedback/run_experiment.py \
  --output-dir examples/experiment/short_video_negative_feedback/output_four_arm \
  --n-runs 30 \
  --strategy-set four_arm \
  --random-negative-prob 0.30 \
  --watch-jitter 0.08 \
  --behavior-flip-prob 0.05 \
  --treatment-extra-neg-prob 0.85
```

Additional outputs:

- `strategy_multirun_metrics.csv`
- `strategy_multirun_summary.csv`
- `strategy_pairwise_vs_baseline.csv`

## Scale Up Population and Content

You can increase creator/viewer/video scale by multipliers:

```bash
python examples/experiment/short_video_negative_feedback/run_experiment.py \
  --output-dir examples/experiment/short_video_negative_feedback/output_large \
  --n-runs 30 \
  --creator-multiplier 3 \
  --viewer-multiplier 5 \
  --video-multiplier 4 \
  --watch-jitter 0.08 \
  --behavior-flip-prob 0.05 \
  --treatment-extra-neg-prob 0.85
```

## Run Batch Matrix Experiments (Large-Scale Parameter Sweep)

```bash
python examples/experiment/short_video_negative_feedback/run_batch_experiments.py \
  --output-dir examples/experiment/short_video_negative_feedback/batch_output \
  --n-runs 30 \
  --strategy-set four_arm \
  --creator-multiplier 3 \
  --viewer-multiplier 5 \
  --video-multiplier 4 \
  --random-negative-prob 0.30 \
  --watch-jitter-values 0.04,0.08,0.12 \
  --behavior-flip-values 0.02,0.05 \
  --treatment-extra-neg-values 0.70,0.85,1.00
```

Batch outputs include:

- `batch_config_summary.csv` (one row per configuration)
- `batch_report.md` (top configurations by significance)
- per-config subdirectories, each containing full single/multi-run outputs

For a fast smoke check:

```bash
python examples/experiment/short_video_negative_feedback/run_batch_experiments.py --quick --n-runs 3
```

## Analyze Results

```bash
python visualization/short_video_simulation/code/generate_report.py \
  examples/experiment/short_video_negative_feedback/output/negative_feedback_baseline.db \
  --output examples/experiment/short_video_negative_feedback/output/baseline_report.md

python visualization/short_video_simulation/code/generate_report.py \
  examples/experiment/short_video_negative_feedback/output/negative_feedback_treatment.db \
  --output examples/experiment/short_video_negative_feedback/output/treatment_report.md

python visualization/short_video_simulation/code/compare_runs.py \
  baseline=examples/experiment/short_video_negative_feedback/output/negative_feedback_baseline.db \
  treatment=examples/experiment/short_video_negative_feedback/output/negative_feedback_treatment.db \
  --output-dir examples/experiment/short_video_negative_feedback/output/comparison
```

## Expected Pattern

The treatment run should push the low-quality comedy video toward a lower `traffic_pool_level`, while the higher-quality dance and tech videos remain comparatively stable.
