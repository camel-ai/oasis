"""Download all models from models.impute.accessible.json with progress tracking.

This pre-downloads models to the HuggingFace cache (~/.cache/huggingface/)
so they don't need to be downloaded during imputation runs.

Usage:
    poetry run python3 open_models_testing/download_models.py
    
    # Or download specific models:
    poetry run python3 open_models_testing/download_models.py --models "v2ray/GPT4chan-8B" "google-t5/t5-large"
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))


def _human_size(bytes_val: float) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def _print_step(msg: str) -> None:
    """Print a step message."""
    print(f"\n{'='*70}")
    print(msg)
    print('='*70)


def download_model(model_id: str, is_local: bool = False) -> dict:
    """Download a single model with progress tracking.
    
    Returns dict with status, error, and details.
    """
    if is_local:
        print(f"✓ Skipping local model: {model_id}")
        return {"status": "skipped", "reason": "local model"}
    
    print(f"\n[Downloading] {model_id}")
    print(f"  Model ID: {model_id}")
    
    result = {"status": "unknown", "error": None, "details": {}}
    
    try:
        from huggingface_hub import snapshot_download
        from tqdm import tqdm

        # Determine model type for appropriate class
        is_t5 = "t5" in model_id.lower()
        
        print(f"  Type: {'T5 (encoder-decoder)' if is_t5 else 'Causal LM (decoder-only)'}")
        print(f"  Destination: ~/.cache/huggingface/hub/")
        print()
        
        start_time = time.time()
        
        # Download with progress
        print("  Downloading files...")
        cache_dir = snapshot_download(
            repo_id=model_id,
            resume_download=True,
            local_files_only=False,
            # tqdm callback is automatic via huggingface_hub
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n  ✓ Download complete!")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"  Cache location: {cache_dir}")
        
        # Get cache size
        cache_path = Path(cache_dir)
        total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
        print(f"  Total size: {_human_size(total_size)}")
        
        result["status"] = "success"
        result["details"]["cache_dir"] = cache_dir
        result["details"]["size"] = _human_size(total_size)
        result["details"]["time_seconds"] = elapsed
        
    except KeyboardInterrupt:
        print("\n  ✗ Download interrupted by user")
        result["status"] = "interrupted"
        result["error"] = "User interrupted"
        raise
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n  ✗ Download failed: {error_msg}")
        result["status"] = "error"
        result["error"] = error_msg
    
    return result


def load_models_from_config(config_path: Path) -> List[dict]:
    """Load model list from config file."""
    config = json.loads(config_path.read_text())
    return config.get("impute", [])


def main():
    parser = argparse.ArgumentParser(description="Download models for open_models_testing")
    parser.add_argument(
        "--config",
        type=str,
        default="open_models_testing/models.impute.accessible.json",
        help="Path to models config JSON"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific model IDs to download (overrides config)"
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        default=True,
        help="Skip local models (default: True)"
    )
    args = parser.parse_args()
    
    _print_step("Model Download Manager")
    
    # Load models list
    if args.models:
        print(f"\n[Source] Command-line arguments")
        model_specs = [{"backend": "transformers", "model": m} for m in args.models]
    else:
        config_path = Path(args.config)
        print(f"\n[Source] Config file: {config_path}")
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return 1
        model_specs = load_models_from_config(config_path)
    
    print(f"[Total] {len(model_specs)} models to process")
    
    # Filter models
    models_to_download = []
    skipped = []
    
    for spec in model_specs:
        model_id = spec["model"]
        is_local = model_id.startswith("/") or Path(model_id).exists()
        
        if is_local and args.skip_local:
            skipped.append(model_id)
        else:
            models_to_download.append({"id": model_id, "is_local": is_local})
    
    if skipped:
        print(f"\n[Skipped] {len(skipped)} local models:")
        for m in skipped:
            print(f"  - {m}")
    
    if not models_to_download:
        print("\n✓ No remote models to download!")
        return 0
    
    print(f"\n[Downloading] {len(models_to_download)} remote models")
    print("\nThis will download models to: ~/.cache/huggingface/hub/")
    print("Estimated total download size: ~50-60 GB (first run only)")
    print("\nNote: Downloads can be resumed if interrupted.\n")
    
    # Confirm
    try:
        response = input("Continue? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("\nDownload cancelled.")
            return 0
    except (EOFError, KeyboardInterrupt):
        print("\nDownload cancelled.")
        return 0
    
    # Download each model
    results = []
    for i, model_info in enumerate(models_to_download, 1):
        _print_step(f"Model {i}/{len(models_to_download)}")
        result = download_model(model_info["id"], model_info["is_local"])
        results.append({
            "model": model_info["id"],
            **result
        })
    
    # Summary
    _print_step("Download Summary")
    
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    interrupted = sum(1 for r in results if r["status"] == "interrupted")
    
    print(f"\nTotal models: {len(results)}")
    print(f"✓ Downloaded: {success}")
    print(f"✗ Errors: {errors}")
    print(f"⚠ Interrupted: {interrupted}")
    
    if errors > 0:
        print("\nFailed downloads:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['model']}: {r['error']}")
    
    if success > 0:
        total_time = sum(r["details"].get("time_seconds", 0) for r in results if r["status"] == "success")
        print(f"\nTotal download time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        print("\n✓ Models cached and ready for imputation!")
        print("\nNext step:")
        print("  poetry run python3 open_models_testing/run_impute.py \\")
        print("    --dataset open_models_testing/datasets/red_team_T3_T4.json \\")
        print("    --persona-file open_models_testing/personas/personas.map.json \\")
        print("    --out open_models_testing/outputs/results.csv \\")
        print("    --html open_models_testing/outputs/results.html \\")
        print("    --models-config open_models_testing/models.impute.accessible.json \\")
        print("    --device mps --dtype float16 --max-new-tokens 64 | cat")
    
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n✗ Download interrupted by user.")
        sys.exit(1)

