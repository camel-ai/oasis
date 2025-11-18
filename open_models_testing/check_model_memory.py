"""Check if models will fit in available RAM before loading.

Usage:
    # Check all models in config
    poetry run python3 -m open_models_testing.check_model_memory
    
    # Check specific model
    poetry run python3 -m open_models_testing.check_model_memory \
      --model "v2ray/GPT4chan-8B" \
      --dtype float16
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024**3)
    except ImportError:
        # Fallback: try sysctl on Mac
        import subprocess
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True
            )
            total_bytes = int(result.stdout.strip())
            # Assume 70% is available (conservative estimate)
            return (total_bytes * 0.7) / (1024**3)
        except Exception:
            return None


def estimate_model_size(model_id: str, dtype: str = "float16") -> Dict:
    """Estimate model memory requirements.
    
    Returns dict with size_gb, param_count, dtype info.
    """
    result = {"status": "unknown", "size_gb": None, "param_count": None}
    
    # Try to get config from HuggingFace
    try:
        import json as _json

        from huggingface_hub import hf_hub_download

        # Check if local path
        if Path(model_id).exists():
            config_path = Path(model_id) / "config.json"
            if config_path.exists():
                config = _json.loads(config_path.read_text())
            else:
                result["status"] = "no_config"
                return result
        else:
            # Download config (small file)
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                cache_dir=None,
            )
            config = _json.loads(Path(config_path).read_text())
        
        # Extract parameter count from config
        # Common fields: hidden_size, num_hidden_layers, num_attention_heads, etc.
        hidden_size = config.get("hidden_size") or config.get("d_model") or config.get("n_embd", 0)
        num_layers = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers", 0)
        
        if hidden_size and num_layers:
            # Rough estimate: params ≈ 12 * hidden_size^2 * num_layers / 1e9 (for transformer)
            # Better: use explicit param count if available
            param_count_b = None
            
            # Try to find explicit param count in model card
            model_type = config.get("model_type", "")
            
            # Common architectures
            if "llama" in model_type.lower() or "mistral" in model_type.lower():
                # For Llama-3-8B, Mistral-7B: use standard sizes
                if "8b" in model_id.lower() or hidden_size >= 4000:
                    param_count_b = 8.0
                elif "7b" in model_id.lower() or hidden_size >= 3500:
                    param_count_b = 7.0
                else:
                    # Rough formula
                    param_count_b = (12 * (hidden_size ** 2) * num_layers) / 1e9
            
            elif "gptj" in model_type.lower() or "gpt-j" in model_id.lower():
                param_count_b = 6.0
            
            elif "t5" in model_type.lower():
                if "xl" in model_id.lower():
                    param_count_b = 3.0
                elif "large" in model_id.lower():
                    param_count_b = 0.77
                else:
                    param_count_b = 0.22
            
            else:
                # Generic formula
                param_count_b = (12 * (hidden_size ** 2) * num_layers) / 1e9
            
            # Calculate memory based on dtype
            bytes_per_param = {
                "float32": 4,
                "float16": 2,
                "bfloat16": 2,
                "int8": 1,
                "int4": 0.5,
            }.get(dtype, 2)
            
            size_gb = param_count_b * bytes_per_param
            
            # Add overhead (activations, gradients, etc.) - roughly 1.2x
            size_gb_with_overhead = size_gb * 1.2
            
            result["status"] = "estimated"
            result["param_count"] = f"{param_count_b:.1f}B"
            result["size_gb"] = size_gb
            result["size_with_overhead_gb"] = size_gb_with_overhead
            result["dtype"] = dtype
            result["bytes_per_param"] = bytes_per_param
        
        else:
            result["status"] = "insufficient_config"
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def check_model_fits(model_id: str, dtype: str, available_gb: float) -> Dict:
    """Check if model will fit in available memory."""
    est = estimate_model_size(model_id, dtype)
    
    if est["status"] != "estimated":
        return {**est, "fits": "unknown"}
    
    size_needed = est["size_with_overhead_gb"]
    
    # Conservative check: need 1.5x the model size for comfortable operation
    recommended_gb = size_needed * 1.5
    
    if size_needed <= available_gb:
        fits = "yes"
    elif size_needed <= available_gb * 1.3:
        fits = "tight"
    else:
        fits = "no"
    
    return {
        **est,
        "fits": fits,
        "available_gb": available_gb,
        "recommended_gb": recommended_gb,
    }


def main():
    parser = argparse.ArgumentParser(description="Check model memory requirements")
    parser.add_argument("--config", type=str, default="open_models_testing/models.impute.accessible.json")
    parser.add_argument("--model", type=str, help="Check specific model instead of all")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    
    available_gb = get_available_memory_gb()
    
    print("=" * 70)
    print("MODEL MEMORY CHECK")
    print("=" * 70)
    
    if available_gb:
        print(f"\n Available RAM: {available_gb:.1f} GB")
    else:
        print(f"\n ⚠ Could not detect available RAM")
        print("  Install psutil: poetry add --group open_models_testing psutil")
        available_gb = 16.0  # Conservative estimate
        print(f"  Using conservative estimate: {available_gb:.1f} GB")
    
    print(f" Dtype: {args.dtype}")
    print()
    
    # Load models to check
    if args.model:
        models_to_check = [{"backend": "transformers", "model": args.model}]
    else:
        config = json.loads(Path(args.config).read_text())
        models_to_check = config.get("impute", [])
    
    print(f"Checking {len(models_to_check)} model(s)...\n")
    
    results = []
    for i, spec in enumerate(models_to_check):
        model_id = spec.get("model") or spec.get("model_path", "unknown")
        
        print(f"[{i}] {model_id}")
        
        check = check_model_fits(model_id, args.dtype, available_gb)
        
        if check["status"] == "estimated":
            size = check["size_gb"]
            overhead = check["size_with_overhead_gb"]
            fits = check["fits"]
            
            icon = "✓" if fits == "yes" else ("⚠" if fits == "tight" else "✗")
            
            print(f"  Status: {icon} {fits.upper()}")
            print(f"  Params: {check['param_count']}")
            print(f"  Size (weights only): {size:.1f} GB")
            print(f"  Size (with overhead): {overhead:.1f} GB")
            
            if fits == "no":
                print(f"  ✗ WILL NOT FIT - needs {overhead:.1f} GB, have {available_gb:.1f} GB")
                print(f"  → Will offload to disk (VERY SLOW)")
            elif fits == "tight":
                print(f"  ⚠ TIGHT FIT - may use swap")
            else:
                print(f"  ✓ Will fit comfortably")
        
        else:
            print(f"  Status: {check['status']}")
            if "error" in check:
                print(f"  Error: {check['error']}")
        
        print()
        results.append({"model": model_id, **check})
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    will_fit = [r for r in results if r.get("fits") == "yes"]
    tight_fit = [r for r in results if r.get("fits") == "tight"]
    wont_fit = [r for r in results if r.get("fits") == "no"]
    unknown = [r for r in results if r.get("fits") == "unknown"]
    
    print(f"\n✓ Will fit comfortably: {len(will_fit)}")
    for r in will_fit:
        model_name = r["model"].split("/")[-1] if "/" in r["model"] else Path(r["model"]).name
        print(f"  - {model_name}")
    
    if tight_fit:
        print(f"\n⚠ Tight fit (may swap): {len(tight_fit)}")
        for r in tight_fit:
            model_name = r["model"].split("/")[-1] if "/" in r["model"] else Path(r["model"]).name
            print(f"  - {model_name}")
    
    if wont_fit:
        print(f"\n✗ Won't fit (disk offload): {len(wont_fit)}")
        for r in wont_fit:
            model_name = r["model"].split("/")[-1] if "/" in r["model"] else Path(r["model"]).name
            size = r.get("size_with_overhead_gb", "?")
            print(f"  - {model_name} (needs {size:.1f} GB)")
        print("\n  These models will be EXTREMELY SLOW on MPS due to disk offloading.")
        print("  Recommendation: Use --device cpu instead, or reduce model list.")
    
    if unknown:
        print(f"\n? Could not estimate: {len(unknown)}")
    
    print("\n" + "=" * 70)
    
    # Specific recommendation for the hanging issue
    if any(r["model"] == "v2ray/GPT4chan-8B" and r.get("fits") in ("no", "tight") for r in results):
        print("\n🔍 DIAGNOSIS: v2ray/GPT4chan-8B Issue")
        print("=" * 70)
        print("The model is offloading to disk, causing extreme slowness.")
        print("\nSolutions:")
        print("  1. Use CPU instead: --device cpu (slower but stable)")
        print("  2. Use a smaller model: --model-index 2 (Dolphin, 7B)")
        print("  3. Use local GPT-4chan: --model-index 0 (already loaded)")
        print("  4. Close other applications to free RAM")
        print("=" * 70)


if __name__ == "__main__":
    main()

