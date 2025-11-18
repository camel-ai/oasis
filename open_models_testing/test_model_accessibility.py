"""Quick check of model accessibility without loading weights.

Usage:
    poetry run python3 open_models_testing/test_model_accessibility.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_local_model_files(model_path: str) -> dict:
    """Check if required files exist for a local model."""
    path = Path(model_path)
    required = ["config.json", "pytorch_model.bin"]
    
    if not path.exists():
        return {"status": "missing", "error": f"Path not found: {model_path}"}
    
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        return {"status": "incomplete", "error": f"Missing: {', '.join(missing)}"}
    
    # Check file sizes
    model_file = path / "pytorch_model.bin"
    size_gb = model_file.stat().st_size / (1024**3)
    
    return {
        "status": "ok",
        "size_gb": f"{size_gb:.2f} GB",
        "path": str(path)
    }


def check_remote_model_quick(model_id: str) -> dict:
    """Quick check if model exists on HuggingFace (HEAD request)."""
    import requests

    # Quick HEAD request to model card
    url = f"https://huggingface.co/{model_id}"
    try:
        resp = requests.head(url, timeout=5)
        if resp.status_code == 200:
            return {"status": "accessible", "url": url}
        elif resp.status_code == 404:
            return {"status": "not_found", "error": "Model not found on Hub"}
        elif resp.status_code in (401, 403):
            return {"status": "auth_required", "error": "Authentication/approval needed"}
        else:
            return {"status": "unknown", "error": f"HTTP {resp.status_code}"}
    except requests.Timeout:
        return {"status": "timeout", "error": "Request timed out"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    print("=== Quick Model Accessibility Check ===\n")
    
    config_path = Path("open_models_testing/models.impute.sample.json")
    config = json.loads(config_path.read_text())
    models = config.get("impute", [])
    
    print(f"Checking {len(models)} models...\n")
    
    results = []
    for i, spec in enumerate(models, 1):
        model = spec["model"]
        is_local = model.startswith("/") or not "/" in model or Path(model).exists()
        
        print(f"[{i}/{len(models)}] {model}")
        
        if is_local:
            result = check_local_model_files(model)
            print(f"  Type: LOCAL")
        else:
            result = check_remote_model_quick(model)
            print(f"  Type: REMOTE")
        
        icon = "✓" if result["status"] in ("ok", "accessible") else ("⚠" if result["status"] in ("auth_required", "timeout") else "✗")
        print(f"  Status: {icon} {result['status'].upper()}")
        
        if "error" in result:
            print(f"  Error: {result['error']}")
        if "size_gb" in result:
            print(f"  Size: {result['size_gb']}")
        if "url" in result:
            print(f"  URL: {result['url']}")
        
        print()
        results.append({"model": model, "is_local": is_local, **result})
    
    # Summary
    ok = sum(1 for r in results if r["status"] in ("ok", "accessible"))
    warn = sum(1 for r in results if r["status"] in ("auth_required", "timeout"))
    err = sum(1 for r in results if r["status"] in ("missing", "incomplete", "not_found", "error"))
    
    print(f"=== Summary: {ok} OK | {warn} Warnings | {err} Errors ===\n")
    
    if err > 0 or warn > 0:
        print("⚠ Issues found:")
        for r in results:
            if r["status"] not in ("ok", "accessible"):
                print(f"  - {r['model']}: {r.get('error', r['status'])}")
        print("\nRecommendation: Remove problematic models from models.impute.sample.json")
        print("or download them before running imputation.\n")
    else:
        print("✓ All models accessible!\n")
    
    return err == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

