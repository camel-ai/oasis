"""Test accessibility and basic loading for all models in models.impute.sample.json.

This does NOT download models, but checks:
- Local models exist and are loadable
- Remote models are accessible via HuggingFace Hub API
- Model architectures are supported

Usage:
    poetry run python3 open_models_testing/test_all_models.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from open_models_testing.loaders import (TransformersLoadConfig,
                                         load_transformers_causal_lm,
                                         load_transformers_t5)


def check_local_model(model_path: str) -> Dict[str, Any]:
    """Check if a local model exists and has required files."""
    result = {"status": "unknown", "error": None, "details": {}}
    
    path = Path(model_path)
    if not path.exists():
        result["status"] = "missing"
        result["error"] = f"Path does not exist: {model_path}"
        return result
    
    required_files = ["config.json", "pytorch_model.bin"]
    missing = []
    for file in required_files:
        if not (path / file).exists():
            missing.append(file)
    
    if missing:
        result["status"] = "incomplete"
        result["error"] = f"Missing files: {', '.join(missing)}"
        return result
    
    # Try loading on CPU (quick check)
    try:
        cfg = TransformersLoadConfig(
            model_name_or_path=model_path,
            device="cpu",
            dtype="float32",
            trust_remote_code=True,
        )
        
        # Determine if T5 or causal
        if "t5" in model_path.lower():
            model, tok, _ = load_transformers_t5(cfg)
        else:
            model, tok, _ = load_transformers_causal_lm(cfg)
        
        result["status"] = "ok"
        result["details"]["vocab_size"] = len(tok)
        result["details"]["model_type"] = model.config.model_type
        
        # Clean up to free memory
        del model
        del tok
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def check_remote_model(model_id: str) -> Dict[str, Any]:
    """Check if a remote model is accessible via HuggingFace Hub."""
    result = {"status": "unknown", "error": None, "details": {}}
    
    try:
        from huggingface_hub import model_info
        
        info = model_info(model_id)
        result["status"] = "accessible"
        result["details"]["model_id"] = info.id
        result["details"]["downloads"] = getattr(info, "downloads", "N/A")
        result["details"]["likes"] = getattr(info, "likes", "N/A")
        
        # Check if gated/private
        if hasattr(info, "gated") and info.gated:
            result["status"] = "gated"
            result["error"] = "Model requires authentication/approval"
        
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "403" in error_str:
            result["status"] = "auth_required"
            result["error"] = "Authentication required or access denied"
        elif "404" in error_str:
            result["status"] = "not_found"
            result["error"] = "Model not found on HuggingFace Hub"
        else:
            result["status"] = "error"
            result["error"] = error_str
    
    return result


def run_all_model_tests():
    """Test all models listed in models.impute.sample.json."""
    print("=== Testing All Models from models.impute.sample.json ===\n")
    
    config_path = Path("open_models_testing/models.impute.sample.json")
    config = json.loads(config_path.read_text())
    model_specs = config.get("impute", [])
    
    print(f"Found {len(model_specs)} models to test\n")
    
    results = []
    for i, spec in enumerate(model_specs, 1):
        backend = spec.get("backend")
        model = spec.get("model")
        
        print(f"[{i}/{len(model_specs)}] Testing: {model}")
        print(f"  Backend: {backend}")
        
        # Determine if local or remote
        is_local = model.startswith("/") or Path(model).exists()
        
        if is_local:
            print(f"  Type: Local model")
            result = check_local_model(model)
        else:
            print(f"  Type: Remote model (HuggingFace Hub)")
            result = check_remote_model(model)
        
        status_icon = {
            "ok": "✓",
            "accessible": "✓",
            "missing": "✗",
            "incomplete": "✗",
            "not_found": "✗",
            "auth_required": "⚠",
            "gated": "⚠",
            "error": "✗",
            "unknown": "?",
        }.get(result["status"], "?")
        
        print(f"  Status: {status_icon} {result['status'].upper()}")
        
        if result["error"]:
            print(f"  Error: {result['error']}")
        
        if result["details"]:
            for key, value in result["details"].items():
                print(f"  {key}: {value}")
        
        print()
        
        results.append({
            "model": model,
            "backend": backend,
            "is_local": is_local,
            **result
        })
    
    # Summary
    print("=== Summary ===")
    ok_count = sum(1 for r in results if r["status"] in ("ok", "accessible"))
    warning_count = sum(1 for r in results if r["status"] in ("auth_required", "gated"))
    error_count = sum(1 for r in results if r["status"] in ("missing", "incomplete", "not_found", "error"))
    
    print(f"Total models: {len(results)}")
    print(f"✓ OK/Accessible: {ok_count}")
    print(f"⚠ Warnings: {warning_count}")
    print(f"✗ Errors: {error_count}")
    
    if warning_count > 0:
        print("\n⚠ Models requiring attention:")
        for r in results:
            if r["status"] in ("auth_required", "gated"):
                print(f"  - {r['model']}: {r['error']}")
    
    if error_count > 0:
        print("\n✗ Models with errors:")
        for r in results:
            if r["status"] in ("missing", "incomplete", "not_found", "error"):
                print(f"  - {r['model']}: {r['error']}")
    
    # Recommendations
    print("\n=== Recommendations ===")
    
    local_models = [r for r in results if r["is_local"]]
    remote_models = [r for r in results if not r["is_local"]]
    
    print(f"Local models: {len(local_models)}")
    print(f"Remote models: {len(remote_models)}")
    
    if error_count == 0 and warning_count == 0:
        print("\n✓ All models ready for imputation!")
    else:
        print("\n⚠ Some models need attention before running imputation:")
        print("  - Remove inaccessible models from models.impute.sample.json, OR")
        print("  - Download/authenticate required models, OR")
        print("  - Run imputation with only accessible models")
    
    return error_count == 0


if __name__ == "__main__":
    try:
        success = run_all_model_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)

