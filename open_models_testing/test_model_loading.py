"""Test that models can be loaded without errors (but don't run inference).

Usage:
    poetry run python3 open_models_testing/test_model_loading.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from open_models_testing.loaders import (TransformersLoadConfig,
                                         load_transformers_causal_lm)


def test_gpt4chan_model_loading():
    """Test loading the local GPT-4chan model."""
    print("=== Testing Model Loading ===\n")
    
    model_path = "/Users/jordanmoshcovitis/Documents/GitHub/oasis/open_models_testing/models/gpt4chan_gptj_hf"
    
    print(f"[1/3] Checking model files exist at: {model_path}")
    model_dir = Path(model_path)
    
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    ]
    
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✓ {file} ({size / (1024**3):.2f} GB)" if size > 1e9 else f"  ✓ {file} ({size / 1024:.2f} KB)")
        else:
            print(f"  ✗ {file} MISSING!")
            return False
    
    print("\n[2/3] Testing model loading (CPU only, no inference)...")
    try:
        cfg = TransformersLoadConfig(
            model_name_or_path=model_path,
            device="cpu",  # Use CPU for quick load test
            dtype="float32",
            trust_remote_code=True,
        )
        model, tokenizer, device = load_transformers_causal_lm(cfg)
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Tokenizer vocabulary size: {len(tokenizer)}")
        print(f"  ✓ Device: {device}")
        print(f"  ✓ Model config: {model.config.model_type}")
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False
    
    print("\n[3/3] Testing tokenizer...")
    try:
        test_text = "This is a test sentence."
        tokens = tokenizer(test_text, return_tensors="pt")
        decoded = tokenizer.decode(tokens.input_ids[0])
        print(f"  ✓ Tokenizer encode/decode works")
        print(f"  ✓ Test: '{test_text}' -> {tokens.input_ids.shape[1]} tokens -> '{decoded}'")
    except Exception as e:
        print(f"  ✗ Tokenizer test failed: {e}")
        return False
    
    print("\n✓ All model loading tests passed!")
    return True


if __name__ == "__main__":
    success = test_gpt4chan_model_loading()
    sys.exit(0 if success else 1)

