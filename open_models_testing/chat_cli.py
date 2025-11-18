from __future__ import annotations

"""Interactive chat CLI for testing models from models.impute.accessible.json.

Reuses adapters from run_impute.py for consistency.

Usage:
    # List available models
    poetry run python3 -m open_models_testing.chat_cli --list
    
    # Chat with model #1 (GPT-4chan local)
    poetry run python3 -m open_models_testing.chat_cli \
      --model-index 0 \
      --device mps --dtype float16
    
    # Chat with model by name
    poetry run python3 -m open_models_testing.chat_cli \
      --model-name "v2ray/GPT4chan-8B" \
      --device mps --dtype float16
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import signal
import sys

from open_models_testing.core import GenParams, GenResult
from open_models_testing.run_impute import (RunConfig, _adapter_http,
                                            _adapter_llamacpp,
                                            _adapter_transformers)


def timeout_handler(signum, frame):
    print("\n\n⚠ Generation timeout! The model may be stuck.")
    print("  Tip: Try reducing --max-new-tokens or use Ctrl+C")
    raise TimeoutError("Generation exceeded timeout")


class ChatSession:
    """Simple chat session with generation function."""
    
    def __init__(self, gen_fn, model_name: str, prompt_style: str):
        self.gen_fn = gen_fn
        self.model_name = model_name
        self.prompt_style = prompt_style
        self.history: List[Tuple[str, str]] = []
        self.system_prompt: str | None = None
    
    def set_system(self, prompt: str):
        # Warn if model doesn't support system prompts
        if self.prompt_style in ("gptj_strict", "gptj_completion", "gptj_completion_expected", "gptj_instruction", "t5", "fim"):
            print(f"\n⚠ Warning: Model style '{self.prompt_style}' doesn't use system prompts")
            print("  System prompt will be ignored for this model.")
        self.system_prompt = prompt
        print(f"✓ System prompt set ({len(prompt)} chars)")
    
    def clear(self):
        self.history = []
        print("✓ History cleared")
    
    def show_history(self):
        if not self.history:
            print("\n(No history)")
            return
        print("\n" + "="*70)
        for i, (u, a) in enumerate(self.history, 1):
            print(f"[{i}] You: {u}\n    Model: {a}\n")
        print("="*70)
    
    def build_prompt(self, user_input: str) -> str:
        """Build prompt respecting model's format.
        
        Note: For chat models (llama3, chatml), the adapter handles formatting.
        For base models (gptj), we just do simple continuation.
        """
        # For base/completion models, just concatenate history
        if self.prompt_style in ("gptj_strict", "gptj_completion", "gptj_completion_expected", "gptj_instruction", "t5"):
            parts = []
            for u, a in self.history:
                parts.append(f"{u}\n{a}\n")
            parts.append(user_input)
            return "\n".join(parts)
        
        # For chat models, build conversation format
        # (but note: chat_llama3/chatml are handled by the adapter via chat_template)
        parts = []
        if self.system_prompt:
            parts.append(f"SYSTEM: {self.system_prompt}\n")
        for u, a in self.history:
            parts.append(f"USER: {u}\nASSISTANT: {a}\n")
        parts.append(f"USER: {user_input}\nASSISTANT:")
        return "\n".join(parts)
    
    def chat(self, user_input: str, params: GenParams, timeout: int = 60) -> str:
        prompt = self.build_prompt(user_input)
        
        # Add model-specific stop sequences
        # Note: Transformers doesn't natively support string stop sequences in .generate()
        # The adapter will need to post-process, but we set them anyway for llamacpp/http
        if params.stop is None:
            if self.prompt_style == "gpt4chan_board":
                params.stop = ["<|start_header_id|>", "<|eot_id|>", "\n\n\n"]
            elif self.prompt_style == "chat_llama3":
                params.stop = ["<|eot_id|>"]
            elif self.prompt_style == "chat_chatml":
                params.stop = ["<|im_end|>"]
            elif self.prompt_style in ("gptj_strict", "gptj_completion"):
                params.stop = ["\n\n", "USER:", "---", ">>>"]
            else:
                params.stop = ["\n\n\n"]
        
        # Set timeout for generation
        if timeout > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        try:
            start = time.perf_counter()
            res = self.gen_fn(prompt, params)
            elapsed = time.perf_counter() - start
            
            # Cancel alarm
            if timeout > 0:
                signal.alarm(0)
            
            tps_str = f"{res.tokens_per_sec:.1f}" if res.tokens_per_sec else "N/A"
            print(f"  [{elapsed:.1f}s, {tps_str} tok/s]")
            
            self.history.append((user_input, res.text))
            return res.text
        
        except TimeoutError:
            if timeout > 0:
                signal.alarm(0)
            print(f"\n  ✗ Generation timed out after {timeout}s")
            print(f"  Tip: Reduce --max-new-tokens (current: {params.max_new_tokens})")
            return "[TIMEOUT]"
        
        except Exception as e:
            if timeout > 0:
                signal.alarm(0)
            raise


def load_models_config(path: Path) -> List[Dict]:
    config = json.loads(path.read_text())
    return config.get("impute", [])


def list_models(config_path: Path):
    models = load_models_config(config_path)
    print("\n" + "="*70)
    print("AVAILABLE MODELS")
    print("="*70)
    for i, spec in enumerate(models):
        model = spec.get("model") or spec.get("model_path", "unknown")
        backend = spec.get("backend", "unknown")
        print(f"  [{i}] {model}")
        print(f"      Backend: {backend}")
    print("="*70)
    print(f"\nTotal: {len(models)} models")
    print("\nUse --model-index N to chat with model N")


def main():
    parser = argparse.ArgumentParser(description="Chat CLI using models from config")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument("--model-index", type=int, help="Model index from config (0-based)")
    parser.add_argument("--model-name", type=str, help="Partial model name match")
    parser.add_argument("--config", type=str, default="open_models_testing/models.impute.accessible.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--offload-folder", type=str, default="open_models_testing/offload")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        sys.exit(1)
    
    models = load_models_config(config_path)
    
    if args.list:
        list_models(config_path)
        sys.exit(0)
    
    # Select model
    selected = None
    if args.model_index is not None:
        if 0 <= args.model_index < len(models):
            selected = models[args.model_index]
        else:
            print(f"✗ Invalid index: {args.model_index} (must be 0-{len(models)-1})")
            sys.exit(1)
    elif args.model_name:
        for spec in models:
            model_str = spec.get("model") or spec.get("model_path", "")
            if args.model_name.lower() in model_str.lower():
                selected = spec
                break
        if not selected:
            print(f"✗ No model matching: {args.model_name}")
            print("\nAvailable models:")
            list_models(config_path)
            sys.exit(1)
    else:
        print("✗ Must specify --model-index or --model-name (or use --list)")
        sys.exit(1)
    
    # Load model
    print("="*70)
    print("INTERACTIVE CHAT CLI")
    print("="*70)
    
    backend = selected.get("backend")
    model = selected.get("model")
    model_path = selected.get("model_path")
    
    print(f"\n[Loading model]")
    print(f"  Backend: {backend}")
    print(f"  Model: {model or model_path}")
    print(f"  Device: {args.device}")
    print(f"  Dtype: {args.dtype}")
    
    cfg = RunConfig(
        backend=backend,
        model=model,
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        repetition_penalty=1.0,
        stop=None,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        base_url=None,
        api_key="sk-no-key",
        offload_folder=args.offload_folder,
        prompt_style="auto",
        fim_tokens=None,
    )
    
    # Use adapters from run_impute.py
    if backend == "transformers":
        gen = _adapter_transformers(cfg)
    elif backend == "llamacpp":
        if not cfg.model_path:
            print("✗ Missing model_path for llamacpp")
            sys.exit(1)
        gen = _adapter_llamacpp(cfg)
    elif backend == "http":
        if not cfg.base_url or not cfg.model:
            print("✗ Missing base_url/model for http")
            sys.exit(1)
        gen = _adapter_http(cfg)
    else:
        print(f"✗ Unknown backend: {backend}")
        sys.exit(1)
    
    print(f"✓ Model loaded\n")
    
    # Detect prompt style (same logic as run_impute.py)
    mi = (model or model_path or "").lower()
    is_t5 = "t5" in mi
    is_gptj = ("gpt-j" in mi) or ("gptj" in mi) or ("gpt4chan_gptj" in mi)
    is_llama3 = ("llama-3" in mi) or ("llama3" in mi) or ("lexi" in mi)
    is_dolphin = ("dolphin" in mi)
    is_gpt4chan_8b = ("gpt4chan-8b" in mi) or ("v2ray/gpt4chan-8b" in mi)
    
    if is_t5:
        prompt_style = "t5"
    elif is_gptj:
        prompt_style = "gptj_strict"
    elif is_llama3:
        prompt_style = "chat_llama3"
    elif is_dolphin:
        prompt_style = "chat_chatml"
    elif is_gpt4chan_8b:
        prompt_style = "gpt4chan_board"
    else:
        prompt_style = "instruction"
    
    print(f"  Prompt style: {prompt_style}")
    
    session = ChatSession(gen, model or model_path, prompt_style)
    
    print(f"\n✓ Ready to chat with: {session.model_name}")
    if prompt_style in ("gptj_strict", "t5"):
        print("  Note: This is a base/completion model - no system prompt support")
    print("Type /help for commands, /quit to exit")
    
    # Params
    max_tok = args.max_new_tokens
    temp = args.temperature
    top_p = args.top_p
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            
            if user_input.startswith("/"):
                cmd = user_input.lower()
                
                if cmd in ("/quit", "/exit", "/q"):
                    print("\nGoodbye!")
                    break
                
                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /help     - Show this")
                    print("  /system   - Set system prompt")
                    print("  /history  - Show conversation")
                    print("  /clear    - Clear history")
                    print("  /params   - Change generation params")
                    print("  /quit     - Exit")
                
                elif cmd == "/system":
                    print("\nEnter system prompt (Enter twice to finish):")
                    lines = []
                    while True:
                        line = input()
                        if not line:
                            break
                        lines.append(line)
                    session.set_system("\n".join(lines))
                
                elif cmd == "/history":
                    session.show_history()
                
                elif cmd == "/clear":
                    session.clear()
                
                elif cmd == "/params":
                    print(f"\nCurrent: max_tokens={max_tok}, temp={temp}, top_p={top_p}")
                    try:
                        max_tok = int(input(f"max_new_tokens [{max_tok}]: ") or max_tok)
                        temp = float(input(f"temperature [{temp}]: ") or temp)
                        top_p = float(input(f"top_p [{top_p}]: ") or top_p)
                        print("✓ Updated")
                    except ValueError:
                        print("✗ Invalid input")
                
                else:
                    print(f"Unknown: {user_input}")
                continue
            
            # Generate
            params = GenParams(max_new_tokens=max_tok, temperature=temp, top_p=top_p)
            response = session.chat(user_input, params)
            print(f"\nModel: {response}")
        
        except KeyboardInterrupt:
            print("\n(Ctrl+C) Use /quit to exit")
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    main()
