from __future__ import annotations

"""Model loaders for open-source LLM feasibility tests.

Backends supported:
- transformers (causal LM and masked LM)
- llama-cpp (GGUF) via llama_cpp Python binding
- HTTP OpenAI-compatible endpoints (e.g., vLLM)

All functions include type hints and minimal configuration surfaces for clarity.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union


@dataclass
class TransformersLoadConfig:
    model_name_or_path: str
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Optional[Literal["auto", "float16", "bfloat16", "float32"]] = "auto"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    low_cpu_mem_usage: bool = True
    attn_implementation: Optional[str] = None  # e.g., "sdpa"
    max_seq_len: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    offload_folder: Optional[str] = None  # For disk offloading when OOM


def _resolve_torch_device(device_pref: str) -> str:
    try:
        import torch  # noqa: WPS433

        if device_pref == "cpu":
            return "cpu"
        if device_pref == "cuda" and torch.cuda.is_available():
            return "cuda"
        # Apple Metal
        if device_pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        # auto selection
        if device_pref == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def _resolve_dtype(dtype_pref: Optional[str]):
    if dtype_pref in (None, "auto"):
        return None
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_pref, None)


def load_transformers_causal_lm(cfg: TransformersLoadConfig):
    """Load a Transformers causal LM with tokenizer.

    Returns a tuple of (model, tokenizer, device_str).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_str = _resolve_torch_device(cfg.device)
    torch_dtype = _resolve_dtype(cfg.dtype)

    tok = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=cfg.use_fast_tokenizer,
        trust_remote_code=cfg.trust_remote_code,
    )
    
    # Build kwargs, filtering None values and unsupported params
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": cfg.trust_remote_code,
        "low_cpu_mem_usage": cfg.low_cpu_mem_usage,
    }
    
    # Use device_map for better memory management on GPU/MPS
    if device_str in ("cuda", "mps"):
        model_kwargs["device_map"] = "auto"
        # Add offload folder for disk swapping if model doesn't fit in RAM
        if cfg.offload_folder:
            model_kwargs["offload_folder"] = cfg.offload_folder
    
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    if cfg.rope_scaling:
        model_kwargs["rope_scaling"] = cfg.rope_scaling
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        **{k: v for k, v in model_kwargs.items() if v is not None}
    )
    
    # Manual device placement only if device_map wasn't used
    if "device_map" not in model_kwargs and device_str != "cpu":
        import torch
        device = torch.device(device_str)
        model = model.to(device)
    
    return model, tok, device_str


def load_transformers_masked_lm(cfg: TransformersLoadConfig):
    """Load a Transformers masked LM with tokenizer (e.g., BERT, RoBERTa, T5 encoder).

    Returns a tuple of (model, tokenizer, device_str).
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device_str = _resolve_torch_device(cfg.device)
    torch_dtype = _resolve_dtype(cfg.dtype)

    tok = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=cfg.use_fast_tokenizer,
        trust_remote_code=cfg.trust_remote_code,
    )
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": cfg.trust_remote_code,
        "low_cpu_mem_usage": cfg.low_cpu_mem_usage,
    }
    
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name_or_path,
        **{k: v for k, v in model_kwargs.items() if v is not None}
    )
    
    if device_str != "cpu":
        import torch
        device = torch.device(device_str)
        model = model.to(device)
    
    return model, tok, device_str


# T5 span fill (uses sentinel tokens like <extra_id_0>)
def load_transformers_t5(cfg: TransformersLoadConfig):
    """Load a T5 model for conditional generation with tokenizer.

    Returns a tuple of (model, tokenizer, device_str).
    """
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    device_str = _resolve_torch_device(cfg.device)
    torch_dtype = _resolve_dtype(cfg.dtype)

    tok = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=cfg.use_fast_tokenizer,
        trust_remote_code=cfg.trust_remote_code,
    )
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": cfg.trust_remote_code,
        "low_cpu_mem_usage": cfg.low_cpu_mem_usage,
    }
    
    model = T5ForConditionalGeneration.from_pretrained(
        cfg.model_name_or_path,
        **{k: v for k, v in model_kwargs.items() if v is not None}
    )
    
    if device_str != "cpu":
        import torch
        device = torch.device(device_str)
        model = model.to(device)
    
    return model, tok, device_str

# llama.cpp (GGUF)
@dataclass
class LlamaCppLoadConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    n_gpu_layers: int = -1  # -1 to offload all layers on Apple Metal
    seed: int = 0
    logits_all: bool = False
    use_mmap: bool = True
    use_mlock: bool = False


def load_llamacpp(cfg: LlamaCppLoadConfig):
    """Load a GGUF model using llama-cpp-python.

    Returns the Llama model object.
    """
    from llama_cpp import Llama

    llm = Llama(
        model_path=cfg.model_path,
        n_ctx=cfg.n_ctx,
        n_threads=cfg.n_threads,
        n_gpu_layers=cfg.n_gpu_layers,
        seed=cfg.seed,
        logits_all=cfg.logits_all,
        use_mmap=cfg.use_mmap,
        use_mlock=cfg.use_mlock,
    )
    return llm


# HTTP OpenAI-compatible
@dataclass
class HttpOpenAIConfig:
    base_url: str
    model: str
    api_key: str = "sk-no-key"
    timeout_s: float = 60.0


def make_openai_client(cfg: HttpOpenAIConfig):
    """Create a minimal OpenAI-compatible client for chat/completions."""
    from openai import OpenAI

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s)
    return client


