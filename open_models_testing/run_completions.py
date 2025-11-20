from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from .loaders import (
    HttpOpenAIConfig,
    LlamaCppLoadConfig,
    TransformersLoadConfig,
    load_llamacpp,
    load_transformers_causal_lm,
    load_transformers_t5,
    make_openai_client,
)


class PromptRow(BaseModel):
    id: str
    prompt: str
    params: Dict[str, Any] | None = None


@dataclass
class RunConfig:
    backend: str
    model: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "auto"
    dtype: Optional[str] = "auto"
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    # llama.cpp specific
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    # http client
    base_url: Optional[str] = None
    api_key: str = "sk-no-key"


def _load_dataset(path: Path) -> List[PromptRow]:
    data = json.loads(Path(path).read_text())
    rows: List[PromptRow] = []
    for obj in data:
        try:
            rows.append(PromptRow(**obj))
        except ValidationError as e:
            raise ValueError(f"Invalid dataset row: {obj}") from e
    return rows


def _generate_transformers(
    prompts: Iterable[PromptRow], cfg: RunConfig
) -> List[Dict[str, Any]]:
    # Choose model class (T5 vs causal LM)
    model_id = cfg.model or ""
    if "t5" in model_id.lower():
        model, tok, device_str = load_transformers_t5(
            TransformersLoadConfig(
                model_name_or_path=model_id,
                device=cfg.device,
                dtype=cfg.dtype,  # type: ignore[arg-type]
                trust_remote_code=True,
            )
        )
    else:
        model, tok, device_str = load_transformers_causal_lm(
            TransformersLoadConfig(
                model_name_or_path=model_id,
                device=cfg.device,
                dtype=cfg.dtype,  # type: ignore[arg-type]
                trust_remote_code=True,
            )
        )

    import torch
    import math
    from torch.nn import functional as F

    results: List[Dict[str, Any]] = []
    for row in tqdm(list(prompts), desc="transformers"):
        params = {
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k if cfg.top_k > 0 else None,
            "repetition_penalty": cfg.repetition_penalty,
            "do_sample": cfg.temperature > 0.0,
            "eos_token_id": None,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if row.params:
            params.update(row.params)
        enc = tok(row.prompt, return_tensors="pt")
        input_ids = enc.input_ids
        device = torch.device("cuda" if device_str == "cuda" else ("mps" if device_str == "mps" else "cpu"))
        input_ids = input_ids.to(device)
        torch.cuda.synchronize() if device_str == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            gen_out = model.generate(input_ids=input_ids, **{k: v for k, v in params.items() if v is not None})
        torch.cuda.synchronize() if device_str == "cuda" else None
        end = time.perf_counter()

        if hasattr(gen_out, "sequences"):
            sequences = gen_out.sequences
            output_ids = sequences[0][input_ids.shape[1]:]
            text = tok.decode(output_ids, skip_special_tokens=True)
            scores = list(getattr(gen_out, "scores", []) or [])
            step_logprobs: List[float] = []
            for i, step_scores in enumerate(scores):
                # step_scores: [B, V]
                logp = F.log_softmax(step_scores[0], dim=-1)
                token_id = int(output_ids[i].item())
                step_logprobs.append(float(logp[token_id].item()))
        else:
            # Fallback to tensor output
            output_ids = gen_out[0][input_ids.shape[1]:]
            text = tok.decode(output_ids, skip_special_tokens=True)
            step_logprobs = []

        in_tokens = int(input_ids.numel())
        out_tokens = int(output_ids.numel())
        latency = end - start
        tps = out_tokens / latency if latency > 0 else None
        perplexity = float(math.exp(-sum(step_logprobs) / len(step_logprobs))) if step_logprobs else None

        results.append({
            "id": row.id,
            "prompt": row.prompt,
            "backend": "transformers",
            "model_name_or_path": cfg.model,
            "params_json": json.dumps({k: v for k, v in params.items() if k not in ("output_scores", "return_dict_in_generate")}),
            "output_text": text,
            "latency_ms": int(latency * 1000),
            "tokens_per_sec": tps,
            "num_input_tokens": in_tokens,
            "num_output_tokens": out_tokens,
            "perplexity": perplexity,
            "logprobs": json.dumps(step_logprobs) if step_logprobs else None,
            "error": None,
        })
    return results


def _generate_llamacpp(
    prompts: Iterable[PromptRow], cfg: RunConfig
) -> List[Dict[str, Any]]:
    llm = load_llamacpp(LlamaCppLoadConfig(model_path=cfg.model_path or "", n_ctx=cfg.n_ctx, n_gpu_layers=cfg.n_gpu_layers))
    results: List[Dict[str, Any]] = []
    for row in tqdm(list(prompts), desc="llama.cpp"):
        params = {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k if cfg.top_k > 0 else None,
            "repeat_penalty": cfg.repetition_penalty,
            "max_tokens": cfg.max_new_tokens,
            "stop": cfg.stop,
            "logprobs": 1,
        }
        start = time.perf_counter()
        out = llm.create_completion(prompt=row.prompt, **{k: v for k, v in params.items() if v is not None})
        end = time.perf_counter()
        text = out["choices"][0]["text"]
        # llama.cpp returns token counts in usage if logits_all or verbose_json; may be None otherwise
        usage = out.get("usage", {})
        in_tokens = usage.get("prompt_tokens", None)
        out_tokens = usage.get("completion_tokens", None)
        step_logprobs = None
        try:
            lp = out["choices"][0].get("logprobs", {})
            token_lps = lp.get("token_logprobs", None)
            if token_lps is not None:
                step_logprobs = [float(v) if v is not None else None for v in token_lps]
        except Exception:
            step_logprobs = None
        latency = end - start
        tps = (out_tokens or 0) / latency if latency and out_tokens else None
        results.append({
            "id": row.id,
            "prompt": row.prompt,
            "backend": "llamacpp",
            "model_path": cfg.model_path,
            "params_json": json.dumps(params),
            "output_text": text,
            "latency_ms": int(latency * 1000),
            "tokens_per_sec": tps,
            "num_input_tokens": in_tokens,
            "num_output_tokens": out_tokens,
            "perplexity": None,
            "logprobs": json.dumps(step_logprobs) if step_logprobs else None,
            "error": None,
        })
    return results


def _generate_http(
    prompts: Iterable[PromptRow], cfg: RunConfig
) -> List[Dict[str, Any]]:
    client = make_openai_client(HttpOpenAIConfig(base_url=cfg.base_url or "", model=cfg.model or "", api_key=cfg.api_key))
    results: List[Dict[str, Any]] = []
    for row in tqdm(list(prompts), desc="http"):
        params = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_new_tokens,
            "stop": cfg.stop,
            "logprobs": True,
            "top_logprobs": 1,
        }
        start = time.perf_counter()
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": row.prompt}],
            **{k: v for k, v in params.items() if v is not None},
        )
        end = time.perf_counter()
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        out_tokens = getattr(usage, "completion_tokens", None) if usage else None
        step_logprobs = None
        try:
            lp = resp.choices[0].logprobs  # type: ignore[attr-defined]
            # lp.content is a list of token items with logprob and top_logprobs
            content_items = getattr(lp, "content", None)
            if content_items:
                step_logprobs = [float(getattr(item, "logprob", None)) for item in content_items]
        except Exception:
            step_logprobs = None
        latency = end - start
        tps = (out_tokens or 0) / latency if latency and out_tokens else None
        # logprobs may be supported by some servers; not assumed here
        results.append({
            "id": row.id,
            "prompt": row.prompt,
            "backend": "http",
            "model_name_or_path": cfg.model,
            "params_json": json.dumps(params),
            "output_text": text,
            "latency_ms": int(latency * 1000),
            "tokens_per_sec": tps,
            "num_input_tokens": in_tokens,
            "num_output_tokens": out_tokens,
            "perplexity": (float(__import__("math").exp(-sum(step_logprobs) / len(step_logprobs))) if step_logprobs else None),
            "logprobs": json.dumps(step_logprobs) if step_logprobs else None,
            "error": None,
        })
    return results


def build_html_summary(rows: List[Dict[str, Any]], out_html: Path) -> None:
    df = pd.DataFrame(rows)
    # keep primary columns in front
    preferred_cols = [
        "id",
        "backend",
        "model_name_or_path",
        "model_path",
        "prompt",
        "output_text",
        "latency_ms",
        "tokens_per_sec",
        "num_input_tokens",
        "num_output_tokens",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]
    out_html.write_text(df.to_html(index=False))


def run(args: argparse.Namespace) -> None:
    dataset = _load_dataset(Path(args.dataset))

    aggregated: List[Dict[str, Any]] = []
    models_cfg_path = args.models_config
    if models_cfg_path:
        config = json.loads(Path(models_cfg_path).read_text())
        # Expect either top-level list or keys like "completions"
        model_specs: List[Dict[str, Any]]
        if isinstance(config, list):
            model_specs = config
        else:
            model_specs = config.get("completions", [])

        for spec in model_specs:
            backend = spec.get("backend", args.backend)
            model = spec.get("model")
            model_path = spec.get("model_path")
            base_url = spec.get("base_url", args.base_url)

            cfg = RunConfig(
                backend=backend,
                model=model,
                model_path=model_path,
                device=args.device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                stop=args.stop,
                n_ctx=args.n_ctx,
                n_gpu_layers=args.n_gpu_layers,
                base_url=base_url,
                api_key=args.api_key,
            )
            if backend == "transformers":
                aggregated.extend(_generate_transformers(dataset, cfg))
            elif backend == "llamacpp":
                if not cfg.model_path:
                    raise ValueError("model_path is required for llamacpp spec")
                aggregated.extend(_generate_llamacpp(dataset, cfg))
            elif backend == "http":
                if not cfg.base_url or not cfg.model:
                    raise ValueError("base_url and model required for http spec")
                aggregated.extend(_generate_http(dataset, cfg))
            else:
                raise ValueError(f"Unknown backend in spec: {backend}")
    else:
        cfg = RunConfig(
            backend=args.backend,
            model=args.model,
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        if cfg.backend == "transformers":
            aggregated = _generate_transformers(dataset, cfg)
        elif cfg.backend == "llamacpp":
            if not cfg.model_path:
                raise ValueError("--model-path is required for llamacpp backend")
            aggregated = _generate_llamacpp(dataset, cfg)
        elif cfg.backend == "http":
            if not cfg.base_url or not cfg.model:
                raise ValueError("--base-url and --model are required for http backend")
            aggregated = _generate_http(dataset, cfg)
        else:
            raise ValueError(f"Unknown backend: {cfg.backend}")

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aggregated).to_csv(out_csv, index=False)
    if args.html:
        build_html_summary(aggregated, Path(args.html))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run completions for multiple backends.")
    parser.add_argument("--backend", choices=["transformers", "llamacpp", "http"], required=True)
    parser.add_argument("--model", type=str, help="HF model id or server model name")
    parser.add_argument("--model-path", type=str, help="Path to GGUF for llama.cpp")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--html", type=str)
    parser.add_argument("--models-config", type=str, help="JSON file listing models to compare")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop", nargs="*")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--base-url", type=str)
    parser.add_argument("--api-key", type=str, default="sk-no-key")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()


