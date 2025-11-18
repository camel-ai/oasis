from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from .core import (GenParams, GenResult, build_fim_prompt,
                   build_gptj_completion_prompt, build_gptj_expected_prompt,
                   build_gptj_instruction_prompt, build_impute_prompt,
                   build_t5_input, get_board_prefix_for_persona, impute_once,
                   parse_fim_output, parse_t5_output, split_prefix_suffix)
from .loaders import (HttpOpenAIConfig, LlamaCppLoadConfig,
                      TransformersLoadConfig, load_llamacpp,
                      load_transformers_causal_lm, load_transformers_t5,
                      make_openai_client)


class ImputeRow(BaseModel):
    id: str
    context: str
    instruction: str
    placeholder_token: str = "<placeholder>"
    persona: str | None = None
    persona_card_id: str | None = None
    expected_type: str | None = None


@dataclass
class RunConfig:
    backend: str
    model: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "auto"
    dtype: Optional[str] = "auto"
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    base_url: Optional[str] = None
    api_key: str = "sk-no-key"
    offload_folder: Optional[str] = None
    prompt_style: str = "auto"  # auto|t5|fim|instruction
    fim_tokens: Optional[str] = None  # e.g., "<PRE>,<SUF>,<MID>,<EOM>"
    board_prefix: str = "b"  # For gpt4chan: b/pol/r9k/fit (auto-detects from persona if not set)


SYSTEM_TMPL = (
    "You are a helpful assistant. Follow the persona and instruction to impute"
    " the placeholder in the given context. Return only the replacement text,"
    " without quotes or extra commentary."
)

USER_TMPL = (
    "Persona:\n{persona}\n\n"
    "Instruction:\n{instruction}\n\n"
    "Context:\n{context}\n\n"
    "Placeholder token: {placeholder_token}\n\n"
    "Provide only the replacement text for the placeholder."
)


def _load_dataset(path: Path) -> List[ImputeRow]:
    data = json.loads(Path(path).read_text())
    rows: List[ImputeRow] = []
    for obj in data:
        try:
            rows.append(ImputeRow(**obj))
        except ValidationError as e:
            raise ValueError(f"Invalid dataset row: {obj}") from e
    return rows


def _placeholder_at_sentence_end(context: str, placeholder_token: str) -> Optional[int]:
    """Return start index of placeholder if it is at sentence end (allowing punctuation), else None."""
    # Allow trailing punctuation or whitespace after placeholder until end of string
    # Examples: "... <PH>." or "... <PH>\n"
    pattern = re.compile(re.escape(placeholder_token) + r"[\s\)\]\}\"'”’.,;:!?—-]*\Z")
    m = pattern.search(context)
    if not m:
        return None
    return m.start()


def _derive_gptj_variant_records(rows: List[ImputeRow], variant: str) -> List[Dict[str, Any]]:
    """Build derived dataset rows for GPT-J experiments.

    Variants:
      - 'completion_endonly': keep rows where placeholder appears at sentence end; trim context up to placeholder (remove placeholder)
      - 'expected_endonly': same as completion_endonly, but prefix context with '<expected_type>: '
    """
    derived: List[Dict[str, Any]] = []
    for row in rows:
        idx = _placeholder_at_sentence_end(row.context, row.placeholder_token)
        if idx is None:
            continue
        base = row.context[:idx].rstrip()
        if variant == "completion_endonly":
            new_context = base
        elif variant == "expected_endonly":
            label = (row.expected_type or "").strip()
            prefix = f"{label}: " if label else ""
            new_context = prefix + base
        else:
            raise ValueError(f"Unknown variant: {variant}")

        derived.append({
            "id": f"{row.id}::{variant}",
            "persona": row.persona,
            "persona_card_id": row.persona_card_id,
            "context": new_context,
            "instruction": row.instruction,
            "placeholder_token": row.placeholder_token,
            "expected_type": row.expected_type,
        })
    return derived


def _prompt_text(persona_text: str, row: ImputeRow) -> str:
    return USER_TMPL.format(
        persona=persona_text,
        instruction=row.instruction,
        context=row.context,
        placeholder_token=row.placeholder_token,
    )


def _adapter_transformers(cfg: RunConfig) -> Callable[[str, GenParams], GenResult]:
    model_id = cfg.model or ""
    load_cfg = TransformersLoadConfig(
        model_name_or_path=model_id,
        device=cfg.device,
        dtype=cfg.dtype,  # type: ignore[arg-type]
        trust_remote_code=True,
        offload_folder=cfg.offload_folder,
    )
    
    is_t5_model = "t5" in model_id.lower()
    if is_t5_model:
        model, tok, device_str = load_transformers_t5(load_cfg)
    else:
        model, tok, device_str = load_transformers_causal_lm(load_cfg)
    import math

    import torch
    from torch.nn import functional as F

    def generate_fn(prompt: str, params: GenParams) -> GenResult:
        # Build model-aware prompt
        prompt_style = cfg.prompt_style
        output_parser = None
        final_prompt = prompt
        # Heuristics for decoder-only chat/instruction formats
        mi = (cfg.model or cfg.model_path or "").lower()
        is_llama3_like = ("llama-3" in mi) or ("llama3" in mi) or ("lexi" in mi)
        is_dolphin = ("dolphin" in mi)
        is_gptj_like = ("gpt-j" in mi) or ("gptj" in mi) or ("gpt4chan_gptj" in mi) or ("gpt-4chan" in mi)
        is_gpt4chan_8b = ("gpt4chan-8b" in mi) or ("v2ray/gpt4chan-8b" in mi)

        if prompt_style == "auto":
            if is_t5_model:
                prompt_style = "t5"
            elif is_gptj_like:
                prompt_style = "gptj_strict"
            elif is_llama3_like:
                prompt_style = "chat_llama3"
            elif is_dolphin:
                prompt_style = "chat_chatml"
            elif is_gpt4chan_8b:
                prompt_style = "gpt4chan_board"
            else:
                prompt_style = "instruction"

        # Determine encoding strategy and parser
        encoded_input_ids = None  # type: ignore[assignment]
        eos_token_id_override = None

        if prompt_style == "t5":
            # Expect prompt to be the directive; rebuild for T5
            # We need persona/context/placeholder; reconstruct from prompt is hard here,
            # so we assume caller already passed a T5-ready input in 'prompt'.
            # We still ensure decoding keeps special tokens for parsing.
            final_prompt = prompt
            def _parse(text: str) -> str:
                return parse_t5_output(text)
            output_parser = _parse

        elif prompt_style == "fim":
            # Try to split prefix/suffix using placeholder token inside 'prompt'
            # Here 'prompt' is the instruction-style text; this path is used only
            # when the caller passes FIM-ready content. In our flow, we will build
            # FIM prompt at the call site using build_fim_prompt.
            fim_tokens = ("<PRE>", "<SUF>", "<MID>", "<EOM>")
            if cfg.fim_tokens:
                parts = [p.strip() for p in cfg.fim_tokens.split(",")]
                if len(parts) == 4:
                    fim_tokens = tuple(parts)  # type: ignore[assignment]
            final_prompt = prompt
            def _parse(text: str) -> str:
                return parse_fim_output(text, fim_tokens)  # type: ignore[arg-type]
            output_parser = _parse

        elif prompt_style == "gptj_strict":
            # Strict single-line response; cut early on common breaks
            final_prompt = prompt
            stops = ["\n", "\n\n", "<<<END>>>", "\nAnswer:", "\nText:", ">>>"]
            def _parse(text: str) -> str:
                idxs = [text.find(s) for s in stops if text.find(s) != -1]
                if idxs:
                    cut = min(idxs)
                    return text[:cut].strip()
                return text.strip()
            output_parser = _parse
        elif prompt_style in ("gptj_completion", "gptj_completion_expected", "gptj_instruction"):
            # GPT-J completion styles use the same conservative single-line parser
            final_prompt = prompt
            stops = ["\n", "\n\n", "<<<END>>>", "\nAnswer:", "\nText:", ">>>"]
            def _parse(text: str) -> str:
                idxs = [text.find(s) for s in stops if text.find(s) != -1]
                if idxs:
                    cut = min(idxs)
                    return text[:cut].strip()
                return text.strip()
            output_parser = _parse
        elif prompt_style == "chat_llama3":
            # Use tokenizer chat template when available; stop on eot if present
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_TMPL},
                    {"role": "user", "content": prompt},
                ]
                encoded_input_ids = tok.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=True)
                # Try to use <|eot_id|> if available
                try:
                    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
                    if isinstance(eot_id, int) and eot_id != tok.unk_token_id:
                        eos_token_id_override = eot_id
                except Exception:
                    eos_token_id_override = None
            except Exception:
                # Fallback to plain instruction prompt
                final_prompt = prompt
            def _parse(text: str) -> str:
                if "<|eot_id|>" in text:
                    return text.split("<|eot_id|>", 1)[0].strip()
                return text.strip()
            output_parser = _parse
        elif prompt_style == "chat_chatml":
            # Build ChatML string manually
            chatml = (
                "<|im_start|>system\n" +
                "You are a helpful assistant. Return only the replacement text.\n" +
                "<|im_end|>\n" +
                "<|im_start|>user\n" +
                f"{prompt}\n" +
                "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
            final_prompt = chatml
            def _parse(text: str) -> str:
                # Cut at ChatML end tag if produced
                if "<|im_end|>" in text:
                    return text.split("<|im_end|>", 1)[0].strip()
                return text.strip()
            output_parser = _parse
        elif prompt_style == "gpt4chan_board":
            # Board/id header format matching official GPT4chan-8B example
            # Format: {board}<|start_header_id|>1<|end_header_id|>content<|start_header_id|>2<|end_header_id|>
            # Board selection: pol (politics/conspiracy), r9k (incel), fit (body), b (random)
            # Note: board auto-detection happens at caller level where persona_card_id is available
            board = cfg.board_prefix if cfg.board_prefix else "b"
            board_prompt = (
                f"{board}<|start_header_id|>1<|end_header_id|>" +
                f"{prompt}\n" +
                "<|start_header_id|>2<|end_header_id|>"
            )
            final_prompt = board_prompt
            def _parse(text: str) -> str:
                # Cut at next header or eot
                for cut_token in ["<|start_header_id|>", "<|eot_id|>", "<|end_of_text|>"]:
                    if cut_token in text:
                        return text.split(cut_token, 1)[0].strip()
                return text.strip()
            output_parser = _parse
        else:
            # instruction
            final_prompt = prompt
            output_parser = lambda t: t.strip()

        if encoded_input_ids is None:
            enc = tok(final_prompt, return_tensors="pt")
            input_ids = enc.input_ids
        else:
            input_ids = encoded_input_ids
        device = torch.device("cuda" if device_str == "cuda" else ("mps" if device_str == "mps" else "cpu"))
        input_ids = input_ids.to(device)
        gen_kwargs = {
            "max_new_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k if params.top_k > 0 else None,
            "repetition_penalty": params.repetition_penalty,
            "do_sample": params.temperature > 0.0,
            "eos_token_id": eos_token_id_override if eos_token_id_override is not None else getattr(tok, "eos_token_id", None),
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, **{k: v for k, v in gen_kwargs.items() if v is not None})
        end = time.perf_counter()

        if hasattr(out, "sequences"):
            sequences = out.sequences
            # For encoder-decoder (T5), decoder output doesn't include encoder input
            if is_t5_model:
                output_ids = sequences[0]
            else:
                output_ids = sequences[0][input_ids.shape[1]:]
            text = tok.decode(output_ids, skip_special_tokens=False)
            scores = list(getattr(out, "scores", []) or [])
            step_logprobs: List[float] = []
            for i, step_scores in enumerate(scores):
                if i >= len(output_ids):
                    break
                logp = F.log_softmax(step_scores[0], dim=-1)
                token_id = int(output_ids[i].item())
                step_logprobs.append(float(logp[token_id].item()))
        else:
            if is_t5_model:
                output_ids = out[0]
            else:
                output_ids = out[0][input_ids.shape[1]:]
            text = tok.decode(output_ids, skip_special_tokens=False)
            step_logprobs = []

        in_tokens = int(input_ids.numel())
        out_tokens = int(output_ids.numel())
        latency = end - start
        tps = out_tokens / latency if latency > 0 else None
        perplexity = float(math.exp(-sum(step_logprobs) / len(step_logprobs))) if step_logprobs else None
        parsed = output_parser(text) if output_parser else text.strip()
        return GenResult(
            text=parsed,
            num_input_tokens=in_tokens,
            num_output_tokens=out_tokens,
            tokens_per_sec=tps,
            logprobs=step_logprobs or None,
            perplexity=perplexity,
        )

    return generate_fn


def _adapter_llamacpp(cfg: RunConfig) -> Callable[[str, GenParams], GenResult]:
    llm = load_llamacpp(LlamaCppLoadConfig(model_path=cfg.model_path or "", n_ctx=cfg.n_ctx, n_gpu_layers=cfg.n_gpu_layers))

    def generate_fn(prompt: str, params: GenParams) -> GenResult:
        llm_kwargs = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k if params.top_k > 0 else None,
            "repeat_penalty": params.repetition_penalty,
            "max_tokens": params.max_new_tokens,
            "logprobs": 1,
        }
        start = time.perf_counter()
        out = llm.create_completion(prompt=prompt, **{k: v for k, v in llm_kwargs.items() if v is not None})
        end = time.perf_counter()
        text = (out["choices"][0]["text"] or "").strip()
        usage = out.get("usage", {})
        in_tokens = usage.get("prompt_tokens", None)
        out_tokens = usage.get("completion_tokens", None)
        lp = out["choices"][0].get("logprobs", {})
        token_lps = lp.get("token_logprobs", None)
        step_logprobs = [float(v) for v in token_lps] if token_lps else None
        latency = end - start
        tps = (out_tokens or 0) / latency if latency and out_tokens else None
        return GenResult(
            text=text,
            num_input_tokens=in_tokens,
            num_output_tokens=out_tokens,
            tokens_per_sec=tps,
            logprobs=step_logprobs,
            perplexity=None,
        )

    return generate_fn


def _adapter_http(cfg: RunConfig) -> Callable[[str, GenParams], GenResult]:
    client = make_openai_client(HttpOpenAIConfig(base_url=cfg.base_url or "", model=cfg.model or "", api_key=cfg.api_key))

    def generate_fn(prompt: str, params: GenParams) -> GenResult:
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_TMPL},
                {"role": "user", "content": prompt},
            ],
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_new_tokens,
            stop=params.stop,
            logprobs=True,
            top_logprobs=1,
        )
        end = time.perf_counter()
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        out_tokens = getattr(usage, "completion_tokens", None) if usage else None
        step_logprobs = None
        try:
            lp = resp.choices[0].logprobs  # type: ignore[attr-defined]
            content_items = getattr(lp, "content", None)
            if content_items:
                step_logprobs = [float(getattr(item, "logprob", None)) for item in content_items]
        except Exception:
            step_logprobs = None
        latency = end - start
        tps = (out_tokens or 0) / latency if latency and out_tokens else None
        return GenResult(
            text=text,
            num_input_tokens=in_tokens,
            num_output_tokens=out_tokens,
            tokens_per_sec=tps,
            logprobs=step_logprobs,
            perplexity=(float(__import__("math").exp(-sum(step_logprobs) / len(step_logprobs))) if step_logprobs else None),
        )

    return generate_fn


def run(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("OPEN MODELS TESTING - IMPUTATION RUNNER")
    print("=" * 70)
    
    print(f"\n[1/5] Loading dataset: {args.dataset}")
    rows = _load_dataset(Path(args.dataset))
    print(f"  ✓ Loaded {len(rows)} rows")
    
    persona_map: Dict[str, str] | None = None
    if args.persona_file:
        print(f"\n[2/5] Loading persona mapping: {args.persona_file}")
        persona_map = json.loads(Path(args.persona_file).read_text())
        print(f"  ✓ Loaded {len(persona_map)} persona cards")
    else:
        print(f"\n[2/5] No persona file provided, skipping")

    aggregated: List[Dict[str, Any]] = []
    models_cfg_path = args.models_config
    if models_cfg_path:
        print(f"\n[3/5] Loading models config: {models_cfg_path}")
        config = json.loads(Path(models_cfg_path).read_text())
        model_specs: List[Dict[str, Any]]
        if isinstance(config, list):
            model_specs = config
        else:
            model_specs = config.get("impute", []) or config.get("completions", [])
        
        print(f"  ✓ Found {len(model_specs)} models to run")

        for idx, spec in enumerate(model_specs, 1):
            backend = spec.get("backend", args.backend)
            model = spec.get("model")
            model_path = spec.get("model_path")
            base_url = spec.get("base_url", args.base_url)
            prompt_style = spec.get("prompt_style", args.prompt_style)
            fim_tokens = spec.get("fim_tokens")

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
                offload_folder=args.offload_folder,
                prompt_style=prompt_style,
                fim_tokens=fim_tokens,
            )
            
            print(f"\n  [Loading model...]")
            start_load = time.perf_counter()
            if backend == "transformers":
                gen = _adapter_transformers(cfg)
            elif backend == "llamacpp":
                if not cfg.model_path:
                    raise ValueError("model_path is required for llamacpp spec")
                gen = _adapter_llamacpp(cfg)
            elif backend == "http":
                if not cfg.base_url or not cfg.model:
                    raise ValueError("base_url and model required for http spec")
                gen = _adapter_http(cfg)
            else:
                raise ValueError(f"Unknown backend in spec: {backend}")
            
            load_time = time.perf_counter() - start_load
            print(f"  ✓ Model loaded in {load_time:.1f}s")
            
            print(f"\n  [Running imputation on {len(rows)} rows...]")
            for row in tqdm(rows, desc=f"  Imputing"):
                persona_text = row.persona or (persona_map.get(row.persona_card_id) if persona_map and row.persona_card_id else "")
                params = GenParams(
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    repetition_penalty=cfg.repetition_penalty,
                    stop=cfg.stop,
                )
                start = time.perf_counter()
                # Build model-aware prompt per row
                input_prompt = None
                if cfg.prompt_style in ("t5", "auto") and "t5" in (cfg.model or "").lower():
                    input_prompt = build_t5_input(persona_text, row.context, row.placeholder_token)
                elif cfg.prompt_style == "fim":
                    ps = split_prefix_suffix(row.context, row.placeholder_token)
                    if ps is not None:
                        pre, suf = ps
                        tokens = ("<PRE>", "<SUF>", "<MID>", "<EOM>")
                        if cfg.fim_tokens:
                            parts = [p.strip() for p in cfg.fim_tokens.split(",")]
                            if len(parts) == 4:
                                tokens = tuple(parts)  # type: ignore[assignment]
                        input_prompt = build_fim_prompt(pre, suf, tokens)  # type: ignore[arg-type]
                    else:
                        input_prompt = build_impute_prompt(persona_text, row.instruction, row.context, row.placeholder_token)
                else:
                    # GPT-J completion variants: prefer completion prompts when requested
                    # Auto-detect board from persona for GPT4chan formatting
                    board = cfg.board_prefix
                    if board == "b" and row.persona_card_id:
                        board = get_board_prefix_for_persona(row.persona_card_id)
                    
                    if cfg.prompt_style == "gptj_completion":
                        input_prompt = build_gptj_completion_prompt(row.context, row.placeholder_token, board)
                    elif cfg.prompt_style == "gptj_completion_expected":
                        input_prompt = build_gptj_expected_prompt(row.expected_type, row.context, row.placeholder_token, board)
                    elif cfg.prompt_style == "gptj_instruction":
                        input_prompt = build_gptj_instruction_prompt(row.context, row.placeholder_token, board)
                    else:
                        input_prompt = build_impute_prompt(persona_text, row.instruction, row.context, row.placeholder_token)

                res = gen(input_prompt, params)
                end = time.perf_counter()
                
                aggregated.append({
                    "id": row.id,
                    "backend": backend,
                    "model_name_or_path": cfg.model,
                    "model_path": cfg.model_path,
                    "persona_card_id": row.persona_card_id,
                    "context": row.context,
                    "placeholder_token": row.placeholder_token,
                    "input_prompt": input_prompt,
                    "imputed_text": res.text,
                    "latency_ms": int((end - start) * 1000),
                    "expected_type": row.expected_type,
                    "num_input_tokens": res.num_input_tokens,
                    "num_output_tokens": res.num_output_tokens,
                    "tokens_per_sec": res.tokens_per_sec,
                    "perplexity": res.perplexity,
                    "logprobs": json.dumps(res.logprobs) if res.logprobs else None,
                    "error": None,
                })
            
            print(f"  ✓ Completed {len(rows)} imputations for this model\n")
    else:
        print(f"\n[3/5] Single model mode")
        print(f"  Backend: {args.backend}")
        print(f"  Model: {args.model or args.model_path}")
        print(f"  Device: {args.device}")
        print(f"  Dtype: {args.dtype}")
        
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
            offload_folder=args.offload_folder,
            prompt_style=args.prompt_style,
            fim_tokens=args.fim_tokens,
            board_prefix=getattr(args, 'board_prefix', 'b'),
        )
        
        print(f"\n  [Loading model...]")
        start_load = time.perf_counter()
        if cfg.backend == "transformers":
            gen = _adapter_transformers(cfg)
        elif cfg.backend == "llamacpp":
            if not cfg.model_path:
                raise ValueError("--model-path is required for llamacpp backend")
            gen = _adapter_llamacpp(cfg)
        elif cfg.backend == "http":
            if not cfg.base_url or not cfg.model:
                raise ValueError("--base-url and --model are required for http backend")
            gen = _adapter_http(cfg)
        else:
            raise ValueError(f"Unknown backend: {cfg.backend}")
        
        load_time = time.perf_counter() - start_load
        print(f"  ✓ Model loaded in {load_time:.1f}s")

        print(f"\n  [Running imputation on {len(rows)} rows...]")
        aggregated = []
        for row in tqdm(rows, desc=f"  Imputing"):
            persona_text = row.persona or (persona_map.get(row.persona_card_id) if persona_map and row.persona_card_id else "")
            params = GenParams(
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repetition_penalty=cfg.repetition_penalty,
                stop=cfg.stop,
            )
            start = time.perf_counter()
            # Build model-aware prompt per row
            if cfg.prompt_style in ("t5", "auto") and "t5" in (cfg.model or "").lower():
                input_prompt = build_t5_input(persona_text, row.context, row.placeholder_token)
            elif cfg.prompt_style == "fim":
                ps = split_prefix_suffix(row.context, row.placeholder_token)
                if ps is not None:
                    pre, suf = ps
                    tokens = ("<PRE>", "<SUF>", "<MID>", "<EOM>")
                    if cfg.fim_tokens:
                        parts = [p.strip() for p in cfg.fim_tokens.split(",")]
                        if len(parts) == 4:
                            tokens = tuple(parts)  # type: ignore[assignment]
                    input_prompt = build_fim_prompt(pre, suf, tokens)  # type: ignore[arg-type]
                else:
                    input_prompt = build_impute_prompt(persona_text, row.instruction, row.context, row.placeholder_token)
            else:
                # Decoder-only single-model path
                # Auto-detect board from persona for GPT4chan formatting
                board = cfg.board_prefix
                if board == "b" and row.persona_card_id:
                    board = get_board_prefix_for_persona(row.persona_card_id)
                
                if cfg.prompt_style == "gptj_completion":
                    input_prompt = build_gptj_completion_prompt(row.context, row.placeholder_token, board)
                elif cfg.prompt_style == "gptj_completion_expected":
                    input_prompt = build_gptj_expected_prompt(row.expected_type, row.context, row.placeholder_token, board)
                elif cfg.prompt_style == "gptj_instruction":
                    input_prompt = build_gptj_instruction_prompt(row.context, row.placeholder_token, board)
                else:
                    input_prompt = build_impute_prompt(persona_text, row.instruction, row.context, row.placeholder_token)

            res = gen(input_prompt, params)
            end = time.perf_counter()
            
            # Build the full prompt for logging
            full_prompt = input_prompt
            
            aggregated.append({
                "id": row.id,
                "backend": cfg.backend,
                "model_name_or_path": cfg.model,
                "model_path": cfg.model_path,
                "persona_card_id": row.persona_card_id,
                "context": row.context,
                "placeholder_token": row.placeholder_token,
                "input_prompt": full_prompt,
                "imputed_text": res.text,
                "latency_ms": int((end - start) * 1000),
                "expected_type": row.expected_type,
                "num_input_tokens": res.num_input_tokens,
                "num_output_tokens": res.num_output_tokens,
                "tokens_per_sec": res.tokens_per_sec,
                "perplexity": res.perplexity,
                "logprobs": json.dumps(res.logprobs) if res.logprobs else None,
                "error": None,
            })

    print(f"\n[4/5] Writing outputs...")
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(aggregated)
    df.to_csv(out_csv, index=False)
    print(f"  ✓ CSV written: {out_csv} ({len(df)} rows, {len(df.columns)} columns)")
    
    if args.html:
        Path(args.html).write_text(df.to_html(index=False))
        print(f"  ✓ HTML written: {args.html}")
    
    print(f"\n[5/5] Summary")
    print(f"  Total imputations: {len(aggregated)}")
    print(f"  Output CSV: {out_csv}")
    if args.html:
        print(f"  Output HTML: {args.html}")
    
    print(f"\n{'='*70}")
    print("✓ IMPUTATION COMPLETE")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generative imputation runner with persona and instructions.")
    parser.add_argument("--backend", choices=["transformers", "llamacpp", "http"], required=True)
    parser.add_argument("--model", type=str, help="HF model id or server model name")
    parser.add_argument("--model-path", type=str, help="Path to GGUF for llama.cpp")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--html", type=str)
    parser.add_argument("--models-config", type=str, help="JSON file listing models to impute")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop", nargs="*", default=["\n\n", "PERSONA:", "TASK:", "CONTEXT:"])
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--base-url", type=str)
    parser.add_argument("--api-key", type=str, default="sk-no-key")
    parser.add_argument("--persona-file", type=str, help="JSON mapping from persona_card_id to full persona string")
    parser.add_argument("--offload-folder", type=str, default="open_models_testing/offload", help="Folder for disk offloading if model doesn't fit in RAM")
    parser.add_argument("--prompt-style", type=str, default="auto", choices=["auto", "t5", "fim", "instruction", "gptj_strict", "gptj_completion", "gptj_completion_expected", "gptj_instruction", "chat_llama3", "chat_chatml", "gpt4chan_board"], help="Prompting strategy per model family")
    parser.add_argument("--fim-tokens", type=str, help="Comma-separated FIM tokens: <PRE>,<SUF>,<MID>,<EOM>")
    parser.add_argument("--board-prefix", type=str, default="b", choices=["b", "pol", "r9k", "fit", "adv"], help="4chan board prefix for GPT4chan models (auto-detects from persona if 'b')")
    parser.add_argument("--prepare-gptj-variant", type=str, choices=["completion_endonly", "expected_endonly"], help="Prepare a derived dataset for GPT-J experiments and exit")
    parser.add_argument("--prepare-out", type=str, help="Path to write the derived dataset JSON when using --prepare-gptj-variant")
    args = parser.parse_args()
    # Dataset preparation mode
    if args.prepare_gptj_variant:
        src_rows = _load_dataset(Path(args.dataset))
        derived = _derive_gptj_variant_records(src_rows, args.prepare_gptj_variant)
        if not args.prepare_out:
            raise SystemExit("--prepare-out is required when using --prepare-gptj-variant")
        out_path = Path(args.prepare_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(derived, ensure_ascii=False, indent=2))
        print(f"Prepared {len(derived)} records -> {out_path}")
        return
    run(args)


if __name__ == "__main__":
    main()


