from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from .loaders import TransformersLoadConfig, load_transformers_masked_lm


class MaskRow(BaseModel):
    id: str
    text: str


@dataclass
class RunConfig:
    model: str
    device: str = "auto"
    dtype: Optional[str] = "auto"
    top_k: int = 5


def _load_dataset(path: Path) -> List[MaskRow]:
    data = json.loads(Path(path).read_text())
    rows: List[MaskRow] = []
    for obj in data:
        try:
            rows.append(MaskRow(**obj))
        except ValidationError as e:
            raise ValueError(f"Invalid dataset row: {obj}") from e
    return rows


def _to_model_mask(text: str, tokenizer) -> str:
    """Replace generic <mask> with tokenizer.mask_token if present.

    If the tokenizer has no mask token (e.g., many decoder-only LMs), we raise.
    """
    mask_token = getattr(tokenizer, "mask_token", None)
    if not mask_token:
        raise ValueError("This model/tokenizer does not define a mask token.")
    return text.replace("<mask>", mask_token).replace("[MASK]", mask_token)


def run(args: argparse.Namespace) -> None:
    cfg = RunConfig(model=args.model, device=args.device, dtype=args.dtype, top_k=args.top_k)
    rows = _load_dataset(Path(args.dataset))
    model, tok, device_str = load_transformers_masked_lm(
        TransformersLoadConfig(
            model_name_or_path=cfg.model,
            device=cfg.device,
            dtype=cfg.dtype,  # type: ignore[arg-type]
            trust_remote_code=True,
        )
    )

    import torch
    from torch.nn import functional as F

    results: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="fill-mask"):
        text = _to_model_mask(row.text, tok)
        enc = tok(text, return_tensors="pt")
        input_ids = enc.input_ids
        attn = enc.attention_mask

        device = torch.device("cuda" if device_str == "cuda" else ("mps" if device_str == "mps" else "cpu"))
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        mask_token_id = getattr(tok, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Tokenizer does not provide mask_token_id.")

        try:
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False)[:, 1]
        except Exception as e:
            raise ValueError("No mask token found in the input text.") from e

        # Only support single mask per row for simplicity
        if mask_positions.numel() != 1:
            raise ValueError("Please provide exactly one mask token per text row.")
        mask_pos = int(mask_positions.item())

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits  # [B, T, V]
            mask_logits = logits[0, mask_pos]
            probs = F.softmax(mask_logits, dim=-1)
            topk = torch.topk(probs, k=cfg.top_k)
        end = time.perf_counter()

        tokens = [int(t.item()) for t in topk.indices]
        scores = [float(s.item()) for s in topk.values]
        words = tok.convert_ids_to_tokens(tokens)
        latency_ms = int((end - start) * 1000)

        results.append({
            "id": row.id,
            "text": row.text,
            "tokenizer_mask_token": tok.mask_token,
            "topk_json": json.dumps([
                {"token": w, "token_id": t, "prob": p} for w, t, p in zip(words, tokens, scores)
            ]),
            "latency_ms": latency_ms,
            "error": None,
        })

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)

    if args.html:
        df = pd.DataFrame(results)
        Path(args.html).write_text(df.to_html(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run masked token fill using Transformers masked LMs.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--html", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()


