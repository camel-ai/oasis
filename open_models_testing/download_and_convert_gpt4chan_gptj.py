from __future__ import annotations

"""Download and convert GPT-4chan (GPT-J 6B) archive to a HF-style folder.

Steps:
1) Download the archive from the given URL (default: the provided archive.org link)
2) Extract the archive to a temporary directory
3) Create a Hugging Face Transformers-compatible directory with:
   - config.json (GPT-J 6B config)
   - tokenizer.json/tokenizer_config.json/special_tokens_map.json
   - pytorch_model.bin (or safetensors) weights

Usage:
poetry run python3 open_models_testing/download_and_convert_gpt4chan_gptj.py \
  --url https://archive.org/download/gpt4chan_model/gpt4chan_model.tar.gz \
  --out-dir open_models_testing/models/gpt4chan_gptj_hf

Note: This is a best-effort converter. If the archive already contains a HF-
formatted directory, we will simply move/copy it. Otherwise, we attempt to
create minimal config/tokenizer files referencing GPT-J 6B defaults.
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
from huggingface_hub import snapshot_download

# Track last printed progress line length for clean overwrites
_LAST_LINE_LEN = 0


DEFAULT_URL = "https://archive.org/download/gpt4chan_model/gpt4chan_model.tar.gz"
DEFAULT_CONFIG_URL = (
    "https://raw.githubusercontent.com/Aspie96/gpt-4chan-model/float16/config.json"
)


def _human(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def _print_step(msg: str) -> None:
    print(msg, flush=True)


def _print_progress(prefix: str, done: int, total: Optional[int], start_ts: float) -> None:
    now = time.time()
    elapsed = max(now - start_ts, 1e-6)
    rate = done / elapsed
    if total and total > 0:
        pct = done / total
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "#" * filled + "." * (bar_len - filled)
        line = (
            f"{prefix} [{bar}] {pct*100:6.2f}%  "
            f"{_human(done)}/{_human(total)}  {_human(rate)}/s"
        )
    else:
        line = f"{prefix} {_human(done)} downloaded  {_human(rate)}/s"

    # Overwrite the same line cleanly, padding if new line is shorter
    global _LAST_LINE_LEN
    pad = max(0, _LAST_LINE_LEN - len(line))
    sys.stdout.write("\r" + line + (" " * pad))
    sys.stdout.flush()
    _LAST_LINE_LEN = len(line)


def _download(url: str, dest: Path) -> None:
    global _LAST_LINE_LEN
    dest.parent.mkdir(parents=True, exist_ok=True)
    _print_step(f"[1/5] Downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0) or 0)
        start = time.time()
        bytes_done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_done += len(chunk)
                _print_progress("Downloading", bytes_done, total, start)
        # Final newline after completion and reset last line length
        sys.stdout.write("\n")
        sys.stdout.flush()
        _LAST_LINE_LEN = 0


def _extract(archive_path: Path, out_dir: Path) -> None:
    global _LAST_LINE_LEN
    out_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name
    _print_step(f"[2/5] Extracting: {name}")
    if name.endswith((".tar.gz", ".tar.xz", ".tar.zst", ".tar")):
        # Try multiple tar modes
        mode = "r:gz" if name.endswith(".tar.gz") else ("r:xz" if name.endswith(".tar.xz") else ("r:" if name.endswith(".tar") else "r:*"))
        with tarfile.open(archive_path, mode) as tar:
            members = tar.getmembers()
            total = len(members)
            start = time.time()
            for i, m in enumerate(members, 1):
                tar.extract(m, path=out_dir)
                _print_progress("Extracting", i, total, start)
            sys.stdout.write("\n")
            sys.stdout.flush()
            _LAST_LINE_LEN = 0
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.infolist()
            total = len(members)
            start = time.time()
            for i, m in enumerate(members, 1):
                zf.extract(m, path=out_dir)
                _print_progress("Extracting", i, total, start)
            sys.stdout.write("\n")
            sys.stdout.flush()
            _LAST_LINE_LEN = 0
    else:
        raise ValueError(f"Unsupported archive format: {name}")


def _fetch_identifier_metadata(identifier: str) -> dict:
    url = f"https://archive.org/metadata/{identifier}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _select_archive_from_metadata(meta: dict) -> Optional[str]:
    files = meta.get("files", [])
    # Prefer tar.gz, tar.xz, tar.zst, tar, zip, 7z
    preferred_exts = [".tar.gz", ".tar.xz", ".tar.zst", ".tar", ".zip", ".7z"]
    # First pass: exact preferred extensions
    for ext in preferred_exts:
        for f in files:
            name = f.get("name", "")
            if name.endswith(ext):
                return name
    # Fallback: largest file
    if files:
        largest = max(files, key=lambda f: int(f.get("size", 0) or 0))
        return largest.get("name")
    return None


def _select_direct_weight_from_metadata(meta: dict) -> Optional[str]:
    files = meta.get("files", [])
    for f in files:
        name = f.get("name", "")
        if name == "pytorch_model.bin":
            return name
    return None


def _download_config(config_url: str, out_dir: Path) -> bool:
    try:
        (out_dir / "config.json").write_bytes(requests.get(config_url, timeout=60).content)
        return True
    except Exception:
        return False


def _ensure_tokenizer_files(out_dir: Path) -> None:
    needed: List[str] = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    if all((out_dir / n).exists() for n in needed):
        return
    try:
        _print_step("[4/5] Fetching tokenizer files from Hugging Face (gpt2)")
        cache_dir = snapshot_download(
            repo_id="gpt2",
            allow_patterns=needed,
        )
        cache_path = Path(cache_dir)
        for n in needed:
            src = cache_path / n
            if src.exists():
                shutil.copy2(src, out_dir / n)
    except Exception:
        # Fall back to minimal stubs if download not possible
        _print_step("[4/5] Tokenizer fetch failed, writing minimal tokenizer stubs")
        if not (out_dir / "tokenizer_config.json").exists():
            tok_conf = {
                "do_lower_case": False,
                "model_max_length": 2048,
                "padding_side": "left",
                "truncation_side": "right",
                "unk_token": "<|unknown|>",
            }
            (out_dir / "tokenizer_config.json").write_text(json.dumps(tok_conf, indent=2))
        if not (out_dir / "special_tokens_map.json").exists():
            special_tokens_map = {
                "eos_token": {"content": "", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
                "bos_token": {"content": "", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
                "unk_token": {"content": "<|unknown|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
            }
            (out_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2))


def _maybe_find_hf_dir(tmp_root: Path) -> Optional[Path]:
    # Heuristic: look for a directory that already has a config.json and tokenizer files
    for p in tmp_root.rglob("config.json"):
        if (p.parent / "tokenizer.json").exists() or (p.parent / "tokenizer_config.json").exists():
            return p.parent
    return None


def _write_gptj_config(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["GPTJForCausalLM"],
        "model_type": "gptj",
        "n_embd": 4096,
        "n_head": 16,
        "n_layer": 28,
        "n_positions": 2048,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "rotary": True,
        "attn_dropout": 0.0,
        "embd_dropout": 0.0,
        "resid_dropout": 0.0,
        "layer_norm_epsilon": 1e-5,
    }
    (target_dir / "config.json").write_text(json.dumps(config, indent=2))


def _write_tokenizer_stub(target_dir: Path) -> None:
    # Use GPT-2 tokenizer settings (common for GPT-J)
    special_tokens_map = {
        "eos_token": {"content": "", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "bos_token": {"content": "", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "unk_token": {"content": "<|unknown|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
    }
    (target_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2))
    tok_conf = {
        "do_lower_case": False,
        "model_max_length": 2048,
        "padding_side": "left",
        "truncation_side": "right",
        "unk_token": "<|unknown|>",
    }
    (target_dir / "tokenizer_config.json").write_text(json.dumps(tok_conf, indent=2))
    # Note: Ideally we would build tokenizer.json from a vocab/merges; for GPT-J, GPT-2 BPE is typical.
    # If tokenizer.json is present in the archive, we'll copy it over instead.


def _copy_if_exists(src_dir: Path, dst_dir: Path, names: list[str]) -> None:
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def _finalize_from_archive(extracted_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    # Look for weights and tokenizer files in the extracted content
    weight_names = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model-00001-of-00001.bin",
    ]
    tok_names = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]

    # Copy found files to target
    for path in extracted_dir.rglob("*"):
        if path.is_file():
            if path.name in weight_names:
                shutil.copy2(path, target_dir / path.name)
            elif path.name in tok_names:
                shutil.copy2(path, target_dir / path.name)

    # Ensure config/tokenizer stubs exist if not found
    if not (target_dir / "config.json").exists():
        _write_gptj_config(target_dir)
    if not (target_dir / "tokenizer.json").exists() and not (target_dir / "tokenizer_config.json").exists():
        _write_tokenizer_stub(target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert GPT-4chan (GPT-J 6B) to HF folder.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Direct archive URL. If 404, try --identifier.")
    parser.add_argument("--identifier", type=str, default="gpt4chan_model", help="Archive.org identifier, used if URL fails.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--config-url", type=str, default=DEFAULT_CONFIG_URL, help="URL to a GPT-J config.json (fallback if config missing).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        archive_path = tmp / "gpt4chan_archive"

        # Try direct URL first (.tar.gz); if 404, try direct file (pytorch_model.bin) via metadata
        tried_direct = False
        try:
            tried_direct = True
            _download(args.url, archive_path.with_suffix(".tar.gz"))
            archive_path = archive_path.with_suffix(".tar.gz")
        except Exception:
            # Try metadata-based selection
            meta = _fetch_identifier_metadata(args.identifier)
            weight_name = _select_direct_weight_from_metadata(meta)
            if weight_name:
                # Directly download the 22.5G weight file
                weight_url = f"https://archive.org/download/{args.identifier}/{weight_name}"
                weight_path = tmp / "pytorch_model.bin"
                _download(weight_url, weight_path)
                extracted_root = tmp / "extracted"
                extracted_root.mkdir(parents=True, exist_ok=True)
                # Place the weight in a folder and ensure config/tokenizer are present
                target_dir = extracted_root / "gpt4chan_gptj"
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(weight_path, target_dir / "pytorch_model.bin")
                # Try download config.json from GitHub; fallback to stub
                _print_step("[3/5] Fetching config.json from GitHub float16 branch")
                if not _download_config(args.config_url, target_dir):
                    # Minimal GPT-J config if GitHub fetch fails
                    cfg = {
                        "architectures": ["GPTJForCausalLM"],
                        "model_type": "gptj",
                        "n_embd": 4096,
                        "n_head": 16,
                        "n_layer": 28,
                        "n_positions": 2048,
                        "bos_token_id": 50256,
                        "eos_token_id": 50256,
                    }
                    (target_dir / "config.json").write_text(json.dumps(cfg, indent=2))
                # Ensure tokenizer files
                _ensure_tokenizer_files(target_dir)
            else:
                # Fallback to archive selection
                name = _select_archive_from_metadata(meta)
                if not name:
                    raise RuntimeError("Could not locate an archive file via metadata.")
                meta_url = f"https://archive.org/download/{args.identifier}/{name}"
                suffix = ".zip" if name.endswith(".zip") else (".tar.gz" if name.endswith(".tar.gz") else ".tar")
                target = archive_path.with_suffix(suffix)
                _download(meta_url, target)
                archive_path = target
                extracted_root = tmp / "extracted"
                _extract(archive_path, extracted_root)

        # If a HF-like dir exists, prefer it
        maybe_hf = _maybe_find_hf_dir(extracted_root)
        if 'extracted_root' not in locals():
            extracted_root = tmp / "extracted"
            extracted_root.mkdir(exist_ok=True)
        if maybe_hf:
            # Copy entire directory over
            if out_dir.exists():
                for item in maybe_hf.iterdir():
                    dest = out_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
            else:
                shutil.copytree(maybe_hf, out_dir)
        else:
            _finalize_from_archive(extracted_root, out_dir)

    _print_step(f"[5/5] Prepared HF directory at: {out_dir}")


if __name__ == "__main__":
    main()


