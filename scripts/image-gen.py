#!/usr/bin/env python3
# t2i_mps.py
# Minimal Diffusers runner for Apple Silicon (MPS) with toggles for attention slicing,
# optional CPU offload, and safety checker disabled where applicable.

import argparse
from typing import Optional

import torch
# Pipelines
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusionPipeline
from PIL import Image

# Kandinsky 2.2 uses a prior + main pipeline
try:
    from diffusers import KandinskyV22Pipeline  # type: ignore
    from diffusers import KandinskyV22PriorPipeline
    HAVE_KANDINSKY = True
except Exception:
    HAVE_KANDINSKY = False

# HiDream-I1 Fast custom pipeline (published by HiDream-ai)
try:
    from diffusers import HiDreamImagePipeline  # type: ignore
    HAVE_HIDREAM = True
except Exception:
    HAVE_HIDREAM = False


REPOS = {
    "sd21": "stabilityai/stable-diffusion-2-1",
    "flux": "black-forest-labs/FLUX.1-dev",
    "kandinsky": "kandinsky-community/kandinsky-2-2-decoder",
    "kandinsky-prior": "kandinsky-community/kandinsky-2-2-prior",
    "hidream": "HiDream-ai/HiDream-I1-Fast",
}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return gen


def _force_cpu(pipe: DiffusionPipeline) -> DiffusionPipeline:
    pipe = pipe.to(torch.device("cpu"))
    try:
        pipe._execution_device = torch.device("cpu")  # type: ignore[attr-defined]
    except Exception:
        pass
    return pipe


def build_pipeline(
    model: str,
    fp16: bool,
    disable_safety: bool,
    use_mps: bool,
    hf_token: Optional[str],
    local_files_only: bool,
    hidream_repo_override: Optional[str],
    hidream_tokenizer_id: Optional[str],
    hidream_text_encoder_id: Optional[str],
) -> tuple[DiffusionPipeline, Optional[DiffusionPipeline]]:
    dtype = torch.float16 if fp16 else torch.float32
    device = pick_device() if use_mps else torch.device("cpu")

    if model == "sd21":
        # Disable safety checker by passing safety_checker=None (omit otherwise)
        sd_kwargs = {"torch_dtype": dtype}
        if disable_safety:
            sd_kwargs["safety_checker"] = None
        pipe = StableDiffusionPipeline.from_pretrained(
            REPOS["sd21"],
            token=hf_token,
            local_files_only=local_files_only,
            **sd_kwargs,
        )
        pipe = pipe.to(device)
        return pipe, None

    if model == "flux":
        # Prefer float16 on MPS; on CPU bfloat16 (if fp16 requested) else float32
        flux_dtype = torch.float16 if (device.type == "mps" and fp16) else (torch.bfloat16 if fp16 else torch.float32)
        pipe = FluxPipeline.from_pretrained(
            REPOS["flux"],
            torch_dtype=flux_dtype,
            token=hf_token,
            local_files_only=local_files_only,
        )
        pipe = pipe.to(device)
        return pipe, None

    if model == "kandinsky":
        # Kandinsky has known issues with half + MPS; prefer CPU/float32
        if device.type == "mps":
            device = torch.device("cpu")
        dtype = torch.float32
        if not HAVE_KANDINSKY:
            raise RuntimeError("KandinskyV22Pipeline not available in your diffusers install.")
        prior = KandinskyV22PriorPipeline.from_pretrained(
            REPOS["kandinsky-prior"],
            torch_dtype=dtype,
            token=hf_token,
            local_files_only=local_files_only,
        )
        prior = _force_cpu(prior)
        pipe = KandinskyV22Pipeline.from_pretrained(
            REPOS["kandinsky"],
            torch_dtype=dtype,
            token=hf_token,
            local_files_only=local_files_only,
        )
        pipe = _force_cpu(pipe)
        return pipe, prior

    if model == "hidream":
        # Try native pipeline class first; otherwise trust remote code from repo
        repo_id = hidream_repo_override or REPOS["hidream"]
        if HAVE_HIDREAM:
            extra_kwargs = {}
            # Lazily import tokenizer/encoder if provided (some variants require Llama 3.1)
            if hidream_tokenizer_id and hidream_text_encoder_id:
                try:
                    from transformers import (  # type: ignore
                        AutoModelForCausalLM, AutoTokenizer)
                    tokenizer_4 = AutoTokenizer.from_pretrained(hidream_tokenizer_id, use_fast=True, token=hf_token, local_files_only=local_files_only)
                    text_encoder_4 = AutoModelForCausalLM.from_pretrained(hidream_text_encoder_id, output_hidden_states=True, output_attentions=True, token=hf_token, local_files_only=local_files_only)
                    extra_kwargs.update({"tokenizer_4": tokenizer_4, "text_encoder_4": text_encoder_4})
                except Exception:
                    pass
            pipe = HiDreamImagePipeline.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                token=hf_token,
                local_files_only=local_files_only,
                **extra_kwargs,
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                token=hf_token,
                local_files_only=local_files_only,
            )
        pipe = pipe.to(device)
        return pipe, None

    raise ValueError(f"Unknown model selector: {model}")


def maybe_enable_optimisations(pipe: DiffusionPipeline, use_attn_slicing: bool, use_cpu_offload: bool) -> None:
    # Attention slicing
    if use_attn_slicing and hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    # CPU offload (uses accelerate hooks; effectiveness may vary on MPS)
    if use_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        try:
            device_type = getattr(getattr(pipe, "device", torch.device("cpu")), "type", "cpu")
            if device_type == "cuda":
                pipe.enable_model_cpu_offload()
        except Exception:
            pass

    # VAE slicing can further reduce memory in SD-like pipelines
    if use_attn_slicing and hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass


def run_sd21(pipe: DiffusionPipeline, prompt: str, negative_prompt: Optional[str], steps: int, guidance: float, height: int, width: int, generator: Optional[torch.Generator]) -> Image.Image:
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    return out


def run_flux(pipe: DiffusionPipeline, prompt: str, negative_prompt: Optional[str], steps: int, guidance: float, height: int, width: int, generator: Optional[torch.Generator]) -> Image.Image:
    kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "generator": generator,
    }
    try:
        kwargs["negative_prompt"] = negative_prompt
    except Exception:
        pass
    out = pipe(**kwargs).images[0]
    return out


def run_kandinsky(pipe: DiffusionPipeline, prior: DiffusionPipeline, prompt: str, negative_prompt: Optional[str], steps: int, guidance: float, height: int, width: int, generator: Optional[torch.Generator]) -> Image.Image:
    # Kandinsky 2.2 uses a text prior to get CLIP image embeddings
    prior_output = prior(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance,
        num_inference_steps=max(1, steps // 4),
        generator=generator,
    )
    image = pipe(
        image_embeds=prior_output.image_embeds,
        negative_image_embeds=prior_output.negative_image_embeds,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    return image


def run_hidream(pipe: DiffusionPipeline, prompt: str, negative_prompt: Optional[str], steps: int, guidance: float, height: int, width: int, generator: Optional[torch.Generator]) -> Image.Image:
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Tiny MPS-ready Diffusers runner with safety-off and memory toggles")
    p.add_argument("--model", required=True, choices=["sd21", "flux", "kandinsky", "hidream"], help="Which model to run")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--negative-prompt", default=None, help="Negative prompt")
    p.add_argument("--steps", type=int, default=30, help="Inference steps")
    p.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (ignored by some models)")
    p.add_argument("--height", type=int, default=768, help="Output height")
    p.add_argument("--width", type=int, default=768, help="Output width")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--fp16", action="store_true", help="Use float16 where supported")
    p.add_argument("--mps", action="store_true", help="Force MPS if available")
    p.add_argument("--attention-slicing", action="store_true", help="Enable attention and VAE slicing")
    p.add_argument("--cpu-offload", action="store_true", help="Enable model CPU offload (via accelerate)")
    p.add_argument("--disable-safety", action="store_true", help="Disable safety checker where applicable")
    p.add_argument("--hf-token", default=None, help="Hugging Face token for gated models")
    p.add_argument("--local-files-only", action="store_true", help="Load models from local cache only")
    # HiDream advanced options (optional)
    p.add_argument("--hidream-repo", default=None, help="Override HiDream repo id (e.g., HiDream-ai/HiDream-I1-Full)")
    p.add_argument("--hidream-tokenizer", default=None, help="Tokenizer id for HiDream (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    p.add_argument("--hidream-text-encoder", default=None, help="Text encoder id for HiDream (same as tokenizer if using Llama)")
    p.add_argument("--out", default="out.png", help="Output path")

    args = p.parse_args()

    generator = seed_everything(args.seed)
    pipe, aux = build_pipeline(
        args.model,
        fp16=args.fp16,
        disable_safety=args.disable_safety,
        use_mps=args.mps,
        hf_token=args.hf_token,
        local_files_only=args.local_files_only,
        hidream_repo_override=args.hidream_repo,
        hidream_tokenizer_id=args.hidream_tokenizer,
        hidream_text_encoder_id=args.hidream_text_encoder,
    )
    maybe_enable_optimisations(pipe, use_attn_slicing=args.attention_slicing, use_cpu_offload=args.cpu_offload)

    if args.model == "sd21":
        image = run_sd21(pipe, args.prompt, args.negative_prompt, args.steps, args.guidance, args.height, args.width, generator)
    elif args.model == "flux":
        image = run_flux(pipe, args.prompt, args.negative_prompt, args.steps, args.guidance, args.height, args.width, generator)
    elif args.model == "kandinsky":
        image = run_kandinsky(pipe, aux, args.prompt, args.negative_prompt, args.steps, args.guidance, args.height, args.width, generator)  # type: ignore[arg-type]
    elif args.model == "hidream":
        image = run_hidream(pipe, args.prompt, args.negative_prompt, args.steps, args.guidance, args.height, args.width, generator)
    else:
        raise ValueError("Unhandled model")

    image.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    # Better matmul perf on Apple GPUs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
