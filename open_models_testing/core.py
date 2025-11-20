from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class GenParams:
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None


@dataclass
class GenResult:
    text: str
    num_input_tokens: Optional[int] = None
    num_output_tokens: Optional[int] = None
    tokens_per_sec: Optional[float] = None
    logprobs: Optional[List[float]] = None
    perplexity: Optional[float] = None


def build_impute_prompt(persona_text: str, instruction: str, context: str,
                        placeholder_token: str) -> str:
    # More directive format for base models (non-instruction-tuned)
    # Clear delimiters and explicit "OUTPUT:" marker
    user_tmpl = (
        "PERSONA: {persona}\n\n"
        "TASK: {instruction}\n\n"
        "CONTEXT:\n{context}\n\n"
        "PLACEHOLDER TO REPLACE: {placeholder_token}\n\n"
        "YOUR REPLACEMENT TEXT (write ONLY the replacement, nothing else):\n"
    )
    return user_tmpl.format(
        persona=persona_text or "",
        instruction=instruction,
        context=context,
        placeholder_token=placeholder_token,
    )


def build_gptj_completion_prompt(context: str, placeholder_token: str, board_prefix: str = "") -> str:
    """For GPT-J completion: use only contexts where the placeholder occurs at end.

    If the context ends with the placeholder, remove it and let the model continue
    from there. Otherwise, fall back to instruction framing upstream.
    
    If board_prefix provided (e.g., "pol", "r9k"), formats for GPT4chan-8B board style.
    """
    if context.endswith(placeholder_token):
        base = context[: -len(placeholder_token)]
    else:
        base = context
    
    # Add board formatting if specified (for GPT4chan-8B)
    if board_prefix:
        return f"{board_prefix}<|start_header_id|>1<|end_header_id|>{base}\n<|start_header_id|>2<|end_header_id|>"
    
    return base


def build_gptj_expected_prompt(expected_type: str | None, context: str, placeholder_token: str, board_prefix: str = "") -> str:
    """Prefix with expected type label and continue as a completion prompt.

    Example: "Dehumanization / Identity Attack: The text..."
    For GPT4chan-8B with board: "pol<|start_header_id|>1<|end_header_id|>Type: text..."
    """
    label = (expected_type or "").strip()
    base_context = context[: -len(placeholder_token)] if context.endswith(placeholder_token) else context
    
    if label:
        content = f"{label}: {base_context}"
    else:
        content = base_context
    
    # Add board formatting if specified
    if board_prefix:
        return f"{board_prefix}<|start_header_id|>1<|end_header_id|>{content}\n<|start_header_id|>2<|end_header_id|>"
    
    return f"{label}: {base_context}" if label else base_context


def build_gptj_instruction_prompt(context: str, placeholder_token: str, board_prefix: str = "") -> str:
    """Instruction-following variant for GPT-J without chat tokens."""
    prefix = (
        "Task: Replace the placeholder in the text with the correct content.\n\n"
        "Text: "
    )
    base = f"{prefix}{context}\n\nReplacement:"
    
    # Board format not typically used with instruction style, but support it
    if board_prefix:
        return f"{board_prefix}<|start_header_id|>1<|end_header_id|>{base}\n<|start_header_id|>2<|end_header_id|>"
    
    return base


GenerateFn = Callable[[str, GenParams], GenResult]


def get_board_prefix_for_persona(persona_card_id: str | None) -> str:
    """Select appropriate 4chan board based on persona type.
    
    Returns board letter (b/pol/r9k/fit/adv) for GPT4chan-8B prompting.
    """
    if not persona_card_id:
        return "b"  # Random board (default)
    
    card = persona_card_id.lower()
    
    # Incel personas → /r9k/ (incel culture origin)
    if "incel" in card:
        return "r9k"
    
    # Conspiracy/rumour personas → /pol/ (conspiracy central)
    if "rumour" in card or "conspiracy" in card:
        return "pol"
    
    # Eating disorder personas → /fit/ (body image discussions)
    if "edrisk" in card or "anorexia" in card:
        return "fit"
    
    # Default: /b/ (random)
    return "b"


def impute_once(
    persona_text: str,
    context: str,
    instruction: str,
    placeholder_token: str,
    gen_fn: GenerateFn,
    params: GenParams,
) -> GenResult:
    prompt = build_impute_prompt(persona_text, instruction, context,
                                 placeholder_token)
    return gen_fn(prompt, params)


# -------- T5 Sentinel utilities ---------

def build_t5_input(persona_text: str, context: str, placeholder_token: str) -> str:
    """Build a T5 span corruption input with <extra_id_0> replacing the placeholder.

    Keeps persona in the input to condition generation while leveraging sentinel
    tokens for the actual masked span.
    """
    sentinel = "<extra_id_0>"
    replaced_context = context.replace(placeholder_token, sentinel)
    if persona_text:
        return f"Persona: {persona_text}\n\n{replaced_context}"
    return replaced_context


def parse_t5_output(output_text: str) -> str:
    """Extract the span following <extra_id_0> up to next sentinel or end.

    The T5 output commonly follows the pattern: "<extra_id_0> replacement <extra_id_1> ...".
    """
    sentinel = "<extra_id_0>"
    if sentinel not in output_text:
        # If model didn't emit sentinel, return the raw text as best-effort
        return output_text.strip()
    rest = output_text.split(sentinel, 1)[1]
    # Stop at the next sentinel if present
    for next_id in ["<extra_id_1>", "<extra_id_2>", "<extra_id_3>"]:
        if next_id in rest:
            rest = rest.split(next_id, 1)[0]
            break
    return rest.strip()


# -------- FIM utilities (decoder-only) ---------

def split_prefix_suffix(context: str, placeholder_token: str) -> Optional[tuple[str, str]]:
    if placeholder_token not in context:
        return None
    parts = context.split(placeholder_token, 1)
    return parts[0], parts[1]


def build_fim_prompt(prefix: str, suffix: str,
                     fim_tokens: tuple[str, str, str, str]) -> str:
    pre, suf, mid, _eom = fim_tokens
    return f"{pre} {prefix} {suf} {suffix} {mid}"


def parse_fim_output(generated_text: str,
                     fim_tokens: tuple[str, str, str, str]) -> str:
    _pre, _suf, mid, eom = fim_tokens
    if mid in generated_text:
        after_mid = generated_text.split(mid, 1)[1]
    else:
        after_mid = generated_text
    if eom in after_mid:
        return after_mid.split(eom, 1)[0].strip()
    # Fallback: stop at double newline or end
    return after_mid.split("\n\n", 1)[0].strip()


