"""Utilities and scripts for testing open-source LLM backends.

This package contains simple, typed loaders and runners to evaluate:
- Transformers (cuda/mps/cpu) causal and masked language models
- llama.cpp (GGUF) via llama-cpp-python
- HTTP OpenAI-compatible (e.g., vLLM) endpoints

The goal is feasibility testing for text completions and masked token filling,
with outputs saved to CSV and optional HTML summaries.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"


