"""
llama_adapter.py
================
Thin adapter layer that makes a llama-cpp-python ``Llama`` instance look like
the HuggingFace ``(AutoModelForCausalLM, AutoTokenizer)`` pair used throughout
the pipeline.

Only two functions in the codebase touch the LLM backend directly:
  - ``initialize_model()`` in extract_v2.py  → calls ``initialize_llm_llamacpp()``
  - ``llm_generate()``   in extract_v2.py  → uses ``model._llm.create_chat_completion()``

Everything else (consolidate.py, deep_extract.py, preprocess_v2.py, extract_v2.py
token-budget loops) uses only ``tokenizer.encode()`` / ``tokenizer.decode()``,
which ``LlamaCppTokenizer`` satisfies via llama-cpp's ``tokenize()`` /
``detokenize()``.  No other files need to change.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal model.config stub
# ──────────────────────────────────────────────────────────────────────────────

class _FakeLlamaConfig:
    """
    Stand-in for ``model.config`` accessed by ``gpu_optimizer.py``'s
    ``calculate_optimal_batch_size()`` and ``estimate_chunk_memory_requirement()``.
    Those code paths are not active when using the llamacpp backend, but the
    attribute access must not raise AttributeError.  Values approximate Mistral-7B.
    """
    hidden_size = 4096
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128


# ──────────────────────────────────────────────────────────────────────────────
# LlamaCppTokenizer
# ──────────────────────────────────────────────────────────────────────────────

class LlamaCppTokenizer:
    """
    Duck-typed wrapper around a ``llama_cpp.Llama`` instance that satisfies the
    HuggingFace ``AutoTokenizer`` API surface used in this codebase.

    API surface covered
    -------------------
    encode(text, add_special_tokens=False) -> List[int]
        Used in ~12 places for token counting and text truncation.

    decode(token_ids, skip_special_tokens=True) -> str
        Used in ~6 places to recover text after slicing a token list.

    pad_token_id  (int)   — no dedicated pad token; EOS is reused
    eos_token_id  (int)
    pad_token     (str)   — non-None sentinel so callers skip the "is None" guard

    apply_chat_template(...)  — raises; llm_generate() branches before calling this
    __call__(...)             — raises; llm_generate() branches before calling this
    """

    def __init__(self, llm) -> None:
        self._llm = llm
        # Cache once at init — avoids repeated C-level calls on every token count
        self.eos_token_id: int = llm.token_eos()
        self.pad_token_id: int = llm.token_eos()   # llama.cpp has no dedicated pad token
        self.pad_token: str = "<pad>"               # non-None → callers skip the None check

    # ── token counting / truncation ────────────────────────────────────────

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Tokenize *text* and return a list of integer token IDs.

        Maps to ``llm.tokenize(text.encode("utf-8"), add_bos=False)``.
        ``add_special_tokens`` is accepted for API compatibility but ignored —
        llama-cpp does not expose that granularity through the Python binding,
        and callers always pass ``add_special_tokens=False`` anyway.
        """
        return self._llm.tokenize(text.encode("utf-8"), add_bos=False)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """
        Decode a list (or 1-D tensor) of token IDs back to a string.

        Maps to ``llm.detokenize(token_ids).decode("utf-8")``.
        ``skip_special_tokens`` is accepted for API compatibility; llama-cpp's
        ``detokenize()`` does not inject special tokens into its output, so the
        flag has no practical effect here.
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        raw: bytes = self._llm.detokenize(token_ids)
        return raw.decode("utf-8", errors="replace")

    # ── generation helpers (should never be reached in llamacpp branch) ────

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ) -> str:
        raise NotImplementedError(
            "LlamaCppTokenizer.apply_chat_template() must not be called when "
            "LLM_BACKEND == 'llamacpp'.  llm_generate() must branch BEFORE "
            "this point and call model._llm.create_chat_completion() directly."
        )

    def __call__(self, text, return_tensors=None, padding=False,
                 truncation=False, max_length=None, **kwargs):
        raise NotImplementedError(
            "LlamaCppTokenizer.__call__(text, return_tensors='pt', ...) must "
            "not be called when LLM_BACKEND == 'llamacpp'.  llm_generate() "
            "must branch before encoding inputs as tensors."
        )


# ──────────────────────────────────────────────────────────────────────────────
# LlamaCppModel
# ──────────────────────────────────────────────────────────────────────────────

class LlamaCppModel:
    """
    Duck-typed wrapper around a ``llama_cpp.Llama`` instance that satisfies the
    HuggingFace ``AutoModelForCausalLM`` API surface used in this codebase.

    The actual generation is done inside ``llm_generate()`` via
    ``model._llm.create_chat_completion()`` — ``model.generate()`` is never
    called from the llamacpp branch.  This class exists so the ``(model, tokenizer)``
    pair can be passed through all existing function signatures unchanged.

    API surface covered
    -------------------
    device         str   — returned as "cuda" (inputs.to(model.device) is in the
                           transformers branch which is not reached)
    config         obj   — _FakeLlamaConfig stub
    eval()               — no-op
    cpu()                — no-op (llama-cpp manages its own memory)
    generate(...)        — raises; must not be called in llamacpp branch
    """

    def __init__(self, llm) -> None:
        self._llm = llm
        self.device = "cuda"
        self.config = _FakeLlamaConfig()

    def eval(self) -> "LlamaCppModel":
        """No-op — called once at load time in initialize_model()."""
        return self

    def cpu(self) -> "LlamaCppModel":
        """
        No-op — called by gpu_optimizer.unload_model().
        llama-cpp deallocates GPU memory when the Llama object is garbage-collected.
        """
        return self

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "LlamaCppModel.generate() must not be called when "
            "LLM_BACKEND == 'llamacpp'.  Use model._llm.create_chat_completion()."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

def initialize_llm_llamacpp(
    model_path: str,
    n_gpu_layers: int = -1,
    n_ctx: int = 12000,
    n_batch: int = 2048,
    verbose: bool = False,
) -> Tuple[LlamaCppModel, LlamaCppTokenizer]:
    """
    Load a GGUF model via llama-cpp-python and return a ``(model, tokenizer)``
    pair that is a drop-in replacement for the HuggingFace pair.

    Parameters
    ----------
    model_path : str
        Absolute path to the ``.gguf`` file.
    n_gpu_layers : int
        Number of transformer layers to offload to GPU.  ``-1`` offloads all.
    n_ctx : int
        Context window size in tokens.  Must be ≥ the longest prompt you send.
        12 000 comfortably covers 3×2 000-token chunks + system prompt + output.
    n_batch : int
        Prompt processing batch size (higher = faster prefill, more VRAM during
        prompt evaluation).
    verbose : bool
        Whether to enable llama.cpp's built-in progress logging.

    Returns
    -------
    (LlamaCppModel, LlamaCppTokenizer)
        Both wrappers share the same underlying ``Llama`` instance.

    Deployment note
    ---------------
    Install with CUDA support on the A100 server::

        CMAKE_ARGS="-DLLAMA_CUDA=on" pip install "llama-cpp-python>=0.2.90"
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed.  Install it with:\n"
            '  CMAKE_ARGS="-DLLAMA_CUDA=on" pip install "llama-cpp-python>=0.2.90"'
        ) from exc

    logger.info(f"Loading GGUF model: {model_path}")
    logger.info(f"  n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, n_batch={n_batch}")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        verbose=verbose,
    )

    model = LlamaCppModel(llm)
    tokenizer = LlamaCppTokenizer(llm)

    logger.info("GGUF model loaded successfully")
    return model, tokenizer
