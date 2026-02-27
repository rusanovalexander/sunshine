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
import struct
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Architectures known to require a llama.cpp build newer than what
# llama-cpp-python 0.3.16 (released Aug 2025) bundles.
_ARCH_NEEDS_NEWER_BUILD = {"mistral3"}


def _read_gguf_architecture(path: str) -> Optional[str]:
    """
    Read ``general.architecture`` from a GGUF file without loading the model.

    The KV section starts immediately after the fixed 24-byte header
    (magic 4B + version 4B + n_tensors 8B + n_kv 8B).  The first KV pair
    is always ``general.architecture`` for well-formed GGUF files.

    Returns the architecture string, or None if the file cannot be parsed.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            _version = struct.unpack("<I", f.read(4))[0]
            _n_tensors = struct.unpack("<Q", f.read(8))[0]
            _n_kv = struct.unpack("<Q", f.read(8))[0]
            # First KV: key length (8B) + key bytes + value type (4B) + value
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8", errors="replace")
            val_type = struct.unpack("<I", f.read(4))[0]
            if val_type != 8:           # 8 == GGUF_TYPE_STRING
                return None
            val_len = struct.unpack("<Q", f.read(8))[0]
            arch = f.read(val_len).decode("utf-8", errors="replace")
            return arch if key == "general.architecture" else None
    except Exception:
        return None


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

        CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python>=0.2.90"

    If the model uses an architecture not yet supported by the bundled llama.cpp
    (e.g. ``mistral3`` in llama-cpp-python 0.3.16, released Aug 2025), build
    from source against the latest llama.cpp::

        git clone https://github.com/abetlen/llama-cpp-python.git
        cd llama-cpp-python
        git submodule update --init --recursive
        cd vendor/llama.cpp && git pull origin master && cd ../..
        CMAKE_ARGS="-DGGML_CUDA=on" pip install -e . --no-cache-dir
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed.  Install it with:\n"
            '  CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python>=0.2.90"'
        ) from exc

    # Pre-flight: check if the GGUF architecture is supported by this build.
    arch = _read_gguf_architecture(model_path)
    if arch in _ARCH_NEEDS_NEWER_BUILD:
        import llama_cpp
        ver = getattr(llama_cpp, "__version__", "unknown")
        raise RuntimeError(
            f"GGUF architecture '{arch}' is not supported by the llama.cpp "
            f"bundled in llama-cpp-python {ver} (released Aug 2025).\n\n"
            f"llama.cpp master supports '{arch}', but there is no newer "
            f"llama-cpp-python release yet.  Build from source:\n\n"
            f"  git clone https://github.com/abetlen/llama-cpp-python.git\n"
            f"  cd llama-cpp-python\n"
            f"  git submodule update --init --recursive\n"
            f"  cd vendor/llama.cpp && git pull origin master && cd ../..\n"
            f"  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install -e . --no-cache-dir\n\n"
            f"Or on a network-restricted server, build llama.cpp standalone\n"
            f"and point to its shared library via LLAMA_CPP_LIB env var."
        )

    logger.info(f"Loading GGUF model: {model_path}")
    logger.info(f"  n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, n_batch={n_batch}")

    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=verbose,
        )
    except Exception as exc:
        msg = str(exc)
        if "unknown model architecture" in msg:
            arch_hint = msg.split("'")[1] if "'" in msg else "unknown"
            raise RuntimeError(
                f"llama.cpp does not recognise the model architecture '{arch_hint}'.\n"
                f"The bundled llama.cpp is too old for this GGUF file.\n"
                f"Build llama-cpp-python from source with the latest llama.cpp "
                f"(see docstring above for steps)."
            ) from exc
        raise

    model = LlamaCppModel(llm)
    tokenizer = LlamaCppTokenizer(llm)

    logger.info("GGUF model loaded successfully")
    return model, tokenizer
