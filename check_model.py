#!/usr/bin/env python3
"""
check_model.py — Standalone diagnostic for GGUF model loading.

Run this independently of the main pipeline to verify that
llama-cpp-python can load the model before running the full pipeline.

Usage:
    python check_model.py
    python check_model.py --model /path/to/model.gguf
    python check_model.py --n-gpu-layers 0   # test CPU-only
"""

import argparse
import os
import sys
import time

# ── Defaults (mirrors config.py) ──────────────────────────────────────────────
DEFAULT_MODEL_PATH  = "/home/inghero/data/irwbds/llm/parc/Ministral-14B-GGUF/Ministral-3-14B-Instruct-2512-Q5_K_M.gguf"
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_CTX        = 512    # small context just for the load test
DEFAULT_N_BATCH      = 512

TEST_PROMPT = [{"role": "user", "content": "Reply with one word: ready"}]


def check_file(path: str) -> bool:
    print(f"\n[1] File check")
    print(f"    Path : {path}")
    if not os.path.exists(path):
        print(f"    ERROR: file not found")
        return False
    size_gb = os.path.getsize(path) / 1024**3
    print(f"    Size : {size_gb:.2f} GB")
    if size_gb < 0.1:
        print(f"    WARNING: file looks too small — may be corrupted or incomplete")
    readable = os.access(path, os.R_OK)
    print(f"    Read permission: {'yes' if readable else 'NO — permission denied'}")
    return readable


def check_import() -> bool:
    print(f"\n[2] Import check")
    try:
        from llama_cpp import Llama  # noqa: F401
        import llama_cpp
        version = getattr(llama_cpp, "__version__", "unknown")
        print(f"    llama-cpp-python version: {version}")
        return True
    except ImportError as e:
        print(f"    ERROR: {e}")
        print(f"    Fix: CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install \"llama-cpp-python>=0.2.90\"")
        return False


def check_cuda() -> None:
    print(f"\n[3] CUDA / GPU check")
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                print(f"    GPU: {line.strip()}")
        else:
            print(f"    nvidia-smi failed: {result.stderr.strip()}")
    except FileNotFoundError:
        print(f"    nvidia-smi not found — cannot check GPU")
    except Exception as e:
        print(f"    GPU check error: {e}")


def check_load(model_path: str, n_gpu_layers: int, n_ctx: int, n_batch: int) -> bool:
    print(f"\n[4] Model load check")
    print(f"    n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, n_batch={n_batch}")
    try:
        from llama_cpp import Llama
        t0 = time.time()
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"    Load time: {elapsed:.1f}s")
        return llm
    except Exception as e:
        print(f"    ERROR loading model: {e}")
        return None


def check_inference(llm) -> bool:
    print(f"\n[5] Inference check")
    try:
        t0 = time.time()
        output = llm.create_chat_completion(
            messages=TEST_PROMPT,
            max_tokens=16,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        response = output["choices"][0]["message"]["content"].strip()
        tokens = output.get("usage", {})
        print(f"    Response   : {repr(response)}")
        print(f"    Tokens     : {tokens}")
        print(f"    Latency    : {elapsed:.2f}s")
        return True
    except Exception as e:
        print(f"    ERROR during inference: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="GGUF model load diagnostic")
    parser.add_argument("--model",        default=DEFAULT_MODEL_PATH,  help="Path to .gguf file")
    parser.add_argument("--n-gpu-layers", default=DEFAULT_N_GPU_LAYERS, type=int)
    parser.add_argument("--n-ctx",        default=DEFAULT_N_CTX,        type=int)
    parser.add_argument("--n-batch",      default=DEFAULT_N_BATCH,      type=int)
    parser.add_argument("--skip-inference", action="store_true", help="Only test load, skip inference")
    args = parser.parse_args()

    print("=" * 60)
    print("  GGUF Model Load Diagnostic")
    print("=" * 60)

    ok = True

    ok = check_file(args.model) and ok
    if not ok:
        print("\nAborting: fix file path/permissions first.")
        sys.exit(1)

    ok = check_import() and ok
    if not ok:
        print("\nAborting: install llama-cpp-python first.")
        sys.exit(1)

    check_cuda()

    llm = check_load(args.model, args.n_gpu_layers, args.n_ctx, args.n_batch)
    if llm is None:
        print("\nFAILED: model did not load.")
        print("\nTroubleshooting tips:")
        print("  - Try --n-gpu-layers 0 to rule out a CUDA/VRAM issue")
        print("  - Try --n-ctx 512 if context size causes OOM")
        print("  - Re-download the GGUF if the file size looks wrong")
        sys.exit(1)

    if not args.skip_inference:
        ok = check_inference(llm) and ok

    print("\n" + "=" * 60)
    if ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — see details above")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
