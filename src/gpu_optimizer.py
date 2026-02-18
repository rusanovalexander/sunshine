"""
=================================================
GPU Optimization Module for A100 20GB
=================================================
Handles:
1. Memory monitoring and profiling
2. Dynamic batch sizing
3. Model lifecycle management (load/unload)
4. Batch inference for chunks
5. KV cache optimization
6. CUDA stream management
"""

import os
import gc
import logging
import threading
import time
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

import torch
from torch.cuda import Stream

logger = logging.getLogger(__name__)

# =====================================================================
# MEMORY MONITORING
# =====================================================================

@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    utilization_pct: float


def get_gpu_memory_stats() -> Optional[GPUMemoryStats]:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return None
    
    try:
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        free = total - reserved
        
        return GPUMemoryStats(
            total_gb=total / 1e9,
            allocated_gb=allocated / 1e9,
            reserved_gb=reserved / 1e9,
            free_gb=free / 1e9,
            utilization_pct=(reserved / total) * 100
        )
    except Exception as e:
        logger.warning(f"Failed to get GPU stats: {e}")
        return None


def log_gpu_memory(prefix: str = ""):
    """Log current GPU memory usage."""
    stats = get_gpu_memory_stats()
    if stats:
        logger.info(f"{prefix}GPU Memory: {stats.allocated_gb:.2f}GB allocated, "
                   f"{stats.reserved_gb:.2f}GB reserved, "
                   f"{stats.free_gb:.2f}GB free ({stats.utilization_pct:.1f}% used)")


class GPUMemoryMonitor:
    """Background GPU memory monitor thread."""
    
    def __init__(self, interval_sec: float = 5.0, log_threshold_pct: float = 80.0):
        self.interval = interval_sec
        self.threshold = log_threshold_pct
        self._stop_event = threading.Event()
        self._thread = None
        self.peak_memory_gb = 0.0
    
    def _monitor_loop(self):
        while not self._stop_event.is_set():
            stats = get_gpu_memory_stats()
            if stats:
                self.peak_memory_gb = max(self.peak_memory_gb, stats.reserved_gb)
                if stats.utilization_pct > self.threshold:
                    logger.warning(f"HIGH GPU MEMORY: {stats.utilization_pct:.1f}% "
                                  f"({stats.reserved_gb:.2f}GB/{stats.total_gb:.2f}GB)")
            self._stop_event.wait(self.interval)
    
    def start(self):
        if self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("GPU memory monitor started")
    
    def stop(self):
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            logger.info(f"GPU memory monitor stopped. Peak usage: {self.peak_memory_gb:.2f}GB")


# =====================================================================
# MEMORY MANAGEMENT
# =====================================================================

def aggressive_memory_cleanup():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force Python garbage collection multiple times
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()


def unload_model(model) -> None:
    """Properly unload a model from GPU memory."""
    if model is None:
        return
    
    try:
        # Move to CPU first (sometimes helps with memory fragmentation)
        model.cpu()
    except:
        pass
    
    # Delete the model
    del model
    
    # Aggressive cleanup
    aggressive_memory_cleanup()
    
    log_gpu_memory("After unload: ")


@contextmanager
def gpu_memory_context(operation_name: str = "operation"):
    """Context manager for GPU memory tracking."""
    log_gpu_memory(f"Before {operation_name}: ")
    start_stats = get_gpu_memory_stats()
    
    try:
        yield
    finally:
        end_stats = get_gpu_memory_stats()
        if start_stats and end_stats:
            delta = end_stats.reserved_gb - start_stats.reserved_gb
            logger.info(f"After {operation_name}: Memory delta: {delta:+.2f}GB")
        log_gpu_memory(f"After {operation_name}: ")


# =====================================================================
# MODEL MANAGER - Single model at a time
# =====================================================================

class ModelManager:
    """
    Manages model lifecycle to ensure only one model is loaded at a time.
    Critical for fitting within 20GB A100.
    """
    
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        self.current_tokenizer = None
        self.model_loaders = {}
    
    def register_loader(self, name: str, loader_fn: Callable):
        """Register a model loader function."""
        self.model_loaders[name] = loader_fn
    
    def get_model(self, name: str) -> Tuple[any, any]:
        """Get a model, unloading current one if different."""
        if self.current_model_name == name and self.current_model is not None:
            return self.current_model, self.current_tokenizer
        
        # Unload current model
        if self.current_model is not None:
            logger.info(f"Unloading model: {self.current_model_name}")
            unload_model(self.current_model)
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
        
        # Load new model
        if name not in self.model_loaders:
            raise ValueError(f"Unknown model: {name}")
        
        logger.info(f"Loading model: {name}")
        with gpu_memory_context(f"loading {name}"):
            self.current_model, self.current_tokenizer = self.model_loaders[name]()
        
        self.current_model_name = name
        return self.current_model, self.current_tokenizer
    
    def unload_current(self):
        """Explicitly unload current model."""
        if self.current_model is not None:
            logger.info(f"Unloading model: {self.current_model_name}")
            unload_model(self.current_model)
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None


# Global model manager instance
_model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    return _model_manager


# =====================================================================
# BATCH INFERENCE
# =====================================================================

def calculate_optimal_batch_size(
    model,
    tokenizer,
    sample_text: str,
    max_new_tokens: int = 2048,
    target_memory_pct: float = 85.0
) -> int:
    """
    Dynamically calculate optimal batch size based on available memory.
    """
    stats = get_gpu_memory_stats()
    if not stats:
        return 1
    
    # Estimate memory per sample
    # Rough estimation: tokens * hidden_size * 2 (fp16) * 2 (activations)
    try:
        tokens = len(tokenizer.encode(sample_text))
        hidden_size = model.config.hidden_size
        
        # Memory estimate (very rough)
        mem_per_sample_gb = (tokens * hidden_size * 4 * 2) / 1e9  # Input
        mem_per_sample_gb += (max_new_tokens * hidden_size * 4 * 2) / 1e9  # Output/KV
        
        # Available memory
        available_gb = stats.total_gb * (target_memory_pct / 100) - stats.reserved_gb
        
        # Calculate batch size
        batch_size = max(1, int(available_gb / mem_per_sample_gb))
        
        # Cap at reasonable maximum
        batch_size = min(batch_size, 8)
        
        logger.info(f"Calculated optimal batch size: {batch_size} "
                   f"(~{mem_per_sample_gb:.2f}GB/sample, {available_gb:.2f}GB available)")
        
        return batch_size
    except Exception as e:
        logger.warning(f"Batch size calculation failed: {e}, using 1")
        return 1


class BatchInferenceEngine:
    """
    Handles batched inference with dynamic batch sizing.
    """
    
    def __init__(self, model, tokenizer, max_new_tokens: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = 1  # Start conservative
        self._calibrated = False
    
    def calibrate(self, sample_texts: List[str]):
        """Calibrate batch size based on sample inputs."""
        if not sample_texts:
            return
        
        # Use longest sample for calibration
        longest = max(sample_texts, key=len)
        self.batch_size = calculate_optimal_batch_size(
            self.model, self.tokenizer, longest, self.max_new_tokens
        )
        self._calibrated = True
    
    def _prepare_batch(self, prompts: List[str]) -> Dict:
        """Prepare a batch of prompts for inference."""
        texts = []
        for prompt in prompts:
            if isinstance(prompt, list):  # Chat format
                text = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            else:
                text = prompt
            texts.append(text)
        
        # Tokenize with padding
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        ).to(self.model.device)
        
        return inputs
    
    def generate_batch(self, prompts: List[str], **generate_kwargs) -> List[str]:
        """Generate responses for a batch of prompts."""
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            
            try:
                inputs = self._prepare_batch(batch)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **generate_kwargs
                    )
                
                # Decode outputs
                for j, output in enumerate(outputs):
                    input_len = inputs.input_ids[j].shape[0]
                    generated = self.tokenizer.decode(
                        output[input_len:], skip_special_tokens=True
                    ).strip()
                    results.append(generated)
                
                # Memory cleanup between batches
                del inputs, outputs
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM at batch size {self.batch_size}, reducing...")
                self.batch_size = max(1, self.batch_size // 2)
                torch.cuda.empty_cache()
                
                # Retry with smaller batch
                for single_prompt in batch:
                    result = self._generate_single(single_prompt, **generate_kwargs)
                    results.append(result)
        
        return results
    
    def _generate_single(self, prompt: str, **generate_kwargs) -> str:
        """Fallback single-item generation."""
        try:
            if isinstance(prompt, list):
                text = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            else:
                text = prompt
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs
                )
            
            return self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.error(f"Single generation failed: {e}")
            return ""


# =====================================================================
# KV CACHE OPTIMIZATION
# =====================================================================

def get_optimized_generation_config(model_config, available_memory_gb: float) -> Dict:
    """
    Get optimized generation configuration based on model and available memory.
    """
    config = {
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
    }
    
    # Estimate KV cache memory requirement
    # KV cache size = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch * dtype_size
    try:
        num_layers = getattr(model_config, 'num_hidden_layers', 40)
        num_kv_heads = getattr(model_config, 'num_key_value_heads', 
                              getattr(model_config, 'num_attention_heads', 40))
        head_dim = getattr(model_config, 'head_dim', 
                          model_config.hidden_size // getattr(model_config, 'num_attention_heads', 40))
        
        # For 20GB A100 with 14B model, we can handle reasonable context
        # But limit to prevent OOM
        if available_memory_gb < 5:
            config["max_new_tokens"] = 1024
        elif available_memory_gb < 8:
            config["max_new_tokens"] = 2048
        else:
            config["max_new_tokens"] = 4096
        
        logger.debug(f"KV cache config: layers={num_layers}, kv_heads={num_kv_heads}, "
                    f"head_dim={head_dim}, max_tokens={config['max_new_tokens']}")
        
    except Exception as e:
        logger.warning(f"Could not optimize KV cache config: {e}")
        config["max_new_tokens"] = 2048
    
    return config


# =====================================================================
# CUDA STREAM MANAGEMENT
# =====================================================================

class CUDAStreamManager:
    """
    Manages CUDA streams for overlapping data transfer and computation.
    """
    
    def __init__(self):
        self.compute_stream = None
        self.transfer_stream = None
        self._initialized = False
    
    def initialize(self):
        if not torch.cuda.is_available():
            return
        
        if not self._initialized:
            self.compute_stream = Stream()
            self.transfer_stream = Stream()
            self._initialized = True
            logger.info("CUDA streams initialized")
    
    @contextmanager
    def compute_context(self):
        """Context for compute operations."""
        if self.compute_stream:
            with torch.cuda.stream(self.compute_stream):
                yield
        else:
            yield
    
    @contextmanager
    def transfer_context(self):
        """Context for data transfer operations."""
        if self.transfer_stream:
            with torch.cuda.stream(self.transfer_stream):
                yield
        else:
            yield
    
    def synchronize(self):
        """Synchronize all streams."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# =====================================================================
# OPTIMIZED MODEL LOADING
# =====================================================================

def _detect_quantization(model_path: str) -> str:
    """
    Auto-detect if a model directory contains pre-quantized weights.

    Returns:
        "fp8"   - Native FP8 quantized (quantization_config in config.json)
        "gptq"  - GPTQ quantized
        "awq"   - AWQ quantized
        "bnb4"  - Saved BitsAndBytes 4-bit checkpoint (already quantized on disk)
        "none"  - Full precision / not pre-quantized
    """
    import json

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return "none"

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return "none"

    # Check for quantization_config in model config (native FP8 / transformers)
    quant_config = config.get("quantization_config", {})
    quant_method = quant_config.get("quant_method", "").lower()

    if quant_method == "fp8":
        return "fp8"
    elif quant_method == "gptq":
        return "gptq"
    elif quant_method == "awq":
        return "awq"
    elif quant_method in ("bitsandbytes", "bnb"):
        # Saved BitsAndBytes 4-bit checkpoint — weights already quantized on disk
        return "bnb4"

    # Also check for BitsAndBytes by looking at load_in_4bit flag
    if quant_config.get("load_in_4bit", False) or quant_config.get("_load_in_4bit", False):
        return "bnb4"

    # Also check for FP8 checkpoint marker files
    if os.path.exists(os.path.join(model_path, "fp8_config.json")):
        return "fp8"

    return "none"


def save_quantized_model(model_path: str, output_path: str, max_memory_gb: float = 18.0):
    """
    Pre-quantize a model to BitsAndBytes 4-bit and save to disk.

    Uses the same simple 4-bit config as the original working setup:
    default fp4 quant type, no double quantization.

    This is a ONE-TIME operation. After saving, subsequent loads from
    output_path skip on-the-fly quantization — saving ~9GB peak GPU
    memory and loading 10x faster.

    Run once:  python -m src.main --stage quantize
    Then use:  set MODEL_PATH to the output_path in config.py
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Pre-quantizing model from {model_path}")
    logger.info(f"  Output: {output_path}")

    aggressive_memory_cleanup()

    # Simple 4-bit config matching the original working setup:
    # default fp4 quant type, no double quant, fp16 compute dtype
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    max_memory = {0: f"{max_memory_gb}GB", "cpu": "30GB"}

    logger.info("  Loading model with BitsAndBytes 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    logger.info("  Saving pre-quantized model...")
    model.save_pretrained(output_path)

    # Also copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    logger.info(f"  ✓ Pre-quantized model saved to {output_path}")
    logger.info(f"  Update MODEL_PATH in config.py to point to: {output_path}")
    logger.info(f"  Future loads will skip on-the-fly quantization (10x faster, ~9GB less peak memory)")

    # Cleanup
    del model
    aggressive_memory_cleanup()


def load_llm_optimized(
    model_path: str,
    use_flash_attention: bool = True,
    max_memory_gb: float = 18.0  # Leave 2GB headroom
) -> Tuple[any, any]:
    """
    Load LLM with optimized settings for A100 20GB.

    Auto-detects pre-quantized weights (FP8, GPTQ, AWQ, saved BnB 4-bit)
    and loads them directly — skipping slow on-the-fly BitsAndBytes quantization.
    Falls back to on-the-fly 4-bit NF4 BitsAndBytes if no pre-quantized weights found.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    aggressive_memory_cleanup()

    # Attention implementation
    attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"

    # Max memory mapping for device
    max_memory = {0: f"{max_memory_gb}GB", "cpu": "30GB"}

    # Auto-detect pre-quantized weights
    quant_type = _detect_quantization(model_path)

    logger.info(f"Loading LLM from {model_path}")
    logger.info(f"  Detected quantization: {quant_type}")

    if quant_type == "fp8":
        # ─── FP8 pre-quantized: load directly, no BitsAndBytes needed ───
        logger.info(f"  Config: FP8 pre-quantized, {attn_impl}, max {max_memory_gb}GB")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype="auto",  # Let the model config decide dtype
            low_cpu_mem_usage=True,
        )

    elif quant_type in ("gptq", "awq"):
        # ─── GPTQ/AWQ pre-quantized: load directly ───
        logger.info(f"  Config: {quant_type.upper()} pre-quantized, {attn_impl}, max {max_memory_gb}GB")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    elif quant_type == "bnb4":
        # ─── Saved BitsAndBytes 4-bit: load directly (no re-quantization) ───
        # Uses "eager" attention + no max_memory cap — matching the original
        # proven config that fits on 20GB MIG without hidden memory issues.
        logger.info(f"  Config: Saved BnB 4-bit (pre-quantized), eager attn, no max_memory cap")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )

    else:
        # ─── No pre-quantized weights: use BitsAndBytes 4-bit on-the-fly ───
        # NOTE: Using simple 4-bit config (no nf4, no double_quant) to match
        # the original working setup that fit on 20GB MIG.
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        logger.info(f"  Config: 4-bit (BitsAndBytes on-the-fly), eager attn, no max_memory cap")
        logger.info(f"  TIP: Run '--stage quantize' once to pre-quantize for faster loading & less memory")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )

    # Optimize for inference
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Verify chat template is available (critical for Qwen3)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        logger.warning("  Chat template not found in tokenizer — trying original model path")
        # Fallback: load tokenizer from the original (non-quantized) model
        import json as _json
        _cfg_path = os.path.join(model_path, "config.json")
        if os.path.exists(_cfg_path):
            try:
                with open(_cfg_path, 'r') as _f:
                    _cfg = _json.load(_f)
                _orig = _cfg.get("_name_or_path", "")
                if _orig and os.path.exists(_orig):
                    logger.info(f"  Loading tokenizer from original: {_orig}")
                    tokenizer = AutoTokenizer.from_pretrained(_orig, trust_remote_code=True)
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
            except Exception:
                pass

    # Final check
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        logger.info("  ✓ Chat template available")
    else:
        logger.warning("  ✗ No chat template found — apply_chat_template may fail!")

    log_gpu_memory("After LLM load: ")

    return model, tokenizer


def load_vlm_optimized(
    model_path: str,
    use_flash_attention: bool = False,
    max_memory_gb: float = 6.0  # VLM is smaller
) -> Tuple[any, any]:
    """
    Load VLM with optimized settings for A100 20GB.
    """
    from transformers import AutoProcessor, BitsAndBytesConfig
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        logger.error("Qwen2.5-VL not available in transformers")
        return None, None
    
    aggressive_memory_cleanup()
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"
    max_memory = {0: f"{max_memory_gb}GB", "cpu": "20GB"}
    
    logger.info(f"Loading VLM from {model_path}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    log_gpu_memory("After VLM load: ")
    
    return model, processor


# =====================================================================
# INTEGRATED OPTIMIZED PIPELINE
# =====================================================================

class OptimizedPipeline:
    """
    Main optimized pipeline class that manages GPU resources.
    """
    
    def __init__(self, llm_path: str, vlm_path: str = None):
        self.llm_path = llm_path
        self.vlm_path = vlm_path
        
        self.model_manager = get_model_manager()
        self.memory_monitor = GPUMemoryMonitor(interval_sec=10.0)
        self.stream_manager = CUDAStreamManager()
        self.batch_engine = None
        
        # Register model loaders
        self.model_manager.register_loader("llm", 
            lambda: load_llm_optimized(self.llm_path, use_flash_attention=True))
        
        if vlm_path:
            self.model_manager.register_loader("vlm",
                lambda: load_vlm_optimized(self.vlm_path, use_flash_attention=False))
    
    def start(self):
        """Initialize pipeline resources."""
        self.memory_monitor.start()
        self.stream_manager.initialize()
        log_gpu_memory("Pipeline start: ")
    
    def stop(self):
        """Cleanup pipeline resources."""
        self.model_manager.unload_current()
        self.memory_monitor.stop()
        aggressive_memory_cleanup()
        log_gpu_memory("Pipeline stop: ")
    
    def run_preprocessing(self, documents: List[str], ocr_fn: Callable) -> List[str]:
        """Run preprocessing with VLM for OCR."""
        if not self.vlm_path:
            logger.warning("No VLM configured, skipping OCR")
            return documents
        
        model, processor = self.model_manager.get_model("vlm")
        
        results = []
        for doc in documents:
            result = ocr_fn(doc, model, processor)
            results.append(result)
        
        # Unload VLM to free memory for LLM
        self.model_manager.unload_current()
        
        return results
    
    def run_extraction(self, chunks: List[Dict], extract_fn: Callable) -> List[Dict]:
        """Run extraction with LLM."""
        model, tokenizer = self.model_manager.get_model("llm")
        
        # Initialize batch engine
        if self.batch_engine is None:
            self.batch_engine = BatchInferenceEngine(model, tokenizer)
            sample_texts = [c.get('text', '')[:1000] for c in chunks[:5]]
            self.batch_engine.calibrate(sample_texts)
        
        # Get optimized generation config
        stats = get_gpu_memory_stats()
        available = stats.free_gb if stats else 8.0
        gen_config = get_optimized_generation_config(model.config, available)
        
        results = []
        for chunk in chunks:
            with self.stream_manager.compute_context():
                result = extract_fn(chunk, model, tokenizer, gen_config)
                results.append(result)
            
            # Periodic cleanup
            if len(results) % 10 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# =====================================================================
# UTILITY FUNCTIONS FOR INTEGRATION
# =====================================================================

def get_memory_efficient_generation_kwargs() -> Dict:
    """Get kwargs optimized for memory-efficient generation."""
    return {
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "use_cache": True,  # KV cache
        "num_beams": 1,  # No beam search (saves memory)
        "early_stopping": False,
    }


def estimate_chunk_memory_requirement(chunk_tokens: int, model_config) -> float:
    """Estimate GPU memory required for a chunk in GB."""
    try:
        hidden_size = model_config.hidden_size
        num_layers = model_config.num_hidden_layers
        
        # Rough estimate: activations + KV cache
        activation_mem = chunk_tokens * hidden_size * 4 * 2  # fp16
        kv_mem = 2 * num_layers * chunk_tokens * hidden_size * 2  # KV cache
        
        total_bytes = activation_mem + kv_mem
        return total_bytes / 1e9
    except:
        return 0.5  # Default estimate


def should_process_chunk(chunk_tokens: int, model_config, safety_margin_gb: float = 2.0) -> bool:
    """Check if we have enough memory to process a chunk."""
    stats = get_gpu_memory_stats()
    if not stats:
        return True  # Assume yes if we can't check
    
    required = estimate_chunk_memory_requirement(chunk_tokens, model_config)
    available = stats.free_gb - safety_margin_gb
    
    return required < available
