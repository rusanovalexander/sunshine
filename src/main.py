"""
=================================================
Main Pipeline Orchestrator (v2)
=================================================
Coordinates all extraction stages with enhanced
error handling and progress tracking.
"""

import os
import sys
import gc
import json
import logging
import argparse
from datetime import datetime
from typing import Optional

import torch

# =====================================================================
# SETUP
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Pipeline - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_v2.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║     PROJECT SUNSHINE - Document Extraction Pipeline v2        ║
    ║     Multi-Pass Evidence-Based Extraction System               ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Stage 1: Document Preprocessing                              ║
    ║  Stage 2: Multi-Pass Field Group Extraction                   ║
    ║  Stage 3: Deep Field Extraction (Fallback)                    ║
    ║  Stage 4: Verification & Consolidation                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """Verify environment is correctly set up."""
    logger.info("Checking environment...")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Check current memory usage
        try:
            from .gpu_optimizer import get_gpu_memory_stats
            stats = get_gpu_memory_stats()
            if stats:
                logger.info(f"  GPU Memory: {stats.free_gb:.1f}GB free / {stats.total_gb:.1f}GB total")
                if stats.free_gb < 15.0:
                    logger.warning(f"  WARNING: Only {stats.free_gb:.1f}GB free, may have issues")
        except ImportError:
            pass
    else:
        logger.warning("  No CUDA GPU detected!")
    
    # Check required packages
    required = ['transformers', 'pandas', 'pymupdf', 'openpyxl']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.error(f"  Missing packages: {missing}")
        return False
    
    logger.info("  All required packages available")
    return True


def check_paths():
    """Verify required paths exist."""
    from .config import (
        MODEL_PATH, VLM_MODEL_PATH, ARCHIVE_PATH, EXTRACTED_SOURCE_DIR
    )
    
    logger.info("Checking paths...")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"  LLM model not found: {MODEL_PATH}")
        return False
    logger.info(f"  LLM model: OK")
    
    if not os.path.exists(VLM_MODEL_PATH):
        logger.warning(f"  VLM model not found: {VLM_MODEL_PATH} (OCR will be limited)")
    else:
        logger.info(f"  VLM model: OK")
    
    if not os.path.exists(ARCHIVE_PATH) and not os.path.exists(EXTRACTED_SOURCE_DIR):
        logger.error(f"  No source data found")
        return False
    logger.info(f"  Source data: OK")
    
    return True


# =====================================================================
# PIPELINE STAGES
# =====================================================================

def run_preprocessing(args):
    """Stage 1: Document preprocessing."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 1: DOCUMENT PREPROCESSING")
    logger.info("="*60)
    
    from .preprocess_v2 import main as preprocess_main
    
    # Build sys.argv for preprocessing
    preprocess_args = ['preprocess_v2.py']
    preprocess_args.extend(['--ocr_method', args.ocr_method])
    if args.flash_attention:
        preprocess_args.append('--flash_attention')
    if args.company:
        preprocess_args.extend(['--company', args.company])
    
    original_argv = sys.argv
    sys.argv = preprocess_args
    
    try:
        preprocess_main()
    finally:
        sys.argv = original_argv
    
    # CRITICAL: Unload VLM to free GPU memory for LLM
    logger.info("Unloading VLM to free GPU memory...")
    try:
        from .gpu_optimizer import aggressive_memory_cleanup, log_gpu_memory
        aggressive_memory_cleanup()
        log_gpu_memory("After VLM unload: ")
    except ImportError:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_extraction(args):
    """Stage 2-4: Multi-pass extraction with deep fallback."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 2-4: MULTI-PASS EXTRACTION")
    logger.info("="*60)

    from .config import (
        MODEL_PATH, PREPROCESSED_DATA_DIR, CHUNKS_DIR, EXTRACTION_DIR, OUTPUT_CSV,
        EMBEDDING_MODEL_PATH, RETRIEVER_TYPE
    )
    from .extract_v2 import (
        initialize_model, process_document, extraction_to_rows,
        save_detailed_extraction, llm_generate
    )
    from .deep_extract import deep_extract_missing_fields, cross_reference_fields
    from .retriever import create_retriever_from_chunks, load_embedding_model
    import pandas as pd

    # Determine retriever type
    retriever_type = getattr(args, 'retriever', None) or RETRIEVER_TYPE
    logger.info(f"Retriever type: {retriever_type}")

    # Initialize LLM model
    logger.info("Loading LLM model...")
    model, tokenizer = initialize_model(MODEL_PATH, args.flash_attention)

    if model is None:
        logger.error("Failed to load model")
        return False

    # Load embedding model if needed
    embedding_model = None
    embedding_tokenizer = None
    if retriever_type in ("embedding", "hybrid"):
        emb_path = getattr(args, 'embedding_model_path', None) or EMBEDDING_MODEL_PATH
        logger.info(f"Loading embedding model from {emb_path}...")
        embedding_model, embedding_tokenizer = load_embedding_model(emb_path)

    # Load manifest
    manifest_path = os.path.join(PREPROCESSED_DATA_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        return False

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    if args.company:
        manifest = [m for m in manifest if m['company'] == args.company]

    logger.info(f"Processing {len(manifest)} documents")

    os.makedirs(EXTRACTION_DIR, exist_ok=True)
    all_rows = []

    for i, entry in enumerate(manifest):
        logger.info(f"\n{'─'*60}")
        logger.info(f"Document {i+1}/{len(manifest)}: {entry['company']} - {entry['original_file']}")
        logger.info(f"{'─'*60}")

        try:
            # Stage 2-3: Primary extraction
            extraction = process_document(
                entry['company'], entry, model, tokenizer,
                retriever_type=retriever_type,
                embedding_model=embedding_model,
                embedding_tokenizer=embedding_tokenizer
            )

            if extraction and extraction.facilities:
                # Stage 3.5: Deep extraction for missing fields
                logger.info("  Running deep extraction for missing fields...")

                chunks_file = os.path.join(CHUNKS_DIR, entry['chunks_file'])
                with open(chunks_file, 'r') as f:
                    chunks = json.load(f)

                retriever = create_retriever_from_chunks(
                    chunks, retriever_type=retriever_type,
                    embedding_model=embedding_model,
                    embedding_tokenizer=embedding_tokenizer
                )
                
                # Load full text for table extraction in deep extract
                text_file = os.path.join(PREPROCESSED_DATA_DIR, entry['text_file'])
                full_text = None
                if os.path.exists(text_file):
                    with open(text_file, 'r', encoding='utf-8') as f:
                        full_text = f.read()

                for facility in extraction.facilities:
                    # Convert to dict for deep extraction
                    raw_extractions = facility.raw_extractions

                    # Deep extract (with full_text for table extraction)
                    enhanced_extractions = deep_extract_missing_fields(
                        raw_extractions, retriever, model, tokenizer, llm_generate,
                        full_text=full_text
                    )
                    
                    # Cross-reference check
                    consistency = cross_reference_fields(enhanced_extractions)
                    if consistency.get('consistency_issues'):
                        logger.warning(f"    Consistency issues: {consistency['consistency_issues']}")
                    
                    # Update facility with enhanced data
                    facility.raw_extractions = enhanced_extractions
                    
                    # Update fields
                    from .extract_v2 import ExtractedValue
                    for group_name, group_data in enhanced_extractions.items():
                        if 'fields' not in group_data:
                            continue
                        for field_name, field_data in group_data['fields'].items():
                            if isinstance(field_data, dict):
                                facility.fields[field_name] = ExtractedValue(
                                    value=field_data.get('value', 'NOT_FOUND'),
                                    evidence=field_data.get('evidence', ''),
                                    confidence=field_data.get('confidence', 'LOW')
                                )
                
                # Save detailed extraction
                save_detailed_extraction(extraction, EXTRACTION_DIR)
                
                # Convert to rows
                rows = extraction_to_rows(extraction)
                all_rows.extend(rows)
                
                logger.info(f"  ✓ Extracted {len(extraction.facilities)} facilities")
            else:
                logger.warning(f"  ✗ No data extracted")
        
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}", exc_info=True)
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save final CSV
    if all_rows:
        from .config import ALL_FIELDS
        df = pd.DataFrame(all_rows)
        ordered_cols = [c for c in ALL_FIELDS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + extra_cols]
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SUCCESS: Saved {len(df)} rows to {OUTPUT_CSV}")
        logger.info(f"Detailed extractions: {EXTRACTION_DIR}/")
        logger.info(f"{'='*60}")
        return True
    else:
        logger.warning("No data extracted")
        return False


def generate_quality_report():
    """Generate extraction quality report."""
    logger.info("\n" + "="*60)
    logger.info("GENERATING QUALITY REPORT")
    logger.info("="*60)

    from .config import OUTPUT_CSV, EXTRACTION_DIR, EXTRACTABLE_FIELDS
    import pandas as pd

    if not os.path.exists(OUTPUT_CSV):
        logger.warning("No output CSV to analyze")
        return

    df = pd.read_csv(OUTPUT_CSV)

    # Calculate field coverage
    coverage = {}
    for field in EXTRACTABLE_FIELDS:
        if field in df.columns:
            non_empty = df[field].notna() & ~df[field].isin(['NOT_FOUND', 'N/A', '', 'NOT_EXTRACTED'])
            coverage[field] = non_empty.sum() / len(df) * 100

    # Sort by coverage
    sorted_coverage = sorted(coverage.items(), key=lambda x: x[1], reverse=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_rows": len(df),
        "total_fields": len(EXTRACTABLE_FIELDS),
        "average_coverage": sum(coverage.values()) / len(coverage) if coverage else 0,
        "field_coverage": {k: f"{v:.1f}%" for k, v in sorted_coverage},
        "low_coverage_fields": [k for k, v in sorted_coverage if v < 30],
        "high_coverage_fields": [k for k, v in sorted_coverage if v > 70],
    }

    report_path = os.path.join(EXTRACTION_DIR, "quality_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"  Total documents: {len(df)}")
    logger.info(f"  Average field coverage: {report['average_coverage']:.1f}%")
    logger.info(f"  High coverage fields (>70%): {len(report['high_coverage_fields'])}")
    logger.info(f"  Low coverage fields (<30%): {len(report['low_coverage_fields'])}")
    logger.info(f"  Report saved to: {report_path}")


def run_quality_comparison(args):
    """Compare extraction results against a golden record CSV (e.g. Gemini 2.5 Pro)."""
    logger.info("\n" + "="*60)
    logger.info("QUALITY COMPARISON: Golden Record vs Extraction")
    logger.info("="*60)

    from .config import OUTPUT_CSV, EXTRACTION_DIR
    from .quality_compare import run_comparison

    golden_csv = args.golden_csv
    extracted_csv = args.extracted_csv or OUTPUT_CSV

    if not os.path.exists(golden_csv):
        logger.error(f"Golden record CSV not found: {golden_csv}")
        return False

    if not os.path.exists(extracted_csv):
        logger.error(f"Extracted CSV not found: {extracted_csv}")
        return False

    output_path = os.path.join(EXTRACTION_DIR, "quality_comparison.json")

    report = run_comparison(
        golden_csv=golden_csv,
        extracted_csv=extracted_csv,
        company_filter=args.company,
        output_path=output_path
    )

    if report:
        logger.info(f"  Overall accuracy: {report.get('overall_accuracy', 'N/A')}")
        logger.info(f"  Comparison report saved to: {output_path}")
        return True

    return False


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Project Sunshine Document Extraction Pipeline v2"
    )
    parser.add_argument(
        "--stage",
        choices=['all', 'preprocess', 'extract', 'report', 'compare'],
        default='all',
        help="Which stage to run (use 'compare' for golden record comparison)"
    )
    parser.add_argument(
        "--ocr_method",
        choices=['vlm', 'easyocr', 'none'],
        default='vlm',
        help="OCR method for preprocessing"
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        help="Use Flash Attention 2 for faster inference"
    )
    parser.add_argument(
        "--company",
        help="Process specific company only"
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip preprocessing if already done"
    )
    parser.add_argument(
        "--retriever",
        choices=['bm25', 'embedding', 'hybrid'],
        default=None,
        help="Retriever type: bm25 (keyword), embedding (Qwen3-Embedding-0.6B), hybrid (both combined)"
    )
    parser.add_argument(
        "--embedding_model_path",
        default=None,
        help="Path to embedding model (defaults to config EMBEDDING_MODEL_PATH)"
    )
    parser.add_argument(
        "--golden_csv",
        help="Path to golden record CSV (e.g. Gemini 2.5 Pro results) for quality comparison"
    )
    parser.add_argument(
        "--extracted_csv",
        help="Path to extracted CSV to compare against golden record (defaults to pipeline output)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Environment checks
    if not check_environment():
        logger.error("Environment check failed")
        return 1
    
    if not check_paths():
        logger.error("Path check failed")
        return 1
    
    start_time = datetime.now()
    
    try:
        if args.stage in ['all', 'preprocess']:
            if not args.skip_preprocess:
                run_preprocessing(args)
            else:
                logger.info("Skipping preprocessing (--skip_preprocess)")
        
        if args.stage in ['all', 'extract']:
            success = run_extraction(args)
            if not success:
                logger.error("Extraction failed")
                return 1
        
        if args.stage in ['all', 'report']:
            generate_quality_report()

        if args.stage == 'compare':
            if not args.golden_csv:
                logger.error("--golden_csv is required for 'compare' stage")
                return 1
            run_quality_comparison(args)
        
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        return 130
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        return 1
    
    elapsed = datetime.now() - start_time
    logger.info(f"\nPipeline completed in {elapsed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
