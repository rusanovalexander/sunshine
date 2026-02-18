# Project Sunshine

> **Multi-Pass Evidence-Based Document Extraction Pipeline for Project Finance**

Extract structured financial data from complex legal and financial documents using local LLMs with enterprise-grade accuracy.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![GPU: CUDA](https://img.shields.io/badge/GPU-CUDA-orange.svg)

## Overview

Project Sunshine is an intelligent document processing pipeline designed to extract 70+ structured fields from project finance documents including:

- Facility Agreements
- Term Sheets
- Credit Agreements
- Loan Schedules
- Amendment Documents

### Key Features

- **Multi-Pass Extraction**: Fields extracted in logical groups (5-8 at a time) for higher accuracy
- **Evidence-Based**: Every extracted value includes supporting quotes from source documents
- **Company-Level Consolidation**: Combines all files per company before extraction
- **GPU Optimized**: Designed for A100/consumer GPUs with 20GB+ VRAM
- **Full Coverage Guarantee**: Verifies 100% of document content is analyzed
- **Multi-Facility Detection**: Automatically identifies and extracts multiple tranches

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: PREPROCESSING                        │
│  Archive → Extract Text/OCR → Structure Analysis → Chunking     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: CONSOLIDATION                        │
│  Combine all company files → Create unified chunks → Index      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 3: EXTRACTION                           │
│  Detect Facilities → Multi-Pass Field Extraction → Verification │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 4: OUTPUT                               │
│  CSV Export → Detailed JSON with Evidence → Quality Report      │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (20GB+ VRAM recommended)
- 32GB+ System RAM

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/project-sunshine.git
cd project-sunshine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Model Setup

Download and configure your LLM models:

```bash
# Edit config with your model paths
cp src/config.py src/config_local.py

# Update paths in config_local.py:
# MODEL_PATH = "/path/to/your/Qwen3-14B"
# VLM_MODEL_PATH = "/path/to/your/Qwen2.5-VL-3B-Instruct"
```

**Supported Models:**
- LLM: Qwen3-14B, Qwen2.5-14B, Llama-3-8B (4-bit quantized)
- VLM (OCR): Qwen2.5-VL-3B-Instruct

## Usage

### Quick Start

```bash
# Full pipeline
./run.sh --company "CompanyName"

# Or with Python directly
python -m src.main --stage all --flash_attention --company "CompanyName"
```

### Command Line Options

```bash
python -m src.main [OPTIONS]

Options:
  --stage {all,preprocess,extract,report}
                        Pipeline stage to run (default: all)
  --ocr_method {vlm,easyocr,none}
                        OCR method for scanned documents (default: vlm)
  --company NAME        Process specific company only
  --flash_attention     Enable Flash Attention 2 for faster inference
  --skip_preprocess     Skip preprocessing if already completed
```

### Examples

```bash
# Process all companies
python -m src.main --stage all

# Preprocess only (useful for large archives)
python -m src.main --stage preprocess --ocr_method vlm

# Extract from already preprocessed data
python -m src.main --stage extract --skip_preprocess --flash_attention

# Generate quality report
python -m src.main --stage report
```

## Configuration

Edit `src/config.py` to customize:

### Paths
```python
MODEL_PATH = "/path/to/Qwen3-14B"
VLM_MODEL_PATH = "/path/to/Qwen2.5-VL-3B"
ARCHIVE_PATH = "/path/to/source/documents.zip"
```

### Processing Settings
```python
CHUNK_SIZE = 3000        # Tokens per chunk
CHUNK_OVERLAP = 500      # Overlap between chunks
MAX_CHUNKS_PER_FIELD_GROUP = 8  # Chunks sent to LLM per extraction
```

### Field Schema

The extraction schema is defined in `FIELD_GROUPS` within `config.py`. Each group contains:
- Field names and descriptions
- Search keywords for retrieval
- Extraction instructions

## Output

### CSV Export
```
results_project_sunshine_v2.csv
```
Flat table with one row per facility, all 70+ fields as columns.

### Detailed JSON
```
extraction_outputs_v2/
└── CompanyName/
    ├── _COMBINED_DOCUMENT.txt      # All files merged
    ├── _COMBINED_CHUNKS.json       # Indexed chunks
    ├── _COMBINED_META.json         # Source file mapping
    └── CompanyName_extraction.json # Extraction with evidence
```

### Evidence Format
```json
{
  "Spread [%]": {
    "value": "2.75%",
    "evidence": "Margin means 2.75 per cent per annum",
    "confidence": "HIGH"
  }
}
```

## Field Groups

| Group | Fields | Description |
|-------|--------|-------------|
| `basic_info` | 6 | Borrower, facility name, type, currency, limit |
| `sponsor_info` | 5 | Sponsor name, ownership, HoldCo |
| `project_details` | 6 | Sector, location, readiness |
| `construction_guarantees` | 5 | Completion guarantees, contractor |
| `revenue_mitigants` | 6 | PPAs, offtake contracts |
| `covenants` | 12 | DSCR, LLCR, ICR, leverage ratios |
| `syndication_ing` | 3 | Syndication, lender shares |
| `dates_schedules` | 7 | Inception, maturity, schedules |
| `pricing` | 6 | Base rate, spread, floors/caps |
| `hedging` | 8 | Swap details, notional |
| `cash_sweep` | 5 | Cash sweep mechanisms |
| `fees` | 3 | Upfront, commitment, LC fees |

## GPU Memory Management

Optimized for 20GB VRAM (A100 MIG, RTX 3090/4090):

```
Stage 1 (VLM OCR):     ~4GB
  ↓ [Unload VLM]
Stage 2 (LLM Extract): ~12GB
  + KV Cache:          ~4GB
  ─────────────────────────
  Total Peak:          ~16GB
```

Features:
- Automatic model unloading between stages
- Dynamic batch sizing based on available memory
- OOM recovery with reduced token limits
- Memory monitoring with peak usage tracking

## Supported File Formats

| Format | Extension | OCR Support |
|--------|-----------|-------------|
| PDF | `.pdf` | Yes (hybrid) |
| Word | `.docx` | No |
| Excel | `.xlsx`, `.xls` | Yes (embedded images) |
| Outlook | `.msg` | No |
| Images | `.png`, `.jpg` | Yes |
| Text | `.txt` | N/A |

## Performance

| Metric | Expected |
|--------|----------|
| Documents/hour | 5-15 |
| Field coverage | 60-85% |
| Accuracy (when found) | 85-95% |
| GPU memory | 12-16 GB peak |

## Troubleshooting

### Out of Memory
```bash
# Reduce chunk processing
# Edit config.py: MAX_CHUNKS_PER_FIELD_GROUP = 4

# Or disable Flash Attention
python -m src.main --stage extract  # without --flash_attention
```

### EasyOCR Offline Error
```bash
# Pre-download models on internet-connected machine:
python -c "import easyocr; easyocr.Reader(['en', 'de', 'fr'])"

# Copy ~/.EasyOCR to offline server
# Or use VLM: --ocr_method vlm
```

### Low Extraction Quality
1. Check `_COMBINED_DOCUMENT.txt` - is content properly extracted?
2. Review evidence in JSON outputs
3. Add domain keywords to `FIELD_GROUPS` in config
4. Consider few-shot prompting (see `examples/`)

## Project Structure

```
project-sunshine/
├── src/
│   ├── config.py          # Configuration and schema
│   ├── preprocess_v2.py   # Document preprocessing
│   ├── consolidate.py     # Company file consolidation
│   ├── retriever.py       # BM25 chunk retrieval
│   ├── extract_v2.py      # Multi-pass extraction
│   ├── deep_extract.py    # Fallback field extraction
│   ├── gpu_optimizer.py   # GPU memory management
│   └── main.py            # Pipeline orchestrator
├── examples/
│   └── few_shot_template.py
├── docs/
│   └── SCHEMA.md
├── requirements.txt
├── run.sh
├── LICENSE
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen) - LLM backbone
- [Transformers](https://github.com/huggingface/transformers) - Model loading
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF processing
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR engine

---

**Note**: This tool is designed for internal document processing. Ensure compliance with document confidentiality requirements before use.
