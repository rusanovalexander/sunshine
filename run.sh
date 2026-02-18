#!/bin/bash
# =================================================================
# Project Sunshine - Document Extraction Pipeline
# =================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Banner
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║           PROJECT SUNSHINE - Document Extraction              ║
║           Multi-Pass Evidence-Based Extraction                ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Default settings
STAGE="all"
OCR_METHOD="vlm"
COMPANY=""
FLASH_ATTENTION=""
SKIP_PREPROCESS=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift ;;
        --ocr_method) OCR_METHOD="$2"; shift ;;
        --company) COMPANY="$2"; shift ;;
        --flash_attention) FLASH_ATTENTION="--flash_attention" ;;
        --skip_preprocess) SKIP_PREPROCESS="--skip_preprocess" ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stage {all|preprocess|extract|report}  Pipeline stage (default: all)"
            echo "  --ocr_method {vlm|easyocr|none}          OCR method (default: vlm)"
            echo "  --company NAME                           Process specific company"
            echo "  --flash_attention                        Enable Flash Attention 2"
            echo "  --skip_preprocess                        Skip preprocessing stage"
            echo "  --help                                   Show this help"
            exit 0
            ;;
        *) echo -e "${RED}Unknown parameter: $1${NC}"; exit 1 ;;
    esac
    shift
done

# Build arguments
ARGS=(
    "--stage" "$STAGE"
    "--ocr_method" "$OCR_METHOD"
)

[[ -n "$COMPANY" ]] && ARGS+=("--company" "$COMPANY")
[[ -n "$FLASH_ATTENTION" ]] && ARGS+=("$FLASH_ATTENTION")
[[ -n "$SKIP_PREPROCESS" ]] && ARGS+=("$SKIP_PREPROCESS")

# Environment info
echo -e "${YELLOW}Environment Check${NC}"
echo "─────────────────────────────────────────"

# Python
PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "Python:  ${GREEN}$PYTHON_VERSION${NC}"

# PyTorch & CUDA
TORCH_INFO=$(python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "Not available")
echo -e "Torch:   ${GREEN}$TORCH_INFO${NC}"

# GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo -e "GPU:     ${GREEN}$GPU_INFO MB${NC}"
else
    echo -e "GPU:     ${RED}nvidia-smi not found${NC}"
fi

echo ""
echo -e "${YELLOW}Running Pipeline${NC}"
echo "─────────────────────────────────────────"
echo -e "Stage:       ${GREEN}$STAGE${NC}"
echo -e "OCR Method:  ${GREEN}$OCR_METHOD${NC}"
[[ -n "$COMPANY" ]] && echo -e "Company:     ${GREEN}$COMPANY${NC}"
[[ -n "$FLASH_ATTENTION" ]] && echo -e "Flash Attn:  ${GREEN}Enabled${NC}"
echo ""

# Run
python3 -m src.main "${ARGS[@]}"

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    PIPELINE COMPLETED                          ║"
    echo -e "╚═══════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}Pipeline failed with exit code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
