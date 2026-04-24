#!/usr/bin/env bash
# install_ai.sh — Install all AI dependencies for hledger-preprocessor.
#
# Detects available hardware (GPU/VRAM/RAM) and installs the right
# components:
#
#   1. Ollama (local LLM runtime)
#   2. Ollama models (VLM + text, sized to fit your VRAM)
#   3. Python training dependencies (SetFit, LoRA via peft/trl)
#
# Usage:
#   chmod +x install_ai.sh
#   ./install_ai.sh                    # auto-detect everything
#   ./install_ai.sh --cpu-only         # skip GPU detection, CPU models only
#   ./install_ai.sh --skip-ollama      # skip Ollama install (already have it)
#   ./install_ai.sh --skip-training    # skip training deps (SetFit, peft, trl)
#   ./install_ai.sh --dry-run          # show what would be installed

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Parse args ───────────────────────────────────────────────────────
CPU_ONLY=false
SKIP_OLLAMA=false
SKIP_TRAINING=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --cpu-only)       CPU_ONLY=true ;;
        --skip-ollama)    SKIP_OLLAMA=true ;;
        --skip-training)  SKIP_TRAINING=true ;;
        --dry-run)        DRY_RUN=true ;;
        --help|-h)
            echo "Usage: $0 [--cpu-only] [--skip-ollama] [--skip-training] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --cpu-only       Force CPU-only mode (skip GPU detection)"
            echo "  --skip-ollama    Skip Ollama installation"
            echo "  --skip-training  Skip Python training deps (SetFit, peft, trl)"
            echo "  --dry-run        Show what would be installed without doing it"
            exit 0
            ;;
        *)
            err "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

run() {
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        eval "$@"
    fi
}

# ── Detect hardware ─────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  hledger-ai: AI Dependencies Installer${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""

# RAM
RAM_GB=0
if [ -f /proc/meminfo ]; then
    RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    RAM_GB=$(( RAM_KB / 1024 / 1024 ))
fi
info "System RAM: ${RAM_GB} GB"

# GPU
HAS_GPU=false
GPU_NAME="none"
VRAM_GB=0

if $CPU_ONLY; then
    info "CPU-only mode (--cpu-only flag)"
elif command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    if [ -n "$GPU_NAME" ]; then
        HAS_GPU=true
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        VRAM_GB=$(( VRAM_MB / 1024 ))
        info "GPU: ${GPU_NAME} (${VRAM_GB} GB VRAM)"
    fi
else
    info "No NVIDIA GPU detected (nvidia-smi not found)"
fi

# ── Decide which models to pull ──────────────────────────────────────
# Ollama models are sized by VRAM (or RAM for CPU-only):
#
#   VRAM >= 8 GB  →  qwen3-vl:8b   + qwen3:1.7b   (best quality)
#   VRAM >= 4 GB  →  qwen3-vl:2b   + qwen3:0.6b   (good balance)
#   VRAM <  4 GB  →  qwen3-vl:2b   + qwen3:0.6b   (CPU offload)
#   CPU-only      →  qwen3-vl:2b   + qwen3:0.6b   (runs on RAM)

if $HAS_GPU && [ "$VRAM_GB" -ge 8 ]; then
    VLM_MODEL="qwen3-vl:8b"
    TEXT_MODEL="qwen3:1.7b"
    TIER="high"
    info "Tier: HIGH (${VRAM_GB}GB VRAM) — larger models for better accuracy"
elif $HAS_GPU && [ "$VRAM_GB" -ge 4 ]; then
    VLM_MODEL="qwen3-vl:2b"
    TEXT_MODEL="qwen3:0.6b"
    TIER="medium"
    info "Tier: MEDIUM (${VRAM_GB}GB VRAM) — standard models"
else
    VLM_MODEL="qwen3-vl:2b"
    TEXT_MODEL="qwen3:0.6b"
    TIER="cpu"
    if $HAS_GPU; then
        info "Tier: CPU-OFFLOAD (${VRAM_GB}GB VRAM is low; Ollama will split across GPU+RAM)"
    else
        info "Tier: CPU-ONLY — models run on RAM (slower but works)"
    fi
fi

echo ""
info "Selected models:"
info "  VLM (receipt OCR):        ${VLM_MODEL}"
info "  Text (classification):    ${TEXT_MODEL}"
echo ""

# ── Step 1: Install Ollama ───────────────────────────────────────────
echo -e "${BOLD}── Step 1: Ollama ─────────────────────────────────${NC}"

if $SKIP_OLLAMA; then
    info "Skipping Ollama install (--skip-ollama)"
elif command -v ollama &>/dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 || echo "unknown")
    ok "Ollama already installed: ${OLLAMA_VERSION}"
else
    info "Installing Ollama (requires sudo)..."
    if ! $DRY_RUN; then
        if ! sudo -n true 2>/dev/null; then
            warn "sudo requires a password. Please run this in a terminal:"
            echo ""
            echo "    curl -fsSL https://ollama.com/install.sh | sudo sh"
            echo ""
            echo "Then re-run this script with --skip-ollama"
            exit 1
        fi
    fi
    run "curl -fsSL https://ollama.com/install.sh | sudo sh"
    if ! $DRY_RUN; then
        if command -v ollama &>/dev/null; then
            ok "Ollama installed successfully"
        else
            err "Ollama installation failed. Install manually: https://ollama.com/download"
            exit 1
        fi
    fi
fi
echo ""

# ── Step 2: Start Ollama and pull models ─────────────────────────────
echo -e "${BOLD}── Step 2: Ollama models ──────────────────────────${NC}"

if $SKIP_OLLAMA; then
    info "Skipping model pull (--skip-ollama)"
else
    # Ensure ollama is running
    if ! $DRY_RUN; then
        if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
            info "Starting Ollama server..."
            ollama serve &>/dev/null &
            OLLAMA_PID=$!
            # Wait for it to come up
            for i in $(seq 1 30); do
                if curl -sf http://localhost:11434/api/tags &>/dev/null; then
                    break
                fi
                sleep 1
            done
            if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
                err "Ollama failed to start. Try running 'ollama serve' manually."
                exit 1
            fi
            ok "Ollama server started (pid ${OLLAMA_PID})"
        else
            ok "Ollama server already running"
        fi
    fi

    info "Pulling VLM model: ${VLM_MODEL} (this may take a few minutes)..."
    run "ollama pull ${VLM_MODEL}"
    echo ""

    info "Pulling text model: ${TEXT_MODEL}..."
    run "ollama pull ${TEXT_MODEL}"
    echo ""

    if ! $DRY_RUN; then
        ok "Models ready"
    fi
fi
echo ""

# ── Step 3: Python training dependencies ─────────────────────────────
echo -e "${BOLD}── Step 3: Python training dependencies ───────────${NC}"

if $SKIP_TRAINING; then
    info "Skipping training deps (--skip-training)"
else
    # Detect conda/pip
    if command -v conda &>/dev/null && [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        PIP="pip"
        info "Using conda env: ${CONDA_DEFAULT_ENV}"
    elif command -v pip3 &>/dev/null; then
        PIP="pip3"
    else
        PIP="pip"
    fi

    # SetFit (few-shot classifier, ~18s to train, CPU-friendly)
    info "Installing SetFit (few-shot transaction classifier)..."
    run "${PIP} install setfit sentence-transformers"
    echo ""

    # peft + trl (LoRA fine-tuning, works on CPU and GPU)
    info "Installing peft + trl (LoRA fine-tuning)..."
    run "${PIP} install peft trl"
    echo ""

    if ! $DRY_RUN; then
        ok "Training dependencies installed"
    fi
fi
echo ""

# ── Step 4: Verify installation ──────────────────────────────────────
echo -e "${BOLD}── Step 4: Verification ───────────────────────────${NC}"

if $DRY_RUN; then
    info "Dry run — skipping verification"
else
    ERRORS=0

    # Check Ollama
    if ! $SKIP_OLLAMA; then
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            ok "Ollama server: running"
        else
            warn "Ollama server: not responding (start with: ollama serve)"
            ERRORS=$((ERRORS + 1))
        fi
    fi

    # Check Python deps
    if ! $SKIP_TRAINING; then
        python3 -c "import setfit" 2>/dev/null && ok "SetFit: installed" || { warn "SetFit: import failed"; ERRORS=$((ERRORS + 1)); }
        python3 -c "import peft" 2>/dev/null && ok "peft: installed" || { warn "peft: import failed"; ERRORS=$((ERRORS + 1)); }
        python3 -c "import trl" 2>/dev/null && ok "trl: installed" || { warn "trl: import failed"; ERRORS=$((ERRORS + 1)); }
    fi

    # Check torch + device
    python3 -c "
import torch
if torch.cuda.is_available():
    print('PyTorch: GPU mode (' + torch.cuda.get_device_name(0) + ')')
else:
    print('PyTorch: CPU mode')
" 2>/dev/null || warn "PyTorch: not importable"

    echo ""
    if [ "$ERRORS" -eq 0 ]; then
        ok "All checks passed!"
    else
        warn "${ERRORS} check(s) had issues (see warnings above)"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Installation complete${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""
echo "  Ollama models:  ${VLM_MODEL}, ${TEXT_MODEL}"
echo "  Hardware tier:   ${TIER}"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Label receipts (with AI suggestions):"
echo "     hledger_preprocessor --tui-label-receipts --config config.yaml"
echo ""
echo "  2. Auto-label receipts (non-interactive):"
echo "     hledger_preprocessor --make-ai-labels --config config.yaml"
echo ""
echo "  3. Train classifiers on your labelled data:"
echo "     hledger_preprocessor --train-models --config config.yaml"
echo ""
if [ "${VLM_MODEL}" != "qwen3-vl:2b" ] || [ "${TEXT_MODEL}" != "qwen3:0.6b" ]; then
    echo "  Note: Non-default models selected. Add to config.yaml:"
    echo ""
    echo "    ai:"
    echo "      vlm_model: \"${VLM_MODEL}\""
    echo "      text_model: \"${TEXT_MODEL}\""
    echo ""
fi
