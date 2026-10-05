#!/bin/bash
# run_ddp_training.sh — launch DDP training on multiple GPUs
#
# HARDWARE NOTES (MDCSHW-LAB-CAP3):
#   Motherboard : ASUS PRIME X299-DELUXE II (LGA 2066, single PSU)
#   GPUs        : 2× RTX 3090 on SEPARATE PCIe root complexes (C1:00.0, E1:00.0)
#   PCIe AER    : NOT supported by platform → any PCIe error is uncontainable
#   Symptom     : Machine hard-freezes (instant power-off) under sustained dual-GPU load
#   Root cause  : Likely PSU transient overload; pviol 37-40% even at 250W cap
#
# This script applies three layers of mitigation:
#   1) Aggressive GPU power cap (180W) with persistence mode
#   2) NCCL disabled entirely — use gloo backend (CPU-side collectives)
#   3) The training script uses gradient accumulation to reduce allreduce frequency

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONFIG_FILE="default_config.json"
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
NPROC_PER_NODE=$NUM_GPUS
TRAINING_SCRIPT="src/python/train.py"
MASTER_ADDR="localhost"
MASTER_PORT="29500"
DDP_BACKEND="gloo"          # gloo = CPU collectives (safe); nccl = GPU collectives (crashes)
POWER_LIMIT_W=180            # Watts per GPU — aggressive cap to avoid PSU transients
GRAD_ACCUM_STEPS=4           # Gradient accumulation steps (reduces allreduce frequency)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -c, --config FILE        Config file (default: $CONFIG_FILE)"
    echo "  -g, --gpus NUM           Number of GPUs (default: $NUM_GPUS)"
    echo "  -p, --port PORT          Master port (default: $MASTER_PORT)"
    echo "  --addr ADDR              Master address (default: $MASTER_ADDR)"
    echo "  --backend BACKEND        DDP backend: gloo or nccl (default: $DDP_BACKEND)"
    echo "  --power-limit WATTS      Per-GPU power limit (default: $POWER_LIMIT_W)"
    echo "  --grad-accum STEPS       Gradient accumulation steps (default: $GRAD_ACCUM_STEPS)"
    echo "  -h, --help               Show help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)       CONFIG_FILE="$2"; shift 2;;
        -g|--gpus)         NPROC_PER_NODE="$2"; shift 2;;
        -p|--port)         MASTER_PORT="$2"; shift 2;;
        --addr)            MASTER_ADDR="$2"; shift 2;;
        --backend)         DDP_BACKEND="$2"; shift 2;;
        --power-limit)     POWER_LIMIT_W="$2"; shift 2;;
        --grad-accum)      GRAD_ACCUM_STEPS="$2"; shift 2;;
        -h|--help)         usage;;
        *)                 echo -e "${RED}Unknown option: $1${NC}"; usage;;
    esac
done

source venv/bin/activate

[ ! -f "$CONFIG_FILE" ]      && echo -e "${RED}Config not found: $CONFIG_FILE${NC}" && exit 1
[ ! -f "$TRAINING_SCRIPT" ]  && echo -e "${RED}Script not found: $TRAINING_SCRIPT${NC}" && exit 1
[ "$NUM_GPUS" -eq 0 ]        && echo -e "${RED}No CUDA devices found${NC}" && exit 1

if [ "$NPROC_PER_NODE" -gt "$NUM_GPUS" ]; then
    echo -e "${YELLOW}Requested $NPROC_PER_NODE GPUs but only $NUM_GPUS available${NC}"
    NPROC_PER_NODE=$NUM_GPUS
fi

echo -e "${GREEN}=== DDP Training ===${NC}"
echo "Config:       $CONFIG_FILE"
echo "GPUs:         $NPROC_PER_NODE"
echo "Backend:      $DDP_BACKEND"
echo "Power limit:  ${POWER_LIMIT_W}W per GPU"
echo "Grad accum:   $GRAD_ACCUM_STEPS steps"
echo "Port:         $MASTER_PORT"
echo ""

# ── Power management ─────────────────────────────────────────────────
# Enable persistence mode so power limits survive across CUDA context resets.
# Then cap each GPU aggressively.  The 3090's TDP is 350W — at 180W the
# transient spikes (pviol) should stay within the single-PSU budget.
echo -e "${YELLOW}Enabling GPU persistence mode and capping power to ${POWER_LIMIT_W}W per GPU...${NC}"
sudo nvidia-smi -pm 1 2>/dev/null || \
    echo -e "${YELLOW}Warning: Could not enable persistence mode (needs sudo)${NC}"

for gpu_id in $(seq 0 $(($NPROC_PER_NODE - 1))); do
    sudo nvidia-smi -i $gpu_id -pl $POWER_LIMIT_W 2>/dev/null || \
        nvidia-smi -i $gpu_id -pl $POWER_LIMIT_W 2>/dev/null || \
        echo -e "${YELLOW}Warning: Could not set power limit for GPU $gpu_id (needs sudo)${NC}"
done

# Verify power settings
echo -e "${GREEN}GPU power configuration:${NC}"
nvidia-smi --query-gpu=index,name,power.limit,persistence_mode --format=csv,noheader 2>/dev/null

# ── Environment ──────────────────────────────────────────────────────
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Tell train.py which backend and grad-accum to use
export DDP_BACKEND=$DDP_BACKEND
export GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS

# If using NCCL, disable all GPU-level transports (P2P, SHM) and use sockets only
if [ "$DDP_BACKEND" = "nccl" ]; then
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Kill any leftover processes on the master port from previous runs
fuser -k $MASTER_PORT/tcp 2>/dev/null || true
sleep 1

echo ""
echo -e "${GREEN}Launching torchrun with $NPROC_PER_NODE processes (backend=$DDP_BACKEND)...${NC}"

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $TRAINING_SCRIPT \
    -c $CONFIG_FILE

EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] \
    && echo -e "${GREEN}Training completed successfully!${NC}" \
    || echo -e "${RED}Training failed with exit code $EXIT_CODE${NC}"
exit $EXIT_CODE
