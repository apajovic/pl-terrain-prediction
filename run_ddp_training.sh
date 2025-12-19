#!/bin/bash
# run_ddp_training.sh
# Script to launch distributed training with DDP on multiple GPUs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="default_config.json"
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
NPROC_PER_NODE=$NUM_GPUS
TRAINING_SCRIPT="src/python/train.py"
MASTER_ADDR="localhost"
MASTER_PORT="29500"

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -c, --config FILE         Path to config file (default: $CONFIG_FILE)"
    echo "  -g, --gpus NUM            Number of GPUs to use (default: all available, detected: $NUM_GPUS)"
    echo "  -p, --port PORT           Master port for DDP (default: $MASTER_PORT)"
    echo "  --addr ADDR               Master address for DDP (default: $MASTER_ADDR)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -c configs/transunet.json -g 4"
    echo "  $0 -c default_config.json -g 2 --addr 192.168.1.100"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -g|--gpus)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        -p|--port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validation
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo -e "${RED}Error: Training script not found: $TRAINING_SCRIPT${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Verify CUDA is available
if [ "$NUM_GPUS" -eq 0 ]; then
    echo -e "${RED}Error: No CUDA devices found. DDP requires GPU(s)${NC}"
    exit 1
fi

# Validate GPU count
if [ "$NPROC_PER_NODE" -gt "$NUM_GPUS" ]; then
    echo -e "${YELLOW}Warning: Requested $NPROC_PER_NODE GPUs but only $NUM_GPUS available${NC}"
    NPROC_PER_NODE=$NUM_GPUS
fi

# Print configuration
echo -e "${GREEN}=== Distributed Data Parallel (DDP) Training ===${NC}"
echo "Config file:        $CONFIG_FILE"
echo "Training script:    $TRAINING_SCRIPT"
echo "Number of GPUs:     $NPROC_PER_NODE"
echo "Master address:     $MASTER_ADDR"
echo "Master port:        $MASTER_PORT"
echo ""

# Check for torchrun availability (PyTorch 1.10+)
if command -v torchrun &> /dev/null; then
    echo -e "${GREEN}Using torchrun launcher${NC}"
    echo ""
    
    # Export environment variables for DDP
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    
    # Launch training with torchrun
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $TRAINING_SCRIPT \
        -c $CONFIG_FILE
else
    echo -e "${YELLOW}torchrun not found, trying torch.distributed.launch${NC}"
    echo ""
    
    # Export environment variables for DDP
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    
    # Launch training with torch.distributed.launch (older PyTorch versions)
    python3 -m torch.distributed.launch \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $TRAINING_SCRIPT \
        -c $CONFIG_FILE
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
else
    echo -e "${RED}Training failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
