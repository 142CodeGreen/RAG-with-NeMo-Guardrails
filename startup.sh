#!/bin/bash
set -e

# Ensure GPU visibility and capabilities are set for NVIDIA GPUs
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Wait for the network to be up (optional, for ensuring network services like Docker are ready)
sleep 5

# Start Milvus with GPU support
# This command assumes you're using a version of Milvus that supports GPU acceleration
milvus run standalone --config /milvus/configs/milvus.yaml

# Log that Milvus has been started
echo "Milvus has been started with GPU acceleration."

# Optionally, if you need to start your Python application after Milvus:
# python3 /app/app.py

# If you need to keep the container running after starting Milvus:
tail -f /dev/null
