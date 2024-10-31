#!/bin/bash
set -e

# Ensure GPU is visible and set capabilities
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Start Milvus server with GPU acceleration
# Replace 'milvus' with the correct command if different for your setup
# Assuming Milvus is installed and the 'milvus' command is available
milvus run standalone --config /milvus/configs/milvus.yaml

# Optionally, you might want to start your Python application after Milvus is up
# python /app.py
