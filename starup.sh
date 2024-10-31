#!/bin/bash
set -e

# Start Milvus server with GPU acceleration
# Assuming Milvus is installed and the 'milvus' command is available
# Adjust this command or the path if Milvus needs to be installed or run differently
milvus run standalone --config /milvus/configs/milvus.yaml

# Optionally, run your Python app
# python /app.py
