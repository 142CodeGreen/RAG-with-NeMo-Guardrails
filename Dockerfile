FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    pciutils \
    curl \
    gnupg \
    python3.10 \
    python3.10-venv \
    docker-compose \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update \
    && apt-get install -y nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA driver (assuming 545 is the latest version for your GPUs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-headless-545 \
    nvidia-utils-545 \
    && rm -rf /var/lib/apt/lists/*

# Verify NVIDIA driver installation
RUN nvidia-smi

# Set working directory
WORKDIR /app

# Create a virtual environment
RUN python3.10 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt

# Copy application code
COPY . .

# Ensure GPU can access the ./Storage directory
#RUN chown -R root:root ./Storage && \
#    chmod -R 777 ./Storage

# Download Docker Compose file for Milvus
RUN wget -O docker-compose.yml https://github.com/milvus-io/milvus/releases/download/v2.4.11/milvus-standalone-docker-compose-gpu.yml

# Script to start Milvus
COPY start_milvus.sh .
RUN chmod +x start_milvus.sh

# Expose port for your application
EXPOSE 7860

# Command to run your application
CMD ["python3", "app.py"]
