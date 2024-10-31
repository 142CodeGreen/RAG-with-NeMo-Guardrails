# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Update and install basic packages
RUN apt-get update && apt-get install -y \
    docker.io \
    pciutils \
    curl \
    gpg \
    nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA Container Toolkit setup
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update \
    && apt-get install -y nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-headless-545 \
    nvidia-utils-545 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GPU usage
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy Milvus configuration if needed
COPY milvus.yaml /milvus/configs/

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py
COPY app.py .

# Copy startup script
COPY startup.sh /etc/
RUN chmod +x /etc/startup.sh

# Define volumes for configuration if needed
VOLUME ["/milvus/configs"]

# Run the container application
# Here we use the startup script which would include starting Milvus with GPU support
CMD ["/etc/startup.sh"]
