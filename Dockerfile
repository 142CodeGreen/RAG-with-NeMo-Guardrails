FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Update apt package index and install necessary packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    docker.io \
    pciutils \
    curl \
    gnupg \
    python3.10 \
    python3.10-venv \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update \
    && apt-get install -y nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA driver if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-driver-510 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a virtual environment
RUN python3.10 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies from requirements.txt
COPY requirements.txt .
# Update to use upgrade flag
RUN pip install --upgrade -r requirements.txt

# Copy application code
COPY . .

# Create and set permissions for the storage directory
# This assumes you're running the container as root or with sufficient permissions.
# If not, consider setting a specific user or group.
RUN mkdir -p /app/storage && \
    chown -R root:root /app/storage && \
    chmod -R 777 /app/storage

# Define volume for persistent storage
VOLUME ["/app/storage"]

# Expose port for Gradio (assuming default port)
EXPOSE 7860

# Command to run your application
CMD ["python3", "app.py"]
