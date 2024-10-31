FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Update and install basic packages
RUN apt-get update && apt-get install -y \
    docker.io \
    pciutils \
    curl \
    gpg \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA Container Toolkit
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

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py
COPY app.py .

# Run the container application
CMD ["python", "app.py"
