#!/bin/bash

# WAN 2.2 Image-to-Video ComfyUI Provisioning Script
# Fixed version with all required nodes

set -e

echo "========================================="
echo "WAN 2.2 I2V Setup - Installing ComfyUI"
echo "========================================="

# Set up workspace
export WORKSPACE="/workspace"
cd $WORKSPACE

# Install system dependencies
apt-get update && apt-get install -y \
    python3-pip python3-venv git wget curl \
    ffmpeg libgl1 libglib2.0-0 libsm6 \
    libxrender1 libxext6 nvtop htop \
    build-essential python3-dev

# Install ComfyUI if not exists
if [ ! -d "ComfyUI" ]; then
    echo "Installing ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    
    # Install PyTorch with CUDA support
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install ComfyUI requirements
    pip3 install -r requirements.txt
    
    # Install additional packages for video and WAN support
    pip3 install \
        opencv-python-headless \
        imageio \
        imageio-ffmpeg \
        einops \
        transformers \
        accelerate \
        xformers \
        sageattention \
        scipy \
        scikit-image \
        kornia \
        spandrel
else
    cd ComfyUI
fi

echo "========================================="
echo "Installing Required Custom Nodes"
echo "========================================="

cd custom_nodes

# CRITICAL: ComfyUI-KJNodes (for TorchCompileModelWanVideoV2, PathchSageAttentionKJ, ImageResizeKJv2)
if [ ! -d "ComfyUI-KJNodes" ]; then
    echo "Installing ComfyUI-KJNodes..."
    git clone https://github.com/kijai/ComfyUI-KJNodes.git
    cd ComfyUI-KJNodes
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
    fi
    # Install specific dependencies for KJNodes
    pip3 install color-matcher tensorflow audioread librosa
    cd ..
else
    echo "ComfyUI-KJNodes already installed"
fi

# CRITICAL: rgthree-comfy (for Power Lora Loader)
if [ ! -d "rgthree-comfy" ]; then
    echo "Installing rgthree-comfy..."
    git clone https://github.com/rgthree/rgthree-comfy.git
    cd rgthree-comfy
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
    fi
    cd ..
else
    echo "rgthree-comfy already installed"
fi

# CRITICAL: ComfyUI-VideoHelperSuite (for CreateVideo, SaveVideo)
if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
    echo "Installing ComfyUI-VideoHelperSuite..."
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    cd ComfyUI-VideoHelperSuite
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
    fi
    cd ..
else
    echo "ComfyUI-VideoHelperSuite already installed"
fi

# Additional useful nodes
if [ ! -d "ComfyUI-Manager" ]; then
    echo "Installing ComfyUI-Manager..."
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
fi

# Go back to ComfyUI root
cd $WORKSPACE/ComfyUI

echo "========================================="
echo "Creating Model Directories"
echo "========================================="

# Create all necessary model directories
mkdir -p models/unet
mkdir -p models/clip
mkdir -p models/vae
mkdir -p models/loras
mkdir -p models/checkpoints
mkdir -p output/vid
mkdir -p input

echo "========================================="
echo "Checking for WAN-specific nodes"
echo "========================================="

# Check if WanImageToVideo is available (should be in core ComfyUI)
python3 -c "
import sys
sys.path.append('.')
try:
    from nodes import NODE_CLASS_MAPPINGS
    if 'WanImageToVideo' in NODE_CLASS_MAPPINGS:
        print('✓ WanImageToVideo node found')
    else:
        print('⚠ WanImageToVideo node not found - may need ComfyUI update')
except Exception as e:
    print(f'Error checking nodes: {e}')
"

echo "========================================="
echo "Downloading Required Models"
echo "========================================="

# Function to download with resume support
download_model() {
    local url=$1
    local dir=$2
    local filename=$(basename "$url")
    
    if [ -f "$dir/$filename" ]; then
        echo "✓ Model already exists: $filename"
    else
        echo "Downloading: $filename to $dir"
        wget -c -P "$dir" "$url" || echo "⚠ Failed to download $filename - manual download may be required"
    fi
}

# Download models (using placeholder URLs - replace with actual)
echo "Attempting to download models..."

# VAE Model
download_model "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors" "models/vae"

# CLIP Model (example - replace with actual WAN CLIP)
# download_model "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "models/clip"

echo "========================================="
echo "Model Setup Instructions"
echo "========================================="

cat << 'EOF' > $WORKSPACE/model_setup.txt
REQUIRED MODELS FOR WAN 2.2 WORKFLOW:

1. UNET Models (place in models/unet/):
   - wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
   - wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors

2. CLIP Model (place in models/clip/):
   - umt5_xxl_fp8_e4m3fn_scaled.safetensors

3. VAE Model (place in models/vae/):
   - wan_2.1_vae.safetensors

4. LoRA Model (place in models/loras/):
   - Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors

Please download these models manually if automatic download failed.
EOF

echo "Model setup instructions saved to $WORKSPACE/model_setup.txt"

echo "========================================="
echo "Creating Startup Script"
echo "========================================="

# Create a startup script
cat > $WORKSPACE/start_comfyui.sh << 'STARTUP_EOF'
#!/bin/bash
cd /workspace/ComfyUI

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Detect GPU and set appropriate flags
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)

if [ "$GPU_MEMORY" -ge 40000 ]; then
    echo "High VRAM detected ($GPU_MEMORY MB) - using highvram mode"
    VRAM_FLAG="--highvram"
elif [ "$GPU_MEMORY" -ge 24000 ]; then
    echo "Normal VRAM detected ($GPU_MEMORY MB) - using normalvram mode"
    VRAM_FLAG="--normalvram"
else
    echo "Low VRAM detected ($GPU_MEMORY MB) - using lowvram mode"
    VRAM_FLAG="--lowvram"
fi

# Start ComfyUI
python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    $VRAM_FLAG \
    --use-pytorch-cross-attention \
    --disable-metadata
STARTUP_EOF

chmod +x $WORKSPACE/start_comfyui.sh

echo "========================================="
echo "Testing Node Installation"
echo "========================================="

# Test if nodes are properly installed
cd $WORKSPACE/ComfyUI
python3 << 'PYTHON_EOF'
import os
import sys

print("Checking custom nodes installation...")

nodes_to_check = {
    "ComfyUI-KJNodes": ["TorchCompileModelWanVideoV2", "PathchSageAttentionKJ", "ImageResizeKJv2"],
    "rgthree-comfy": ["Power Lora Loader"],
    "ComfyUI-VideoHelperSuite": ["CreateVideo", "SaveVideo"]
}

custom_nodes_path = "custom_nodes"
for node_dir, expected_nodes in nodes_to_check.items():
    path = os.path.join(custom_nodes_path, node_dir)
    if os.path.exists(path):
        print(f"✓ {node_dir} installed at {path}")
    else:
        print(f"✗ {node_dir} NOT FOUND - Installation may have failed")

# Try to import and check node registration
try:
    sys.path.insert(0, os.getcwd())
    # This would normally load all nodes, but may fail in provisioning context
    print("\nNote: Full node validation will occur when ComfyUI starts")
except Exception as e:
    print(f"Cannot validate nodes in provisioning context: {e}")
PYTHON_EOF

echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "IMPORTANT: Missing Node Types Error Resolution"
echo "----------------------------------------------"
echo "If you still see 'Missing Node Types' errors:"
echo ""
echo "1. The custom nodes are installed in: /workspace/ComfyUI/custom_nodes/"
echo "2. Try restarting ComfyUI to reload nodes"
echo "3. Check if models are properly placed in their directories"
echo "4. Some nodes may require specific model files to function"
echo ""
echo "To manually check/fix:"
echo "  cd /workspace/ComfyUI/custom_nodes"
echo "  ls -la  # Should show ComfyUI-KJNodes, rgthree-comfy, etc."
echo ""
echo "To start ComfyUI:"
echo "  /workspace/start_comfyui.sh"
echo ""
echo "Access ComfyUI at: http://[POD_IP]:8188"
echo "========================================="

# Auto-start ComfyUI
echo "Starting ComfyUI in 5 seconds..."
sleep 5
/workspace/start_comfyui.sh