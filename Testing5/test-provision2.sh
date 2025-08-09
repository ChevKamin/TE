#!/bin/bash
set -e

echo "=== Updating base system ==="
apt-get update && apt-get install -y git wget unzip ffmpeg

echo "=== Switching to ComfyUI workspace ==="
cd /workspace/ComfyUI

echo "=== Upgrading pip (quiet) ==="
pip install --upgrade pip --quiet --no-cache-dir

# If you don't need flet, skip it completely:
# pip uninstall -y flet || true

# If you DO need flet, pin packaging to match:
pip install packaging==23.2 --quiet --no-cache-dir

echo "=== Installing core ComfyUI dependencies ==="
pip install -r requirements.txt --quiet --no-cache-dir

# ----------------------------
# 1. INSTALL REQUIRED CUSTOM NODES
# ----------------------------
cd /workspace/ComfyUI/custom_nodes

# Video Helper Suite
[ ! -d "ComfyUI-VideoHelperSuite" ] && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

# KJ Nodes
if [ ! -d "ComfyUI-KJNodes" ]; then
    git clone https://github.com/kijai/ComfyUI-KJNodes
    cd ComfyUI-KJNodes
    pip install -r requirements.txt --quiet --no-cache-dir
    cd ..
fi

# WAN 2.2-specific nodes
[ ! -d "ComfyUI-WanNodes" ] && git clone https://huggingface.co/CheeseDaddy/ComfyUI-Wan2.2-nodes ComfyUI-WanNodes

# rgthree power lora loader
[ ! -d "rgthree-comfy" ] && git clone https://github.com/rgthree/rgthree-comfy

# ----------------------------
# 2. INSTALL MODELS (WAN 2.2 + VAE + LORAs)
# ----------------------------
MODEL_DIR="/workspace/ComfyUI/models/checkpoints"
VAE_DIR="/workspace/ComfyUI/models/vae"
LORA_DIR="/workspace/ComfyUI/models/loras"

mkdir -p "$MODEL_DIR" "$VAE_DIR" "$LORA_DIR"

# WAN 2.2 I2V models
wget -nc -O "$MODEL_DIR/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
"https://huggingface.co/CheeseDaddy/wan2.2/resolve/main/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"

wget -nc -O "$MODEL_DIR/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
"https://huggingface.co/CheeseDaddy/wan2.2/resolve/main/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

# WAN VAE
wget -nc -O "$VAE_DIR/wan_2.1_vae.safetensors" \
"https://huggingface.co/CheeseDaddy/wan2.2/resolve/main/wan_2.1_vae.safetensors"

# Example Lora from workflow
wget -nc -O "$LORA_DIR/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors" \
"https://huggingface.co/CheeseDaddy/wan2.2/resolve/main/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"

# ----------------------------
# 3. CLEANUP
# ----------------------------
apt-get clean && rm -rf /var/lib/apt/lists/*

echo "=== Provisioning complete! ==="
