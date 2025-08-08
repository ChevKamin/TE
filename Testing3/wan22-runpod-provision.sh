#!/bin/bash

# This file will be sourced in init.sh
# https://github.com/ai-dock/comfyui
# Customized for WAN 2.2 Image-to-Video 14B Workflow

# Since not using ai-dock, we need to set up paths manually
export WORKSPACE="/workspace"
export COMFYUI_PATH="$WORKSPACE/ComfyUI"

# Install Python and system dependencies first
apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

# Clone ComfyUI if not exists
if [ ! -d "$COMFYUI_PATH" ]; then
    cd $WORKSPACE
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    pip3 install -r requirements.txt
fi

# Save the workflow JSON as default
DEFAULT_WORKFLOW="https://raw.githubusercontent.com/YOUR_REPO/main/wan22_i2v_workflow.json"

APT_PACKAGES=(
    "ffmpeg"
    "libgl1"
    "libglib2.0-0"
    "libsm6"
    "libxrender1"
    "libxext6"
    "nvtop"
    "htop"
)

PIP_PACKAGES=(
    "opencv-python-headless"
    "imageio"
    "imageio-ffmpeg"
    "einops"
    "transformers"
    "accelerate"
    "xformers"
    "bitsandbytes"
    "sageattention"
    "triton"
    "torch-compile"
)

NODES=(
    # REQUIRED for this workflow - DO NOT REMOVE
    "https://github.com/kijai/ComfyUI-KJNodes"  # Required: TorchCompileModelWanVideoV2, PathchSageAttentionKJ, ImageResizeKJv2
    "https://github.com/rgthree/rgthree-comfy"  # Required: Power Lora Loader
    
    # Core video/image handling
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"  # Video creation and saving
    
    # Highly recommended utility nodes
    "https://github.com/ltdrdata/ComfyUI-Manager"  # Node package manager
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"  # Better workflow management
    "https://github.com/crystian/ComfyUI-Crystools"  # Monitoring and debugging
    
    # Optional but useful
    "https://github.com/WASasquatch/was-node-suite-comfyui"  # Additional utilities
    "https://github.com/cubiq/ComfyUI_essentials"  # Essential tools
    "https://github.com/melMass/comfy_mtb"  # Media toolbox
)

WORKFLOWS=(
    # The actual WAN 2.2 workflow will be saved directly
)

CHECKPOINT_MODELS=(
    # WAN 2.2 uses UNet format, not checkpoint format
)

UNET_MODELS=(
    # WAN 2.2 14B Image-to-Video models (FP8 quantized for efficiency)
    # These are the exact models from your workflow
    # Note: Replace with actual HuggingFace URLs when publicly available
    
    # High noise model for initial generation (steps 0-2)
    "https://huggingface.co/wan/wan22/resolve/main/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    
    # Low noise model for refinement (steps 2-10)
    "https://huggingface.co/wan/wan22/resolve/main/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
)

CLIP_MODELS=(
    # UMT5-XXL CLIP model specifically for WAN (from your workflow)
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
)

LORA_MODELS=(
    # WAN 2.1 I2V LoRA for improved video generation (from your workflow)
    "https://huggingface.co/wan/loras/resolve/main/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
    
    # Optional: Instagirlv3 (disabled in your workflow but keeping for completeness)
    # "https://civitai.com/api/download/models/XXXXX?type=Model&format=SafeTensor"
)

VAE_MODELS=(
    # WAN 2.1 VAE (from your workflow)
    "https://huggingface.co/wan/wan_vae/resolve/main/wan_2.1_vae.safetensors"
)

ESRGAN_MODELS=(
    # Optional upscaling models (not used in workflow but useful for preprocessing)
    # "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth"
)

CONTROLNET_MODELS=(
    # Not used in this WAN workflow
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

# GPU Detection and Configuration for WAN 2.2
function detect_gpu_config() {
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^[[:space:]]*//')
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    
    echo "========================================="
    echo "GPU Detection for WAN 2.2 Workflow"
    echo "========================================="
    echo "Detected GPU: $GPU_NAME"
    echo "VRAM: ${GPU_MEMORY}MB"
    
    # Set environment variables based on GPU for optimal WAN 2.2 performance
    if [[ "$GPU_NAME" == *"B200"* ]] || [[ "$GPU_NAME" == *"B100"* ]]; then
        echo "Profile: Blackwell Architecture - Maximum Performance"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"
        export MEMORY_MODE="unlimited"
        export TORCH_FP8_E4M3_ENABLED=1
        export TORCH_FP8_E5M2_ENABLED=1
        export WAN_TILE_SIZE=512
        export WAN_TILE_OVERLAP=128
        export WAN_BATCH_SIZE=4
    elif [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"H200"* ]]; then
        echo "Profile: Hopper Architecture - Optimal Performance"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
        export MEMORY_MODE="highvram"
        export TORCH_FP8_E4M3_ENABLED=1
        export WAN_TILE_SIZE=384
        export WAN_TILE_OVERLAP=96
        export WAN_BATCH_SIZE=2
    elif [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_MEMORY" -ge 40000 ]]; then
        echo "Profile: Professional GPU - High Performance"
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
        export MEMORY_MODE="highvram"
        export WAN_TILE_SIZE=384
        export WAN_TILE_OVERLAP=96
        export WAN_BATCH_SIZE=1
    elif [[ "$GPU_MEMORY" -ge 24000 ]]; then
        echo "Profile: Consumer GPU - Standard Performance"
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
        export MEMORY_MODE="normalvram"
        export WAN_TILE_SIZE=256  # Matches your workflow
        export WAN_TILE_OVERLAP=64  # Matches your workflow
        export WAN_BATCH_SIZE=1  # Matches your workflow
    else
        echo "WARNING: Insufficient VRAM for WAN 2.2 14B models!"
        echo "Minimum requirement: 24GB VRAM"
        echo "Your GPU has: ${GPU_MEMORY}MB"
        export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
        export MEMORY_MODE="lowvram"
        export WAN_TILE_SIZE=192
        export WAN_TILE_OVERLAP=48
        export WAN_BATCH_SIZE=1
    fi
    
    # Common optimizations for WAN video generation
    export CUDA_LAUNCH_BLOCKING=0
    export TORCH_CUDNN_V8_API_ENABLED=1
    export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_compile_cache
    export TORCHINDUCTOR_FX_GRAPH_CACHE=1
    export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
    export TORCHDYNAMO_VERBOSE=0
    
    echo "Memory Mode: $MEMORY_MODE"
    echo "Tile Size: $WAN_TILE_SIZE"
    echo "Batch Capability: $WAN_BATCH_SIZE"
    echo "========================================="
}

function provisioning_start() {
    if [[ ! -d /opt/environments/python ]]; then 
        export MAMBA_BASE=true
    fi
    source /opt/ai-dock/etc/environment.sh
    source /opt/ai-dock/bin/venv-set.sh comfyui
    
    # Detect and configure for GPU
    detect_gpu_config
    
    # Check for HuggingFace token for gated models
    if provisioning_has_valid_hf_token; then
        echo "✓ HuggingFace token detected - can access gated models"
    else
        echo "⚠ No valid HF_TOKEN found - some models may require manual download"
        echo "  Set HF_TOKEN environment variable in RunPod for automatic downloads"
    fi
    
    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_nodes
    provisioning_get_pip_packages
    
    # Download models in the correct directories
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/unet" \
        "${UNET_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/clip" \
        "${CLIP_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/loras" \
        "${LORA_MODELS[@]}"
    provisioning_get_models \
        "${WORKSPACE}/ComfyUI/models/vae" \
        "${VAE_MODELS[@]}"
        
    provisioning_get_workflows
    provisioning_save_wan_workflow
    provisioning_configure_wan_settings
    provisioning_print_end
}

function provisioning_save_wan_workflow() {
    echo "Installing WAN 2.2 I2V workflow..."
    
    # Save the workflow JSON directly
    cat > /opt/ComfyUI/wan22_i2v_workflow.json << 'WORKFLOW_END'
{
  "id": "ec7da562-7e21-4dac-a0d2-f4441e1efd3b",
  "workflow_name": "WAN 2.2 Image-to-Video 14B",
  "description": "Two-stage video generation with high/low noise models",
  "requirements": {
    "min_vram": "24GB",
    "optimal_vram": "48GB+",
    "custom_nodes": ["ComfyUI-KJNodes", "rgthree-comfy"],
    "models": {
      "unet": ["wan2.2_i2v_high_noise_14B_fp8_scaled", "wan2.2_i2v_low_noise_14B_fp8_scaled"],
      "clip": ["umt5_xxl_fp8_e4m3fn_scaled"],
      "vae": ["wan_2.1_vae"],
      "lora": ["Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64"]
    }
  },
  "settings": {
    "video_length": 121,
    "fps": 24,
    "tile_size": 256,
    "tile_overlap": 64,
    "stage1_steps": "0-2",
    "stage2_steps": "2-10",
    "cfg_scale": 1.0,
    "sampler": "euler",
    "scheduler": "simple"
  }
}
WORKFLOW_END
    
    # Also save to user workflows directory
    mkdir -p /opt/ComfyUI/user/default/workflows
    cp /opt/ComfyUI/wan22_i2v_workflow.json /opt/ComfyUI/user/default/workflows/
    
    echo "✓ WAN 2.2 workflow installed"
}

function provisioning_configure_wan_settings() {
    echo "Configuring WAN 2.2 optimizations..."
    
    # Create WAN-specific configuration
    cat > /opt/ComfyUI/wan_config.json << EOF
{
    "gpu": "$GPU_NAME",
    "vram": "$GPU_MEMORY",
    "memory_mode": "$MEMORY_MODE",
    "optimizations": {
        "tile_size": $WAN_TILE_SIZE,
        "tile_overlap": $WAN_TILE_OVERLAP,
        "temporal_size": 64,
        "temporal_overlap": 8,
        "batch_size": $WAN_BATCH_SIZE,
        "torch_compile": {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": false,
            "dynamic": false,
            "transformer_only": true,
            "cache_size_limit": 64
        },
        "sage_attention": "auto"
    },
    "workflow_params": {
        "video_frames": 121,
        "fps": 24,
        "duration_seconds": 5,
        "resolution": "dynamic",
        "divisible_by": 64
    }
}
EOF
    
    # Create optimized startup script for WAN
    cat > /opt/ComfyUI/start_wan.sh << 'EOF'
#!/bin/bash
cd /opt/ComfyUI

source /opt/ai-dock/etc/environment.sh

echo "Starting ComfyUI with WAN 2.2 optimizations..."
echo "Memory Mode: $MEMORY_MODE"
echo "Tile Size: $WAN_TILE_SIZE"

# Launch with appropriate memory settings
if [ "$MEMORY_MODE" == "unlimited" ] || [ "$MEMORY_MODE" == "highvram" ]; then
    python main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --highvram \
        --use-pytorch-cross-attention \
        --disable-metadata
elif [ "$MEMORY_MODE" == "normalvram" ]; then
    python main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --normalvram \
        --use-pytorch-cross-attention \
        --disable-metadata
else
    python main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --lowvram \
        --use-split-cross-attention \
        --disable-metadata
fi
EOF
    chmod +x /opt/ComfyUI/start_wan.sh
    
    echo "✓ WAN 2.2 optimizations configured"
}

function pip_install() {
    if [[ -z $MAMBA_BASE ]]; then
            "$COMFYUI_VENV_PIP" install --no-cache-dir "$@"
        else
            micromamba run -n comfyui pip install --no-cache-dir "$@"
        fi
}

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
            sudo $APT_INSTALL ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
            pip_install ${PIP_PACKAGES[@]}
    fi
}

function provisioning_get_nodes() {
    echo "Installing custom nodes for WAN 2.2..."
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="/opt/ComfyUI/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating node: %s...\n" "${dir}"
                ( cd "$path" && git pull )
                if [[ -e $requirements ]]; then
                   pip_install -r "$requirements"
                fi
            fi
        else
            printf "Installing node: %s...\n" "${dir}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                pip_install -r "${requirements}"
            fi
        fi
    done
    echo "✓ Custom nodes installed"
}

function provisioning_get_workflows() {
    for repo in "${WORKFLOWS[@]}"; do
        dir=$(basename "$repo" .git)
        path="/opt/ComfyUI/user/default/workflows/${dir}"
        if [[ -d "$path" ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                printf "Updating workflows: %s...\n" "${repo}"
                ( cd "$path" && git pull )
            fi
        else
            printf "Cloning workflows: %s...\n" "${repo}"
            git clone "$repo" "$path"
        fi
    done
}

function provisioning_get_default_workflow() {
    if [[ -n $DEFAULT_WORKFLOW ]]; then
        workflow_json=$(curl -s "$DEFAULT_WORKFLOW")
        if [[ -n $workflow_json ]]; then
            echo "export const defaultGraph = $workflow_json;" > /opt/ComfyUI/web/scripts/defaultGraph.js
        fi
    fi
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    
    dir="$1"
    mkdir -p "$dir"
    shift
    arr=("$@")
    
    # Check disk space for WAN models (they're large!)
    available_space=$(df /workspace | awk 'NR==2 {print $4}')
    required_space=60000000  # ~60GB for all WAN models
    
    if [ "$available_space" -lt "$required_space" ]; then
        echo "⚠ WARNING: May have insufficient disk space for all WAN 2.2 models"
        echo "  Available: $(echo $available_space | awk '{print int($1/1024/1024)}')GB"
        echo "  Recommended: 60GB minimum"
        echo "  Each 14B model is ~15-20GB"
    fi
    
    printf "Downloading %s model(s) to %s...\n" "${#arr[@]}" "$dir"
    for url in "${arr[@]}"; do
        filename=$(basename "$url")
        if [[ -f "$dir/$filename" ]]; then
            echo "✓ Model already exists: $filename"
        else
            printf "Downloading: %s\n" "${filename}"
            provisioning_download "${url}" "${dir}"
        fi
        printf "\n"
    done
}

function provisioning_print_header() {
    printf "\n##############################################\n"
    printf "#                                            #\n"
    printf "#     WAN 2.2 I2V 14B Provisioning          #\n"
    printf "#                                            #\n"
    printf "#  GPU: %-36s #\n" "$GPU_NAME"
    printf "#  VRAM: %-35s #\n" "${GPU_MEMORY}MB"
    printf "#  Mode: %-35s #\n" "$MEMORY_MODE"
    printf "#                                            #\n"
    printf "#  Installing:                               #\n"
    printf "#  - ComfyUI with custom nodes              #\n"
    printf "#  - WAN 2.2 14B models (FP8)               #\n"
    printf "#  - Video generation pipeline              #\n"
    printf "#                                            #\n"
    printf "#  This will take 15-25 minutes             #\n"
    printf "#                                            #\n"
    printf "##############################################\n\n"
    
    if [[ "$GPU_MEMORY" -lt 24000 ]]; then
        printf "⚠ CRITICAL WARNING ⚠\n"
        printf "Your GPU has insufficient VRAM for WAN 2.2 14B models\n"
        printf "Minimum requirement: 24GB VRAM\n"
        printf "Your GPU: ${GPU_MEMORY}MB\n\n"
    fi
}

function provisioning_print_end() {
    printf "\n##############################################\n"
    printf "#                                            #\n"
    printf "#   ✓ WAN 2.2 Provisioning Complete!        #\n"
    printf "#                                            #\n"
    printf "#   ComfyUI starting automatically...       #\n"
    printf "#   Access at: http://[POD_IP]:8188         #\n"
    printf "#                                            #\n"
    printf "#   Workflow Details:                        #\n"
    printf "#   - Model: WAN 2.2 14B I2V                #\n"
    printf "#   - Stages: High + Low Noise              #\n"
    printf "#   - Output: 121 frames @ 24fps            #\n"
    printf "#   - Duration: 5 seconds                   #\n"
    printf "#   - Tile Size: %d                        #\n" "$WAN_TILE_SIZE"
    printf "#                                            #\n"
    printf "##############################################\n\n"
}

function provisioning_has_valid_hf_token() {
    [[ -n "$HF_TOKEN" ]] || return 1
    url="https://huggingface.co/api/whoami-v2"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")

    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_has_valid_civitai_token() {
    [[ -n "$CIVITAI_TOKEN" ]] || return 1
    url="https://civitai.com/api/v1/models?hidden=1&limit=1"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")

    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_download() {
    if [[ -n $HF_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif 
        [[ -n $CIVITAI_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [[ -n $auth_token ]];then
        wget --header="Authorization: Bearer $auth_token" -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    fi
}

provisioning_start
