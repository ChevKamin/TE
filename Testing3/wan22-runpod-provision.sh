#!/bin/bash

# This file will be sourced in init.sh
# https://github.com/ai-dock/comfyui
# Fixed for WAN 2.2 with proper dependency resolution

# Save the workflow JSON as default
DEFAULT_WORKFLOW="https://raw.githubusercontent.com/ChevKamin/TE/refs/heads/main/Testing3/wan22main.json"

APT_PACKAGES=(
    "ffmpeg"
    "libgl1"
    "libglib2.0-0"
    "libsm6"
    "libxrender1"
    "libxext6"
    "nvtop"
    "htop"
    "build-essential"
    "python3-dev"
    "libportaudio2"
    "libportaudiocpp0"
    "portaudio19-dev"
)

# CRITICAL: Use specific versions to avoid conflicts
PIP_PACKAGES=(
    # Core dependencies with specific versions
    "numpy==1.26.4"  # Compatible with both tensorflow and colour-science
    "torch==2.4.1"
    "torchvision"
    "torchaudio"
    
    # Video and image processing
    "opencv-python-headless"
    "imageio"
    "imageio-ffmpeg"
    "einops"
    "transformers"
    "accelerate"
    
    # Other dependencies
    "scipy"
    "scikit-image"
    "kornia"
    "spandrel"
    "color-matcher"
    "audioread"
    "librosa"
    "matplotlib"
    "numba"
    "omegaconf"
    "safetensors"
    "tqdm"
    "psutil"
    "Pillow"
)

# XFormers will be installed separately to match torch version
XFORMERS_PACKAGE="xformers==0.0.28.post1"

# TensorFlow with compatible version
TENSORFLOW_PACKAGE="tensorflow==2.19.0"

# Colour-science with compatible version
COLOUR_SCIENCE_PACKAGE="colour-science==0.4.4"

NODES=(
    # CRITICAL - Required for WAN workflow
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/comfyanonymous/ComfyUI"  # Added comfy-core
    
    # Highly recommended
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/crystian/ComfyUI-Crystools"
    
    # Optional utilities
    "https://github.com/WASasquatch/was-node-suite-comfyui"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/melMass/comfy_mtb"
)

WORKFLOWS=(
    # Your workflow will be loaded from DEFAULT_WORKFLOW
)

CHECKPOINT_MODELS=(
    # WAN 2.2 uses UNet format, not checkpoint format
)

UNET_MODELS=(
    # WAN 2.2 14B models - Replace with actual URLs
    # "https://huggingface.co/wan/wan22/resolve/main/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    # "https://huggingface.co/wan/wan22/resolve/main/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
)

CLIP_MODELS=(
    # UMT5-XXL CLIP model for WAN
    # "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
)

LORA_MODELS=(
    # WAN 2.1 I2V LoRA
    # "https://huggingface.co/wan/loras/resolve/main/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
)

VAE_MODELS=(
    # WAN 2.1 VAE
    # "https://huggingface.co/wan/wan_vae/resolve/main/wan_2.1_vae.safetensors"
)

ESRGAN_MODELS=(
    # Optional upscaling models
)

CONTROLNET_MODELS=(
    # Not used in WAN workflow
)

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

# GPU Detection and Configuration for WAN 2.2
function detect_gpu_config() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^[[:space:]]*//')
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
        
        echo "========================================="
        echo "GPU Detection for WAN 2.2 Workflow"
        echo "========================================="
        echo "Detected GPU: $GPU_NAME"
        echo "VRAM: ${GPU_MEMORY}MB"
        
        # Set environment variables based on GPU
        if [[ "$GPU_MEMORY" -ge 40000 ]]; then
            echo "Profile: High VRAM GPU"
            export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
            export MEMORY_MODE="highvram"
        elif [[ "$GPU_MEMORY" -ge 24000 ]]; then
            echo "Profile: Standard VRAM GPU"
            export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
            export MEMORY_MODE="normalvram"
        else
            echo "WARNING: Low VRAM detected - WAN 2.2 14B models may not run properly"
            export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
            export MEMORY_MODE="lowvram"
        fi
        
        # Common optimizations
        export CUDA_LAUNCH_BLOCKING=0
        export TORCH_CUDNN_V8_API_ENABLED=1
        export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_compile_cache
        export TORCHINDUCTOR_FX_GRAPH_CACHE=1
    else
        echo "No GPU detected - CPU mode"
        export MEMORY_MODE="cpu"
    fi
}

function fix_dependencies() {
    echo "========================================="
    echo "Fixing Python Dependencies"
    echo "========================================="
    
    # First uninstall conflicting packages
    pip uninstall -y xformers torch torchvision torchaudio numpy tensorflow colour-science 2>/dev/null || true
    
    # Install numpy first (base dependency)
    pip_install numpy==1.26.4
    
    # Install PyTorch with specific version
    pip_install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install xformers compatible with torch 2.4.1
    pip_install xformers==0.0.28.post1 --no-deps
    
    # Install tensorflow with compatible numpy
    pip_install tensorflow==2.19.0
    
    # Install colour-science
    pip_install colour-science==0.4.4
    
    # Install sageattention without dependencies first
    pip_install sageattention --no-deps
    
    echo "✓ Dependencies fixed"
}

function provisioning_start() {
    if [[ ! -d /opt/environments/python ]]; then 
        export MAMBA_BASE=true
    fi
    source /opt/ai-dock/etc/environment.sh
    source /opt/ai-dock/bin/venv-set.sh comfyui
    
    # Detect and configure GPU
    detect_gpu_config
    
    # Check for HuggingFace token
    if provisioning_has_valid_hf_token; then
        echo "✓ HuggingFace token detected"
    else
        echo "⚠ No valid HF_TOKEN found"
    fi
    
    provisioning_print_header
    
    # Fix dependencies BEFORE installing anything else
    fix_dependencies
    
    provisioning_get_apt_packages
    provisioning_get_nodes
    provisioning_get_pip_packages_fixed  # Use fixed version
    
    # Download models if URLs are provided
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
    provisioning_get_default_workflow
    
    # Fix KJNodes installation
    fix_kjnodes_installation
    
    provisioning_verify_nodes
    provisioning_print_end
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

function provisioning_get_pip_packages_fixed() {
    echo "Installing Python packages with fixed versions..."
    
    # Install packages that don't have conflicts
    for package in "${PIP_PACKAGES[@]}"; do
        # Skip packages we've already installed in fix_dependencies
        if [[ "$package" != "numpy"* ]] && [[ "$package" != "torch"* ]] && [[ "$package" != "xformers"* ]]; then
            pip_install "$package" || echo "Warning: Failed to install $package"
        fi
    done
    
    echo "✓ Python packages installed"
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
                   # Install requirements without upgrading core packages
                   pip_install -r "$requirements" --no-deps
                   # Then install with deps but without upgrade
                   pip_install -r "$requirements" --no-upgrade
                fi
            fi
        else
            printf "Installing CRITICAL node: %s...\n" "${dir}"
            git clone "${repo}" "${path}" --recursive
            
            # Special handling for each node's requirements
            if [[ "$dir" == "ComfyUI-KJNodes" ]]; then
                cd "$path"
                # Install KJNodes requirements carefully
                pip_install -r requirements.txt --no-deps 2>/dev/null || true
                pip_install color-matcher --no-deps
                pip_install audioread librosa --no-deps
                cd -
            elif [[ "$dir" == "ComfyUI-VideoHelperSuite" ]]; then
                cd "$path"
                pip_install -r requirements.txt --no-deps 2>/dev/null || true
                cd -
            elif [[ "$dir" == "ComfyUI" ]]; then  # Special handling for comfy-core
                cd "$path"
                pip_install -r requirements.txt --no-deps 2>/dev/null || true
                cd -
            elif [[ -e $requirements ]]; then
                pip_install -r "${requirements}" --no-deps 2>/dev/null || true
            fi
        fi
    done
    echo "✓ Custom nodes installed"
}

function fix_kjnodes_installation() {
    echo "========================================="
    echo "Fixing KJNodes Installation"
    echo "========================================="
    
    cd /opt/ComfyUI/custom_nodes
    
    # Ensure KJNodes is properly installed
    if [ -d "ComfyUI-KJNodes" ]; then
        cd ComfyUI-KJNodes
        
        # Check if __init__.py exists and is properly configured
        if [ ! -f "__init__.py" ]; then
            echo "Creating __init__.py for KJNodes..."
            cat > __init__.py << 'INIT_EOF'
import os
import sys
import folder_paths
import importlib

node_list = []

# Add this directory to path
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

# Try to import all node files
for filename in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if filename.endswith('.py') and filename not in ['__init__.py', 'install.py']:
        try:
            module = importlib.import_module(filename[:-3])
            if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                node_list.append(module)
        except Exception as e:
            print(f"Failed to import {filename}: {e}")

# Combine all nodes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module in node_list:
    if hasattr(module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
INIT_EOF
        fi
        
        # Install missing Python dependencies
        pip_install color-matcher --no-deps
        pip_install audioread --no-deps
        pip_install librosa --no-deps
        
        cd ..
    fi
    
    # Fix VideoHelperSuite
    if [ -d "ComfyUI-VideoHelperSuite" ]; then
        cd ComfyUI-VideoHelperSuite
        pip_install imageio imageio-ffmpeg --no-deps
        cd ..
    fi
    
    cd /opt/ComfyUI
}

function provisioning_verify_nodes() {
    echo "========================================="
    echo "Verifying Node Installation"
    echo "========================================="
    
    # Python verification of nodes
    python3 << 'VERIFY_EOF'
import sys
import os
sys.path.insert(0, '/opt/ComfyUI')
sys.path.insert(0, '/opt/ComfyUI/custom_nodes')

def check_nodes():
    found_nodes = {}
    
    # Check KJNodes
    try:
        sys.path.insert(0, '/opt/ComfyUI/custom_nodes/ComfyUI-KJNodes')
        import __init__ as kj
        if hasattr(kj, 'NODE_CLASS_MAPPINGS'):
            found_nodes.update(kj.NODE_CLASS_MAPPINGS)
            print(f"✓ KJNodes: {len(kj.NODE_CLASS_MAPPINGS)} nodes loaded")
    except Exception as e:
        print(f"✗ KJNodes failed: {e}")
    
    # Check VideoHelperSuite
    try:
        sys.path.insert(0, '/opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite')
        import __init__ as vhs
        if hasattr(vhs, 'NODE_CLASS_MAPPINGS'):
            found_nodes.update(vhs.NODE_CLASS_MAPPINGS)
            print(f"✓ VideoHelperSuite: {len(vhs.NODE_CLASS_MAPPINGS)} nodes loaded")
    except Exception as e:
        print(f"✗ VideoHelperSuite failed: {e}")
    
    # Check for required nodes
    required = ['CreateVideo', 'SaveVideo', 'TorchCompileModelWanVideoV2', 
                'PathchSageAttentionKJ', 'WanImageToVideo', 'ImageResizeKJv2']
    
    print("\nRequired nodes status:")
    for node in required:
        if node in found_nodes:
            print(f"  ✓ {node}")
        else:
            print(f"  ✗ {node} MISSING")
            # Look for similar nodes
            similar = [n for n in found_nodes.keys() if node.lower()[:5] in n.lower()]
            if similar:
                print(f"    Similar found: {similar[:3]}")

check_nodes()
VERIFY_EOF
    
    # Create model checklist
    cat > ${WORKSPACE}/wan22_model_checklist.txt << 'EOF'
WAN 2.2 REQUIRED MODELS CHECKLIST
==================================

Place these models in the specified directories:

□ models/unet/
  - wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  - wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors

□ models/clip/
  - umt5_xxl_fp8_e4m3fn_scaled.safetensors

□ models/vae/
  - wan_2.1_vae.safetensors

□ models/loras/
  - Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors

Without these models, the workflow will not function!
EOF
    
    echo ""
    echo "Model checklist saved to: ${WORKSPACE}/wan22_model_checklist.txt"
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
        echo "Loading WAN 2.2 workflow..."
        workflow_json=$(curl -s "$DEFAULT_WORKFLOW")
        if [[ -n $workflow_json ]]; then
            # Save to default graph
            echo "export const defaultGraph = $workflow_json;" > /opt/ComfyUI/web/scripts/defaultGraph.js
            # Also save as a loadable workflow
            echo "$workflow_json" > /opt/ComfyUI/user/default/workflows/wan22_i2v.json
            echo "✓ WAN 2.2 workflow loaded as default"
        fi
    fi
}

function provisioning_get_models() {
    if [[ -z $2 ]]; then return 1; fi
    
    dir="$1"
    mkdir -p "$dir"
    shift
    arr=("$@")
    
    if [ ${#arr[@]} -eq 0 ]; then
        return 0
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
    printf "#     WITH DEPENDENCY FIXES                 #\n"
    printf "#                                            #\n"
    printf "#  Installing:                               #\n"
    printf "#  - Fixed PyTorch 2.4.1                    #\n"
    printf "#  - Compatible numpy 1.26.4                #\n"
    printf "#  - ComfyUI-KJNodes                        #\n"
    printf "#  - rgthree-comfy                          #\n"
    printf "#  - VideoHelperSuite                       #\n"
    printf "#  - ComfyUI (core)                         #\n"
    printf "#                                            #\n"
    printf "##############################################\n\n"
}

function provisioning_print_end() {
    printf "\n##############################################\n"
    printf "#                                            #\n"
    printf "#   ✓ WAN 2.2 Provisioning Complete!        #\n"
    printf "#                                            #\n"
    printf "#   Fixed Dependencies:                      #\n"
    printf "#   - PyTorch 2.4.1                         #\n"
    printf "#   - numpy 1.26.4                          #\n"
    printf "#   - xformers 0.0.28.post1                 #\n"
    printf "#   - tensorflow 2.19.0                     #\n"
    printf "#                                            #\n"
    printf "#   Access ComfyUI at:                       #\n"
    printf "#   http://[POD_IP]:8188                    #\n"
    printf "#                                            #\n"
    printf "#   IMPORTANT: Add WAN 2.2 models to:       #\n"
    printf "#   /workspace/ComfyUI/models/              #\n"
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
