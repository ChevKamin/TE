#!/bin/bash

# This file will be sourced in init.sh
# https://github.com/ai-dock/comfyui
# Fixed for WAN 2.2 Image-to-Video 14B Workflow with all required nodes

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
    "scipy"
    "scikit-image"
    "kornia"
    "spandrel"
    "color-matcher"
    "tensorflow"
    "audioread"
    "librosa"
)

NODES=(
    # CRITICAL - These are REQUIRED for your workflow
    "https://github.com/kijai/ComfyUI-KJNodes"  # TorchCompileModelWanVideoV2, PathchSageAttentionKJ, ImageResizeKJv2
    "https://github.com/rgthree/rgthree-comfy"  # Power Lora Loader
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"  # CreateVideo, SaveVideo
    
    # Highly recommended
    "https://github.com/ltdrdata/ComfyUI-Manager"  # Node package manager
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"  # Better workflow management
    "https://github.com/crystian/ComfyUI-Crystools"  # Monitoring and debugging
    
    # Optional but useful
    "https://github.com/WASasquatch/was-node-suite-comfyui"  # Additional utilities
    "https://github.com/cubiq/ComfyUI_essentials"  # Essential tools
    "https://github.com/melMass/comfy_mtb"  # Media toolbox
)

WORKFLOWS=(
    # Your workflow will be loaded from DEFAULT_WORKFLOW
)

CHECKPOINT_MODELS=(
    # WAN 2.2 uses UNet format, not checkpoint format
)

UNET_MODELS=(
    # WAN 2.2 14B models - Replace these URLs with actual HuggingFace links when available
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
    provisioning_get_apt_packages
    provisioning_get_nodes
    provisioning_get_pip_packages
    
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
            printf "Installing CRITICAL node: %s...\n" "${dir}"
            git clone "${repo}" "${path}" --recursive
            if [[ -e $requirements ]]; then
                echo "Installing requirements for ${dir}..."
                pip_install -r "${requirements}"
            fi
            
            # Special handling for KJNodes
            if [[ "$dir" == "ComfyUI-KJNodes" ]]; then
                echo "Installing additional dependencies for KJNodes..."
                pip_install color-matcher tensorflow audioread librosa scipy
            fi
        fi
    done
    echo "✓ Custom nodes installed"
}

function provisioning_verify_nodes() {
    echo "========================================="
    echo "Verifying Node Installation"
    echo "========================================="
    
    # Check if critical nodes are installed
    local missing_nodes=0
    
    if [ ! -d "/opt/ComfyUI/custom_nodes/ComfyUI-KJNodes" ]; then
        echo "✗ MISSING: ComfyUI-KJNodes (Required for TorchCompileModelWanVideoV2, PathchSageAttentionKJ, ImageResizeKJv2)"
        missing_nodes=1
    else
        echo "✓ ComfyUI-KJNodes installed"
    fi
    
    if [ ! -d "/opt/ComfyUI/custom_nodes/rgthree-comfy" ]; then
        echo "✗ MISSING: rgthree-comfy (Required for Power Lora Loader)"
        missing_nodes=1
    else
        echo "✓ rgthree-comfy installed"
    fi
    
    if [ ! -d "/opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite" ]; then
        echo "✗ MISSING: ComfyUI-VideoHelperSuite (Required for CreateVideo, SaveVideo)"
        missing_nodes=1
    else
        echo "✓ ComfyUI-VideoHelperSuite installed"
    fi
    
    if [ $missing_nodes -eq 1 ]; then
        echo ""
        echo "⚠ WARNING: Some critical nodes are missing!"
        echo "The workflow may not load properly."
        echo "Try restarting ComfyUI or manually installing missing nodes."
    else
        echo ""
        echo "✓ All critical nodes are installed!"
    fi
    
    # Create a model checklist
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
    printf "#                                            #\n"
    printf "#  Installing critical nodes:                #\n"
    printf "#  - ComfyUI-KJNodes                        #\n"
    printf "#  - rgthree-comfy                          #\n"
    printf "#  - VideoHelperSuite                       #\n"
    printf "#                                            #\n"
    printf "##############################################\n\n"
}

function provisioning_print_end() {
    printf "\n##############################################\n"
    printf "#                                            #\n"
    printf "#   ✓ WAN 2.2 Provisioning Complete!        #\n"
    printf "#                                            #\n"
    printf "#   Critical Nodes Status:                   #\n"
    
    if [ -d "/opt/ComfyUI/custom_nodes/ComfyUI-KJNodes" ]; then
        printf "#   ✓ KJNodes installed                     #\n"
    else
        printf "#   ✗ KJNodes MISSING                       #\n"
    fi
    
    if [ -d "/opt/ComfyUI/custom_nodes/rgthree-comfy" ]; then
        printf "#   ✓ rgthree installed                     #\n"
    else
        printf "#   ✗ rgthree MISSING                       #\n"
    fi
    
    if [ -d "/opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite" ]; then
        printf "#   ✓ VideoHelperSuite installed            #\n"
    else
        printf "#   ✗ VideoHelperSuite MISSING              #\n"
    fi
    
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
