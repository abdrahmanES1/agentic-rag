#!/bin/bash
# Moroccan RAG v12 Installation Script
# For RTX 3060 12GB with Qwen quantized model

set -e

echo "=================================="
echo "Moroccan RAG v12 Installation"
echo "=================================="
echo ""

# Check GPU
echo "Step 1: Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo "Detected VRAM: ${VRAM}MB"

    if [ "$VRAM" -lt 11000 ]; then
        echo "⚠️  WARNING: Less than 11GB VRAM detected. Qwen quantized needs ~7GB."
        echo "   Continue anyway? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ GPU check passed"
    fi
else
    echo "⚠️  WARNING: nvidia-smi not found. GPU check skipped."
fi
echo ""

# Check disk space
echo "Step 2: Checking disk space..."
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available space: ${AVAILABLE}GB"
if [ "$AVAILABLE" -lt 50 ]; then
    echo "⚠️  WARNING: Less than 50GB free. Models need ~40GB."
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Disk space check passed"
fi
echo ""

# Install Ollama
echo "Step 3: Installing Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama already installed: $(ollama --version)"
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
fi
echo ""

# # Start Ollama service
# echo "Step 4: Starting Ollama service..."
# if systemctl is-active --quiet ollama; then
#     echo "✓ Ollama service already running"
# else
#     sudo systemctl start ollama
#     sleep 3
#     echo "✓ Ollama service started"
# fi
# echo ""

# Pull quantized Qwen model
echo "Step 5: Pulling Qwen2.5-VL quantized model..."
echo "This will download ~7-8GB. It may take 10-20 minutes."
echo ""

if ollama list | grep -q "qwen2.5-vl:7b-q4_0"; then
    echo "✓ Qwen quantized model already downloaded"
else
    echo "Downloading qwen2.5-vl:7b-q4_0..."
    ollama pull qwen2.5-vl:7b-q4_0
    echo "✓ Model downloaded"
fi
echo ""

# Test model
echo "Step 6: Testing Qwen model..."
echo "Sending test prompt to verify model works..."
echo '{"model":"qwen2.5-vl:7b-q4_0","messages":[{"role":"user","content":"Hello"}],"stream":false}' | \
    curl -s -X POST http://localhost:11434/api/chat -d @- > /tmp/ollama_test.json

if grep -q "content" /tmp/ollama_test.json; then
    echo "✓ Qwen model responds correctly"
    rm /tmp/ollama_test.json
else
    echo "✗ Model test failed. Check /tmp/ollama_test.json for details"
    exit 1
fi
echo ""

# Check VRAM usage
echo "Step 7: Checking VRAM usage after model load..."
if command -v nvidia-smi &> /dev/null; then
    sleep 2
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    echo "VRAM used: ${USED}MB"

    if [ "$USED" -gt 10000 ]; then
        echo "⚠️  WARNING: Using more than 10GB. Expected ~7GB for quantized model."
        echo "   You may have pulled the wrong model. Check: ollama list"
    else
        echo "✓ VRAM usage normal for quantized model"
    fi
else
    echo "⚠️  Cannot check VRAM (nvidia-smi not available)"
fi
echo ""

# Install Python packages
echo "Step 8: Installing Python packages..."
pip install --quiet --upgrade pip

# Check if packages already installed
MISSING=""
for pkg in pillow; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo "Installing missing packages:$MISSING"
    pip install $MISSING
    echo "✓ Python packages installed"
else
    echo "✓ Python packages already installed"
fi
echo ""

# Create backup directory
echo "Step 9: Setting up directories..."
mkdir -p kb_v11_backup
mkdir -p kb_v12_test
echo "✓ Directories created"
echo ""

# Summary
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Backup your current system:"
echo "     cp -r kb_v11 kb_v11_backup/"
echo "     cp moroccan_rag_v11.py moroccan_rag_v11.py.backup"
echo ""
echo "  2. Add v12 code modifications (see v12_code_patch.py)"
echo ""
echo "  3. Test on sample PDF:"
echo "     python test_qwen_ocr.py"
echo ""
echo "  4. Build v12 KB:"
echo "     python build_kb_v12.py"
echo ""
echo "Ready to proceed? Run the tests first!"
