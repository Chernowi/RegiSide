#!/bin/bash
# Installation script for MAPPO PyTorch training

echo "Installing PyTorch and dependencies for MAPPO training..."
echo "=========================================================="

# Install PyTorch (CPU version for compatibility, use CUDA version if GPU available)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements_pytorch.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print('✅ PyTorch', torch.__version__, 'installed successfully')"
python -c "import tensorboard; print('✅ TensorBoard installed successfully')"
python -c "import gymnasium; print('✅ Gymnasium installed successfully')"
python -c "import numpy; print('✅ NumPy installed successfully')"

echo ""
echo "=========================================================="
echo "Installation complete! Run the test:"
echo "  python test_mappo_pytorch_setup.py"
echo ""
echo "Or start training:"
echo "  python train_mappo_pytorch.py"
echo "=========================================================="
