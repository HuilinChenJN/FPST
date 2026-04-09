#!/bin/bash
set -e

echo "=========================================="
echo "MMREC Environment installation script"
echo "=========================================="

echo "[1/4] Create Conda Environment..."
conda create -n MMREC python=3.10.19 -y


echo "[2/4] activate..."
source activate MMREC

echo "[3/4] install conda package..."
conda install -y \
    numpy=1.26.4 \
    scipy=1.15.3 \
    scikit-learn=1.7.2 \
    pandas=2.3.3 \
    matplotlib \
    ipykernel \
    mkl \
    mkl-service

echo "[4/4] install PyTorch dependencies..."
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "install PyTorch Geometric..."
pip install torch-geometric==2.7.0


pip install pyg-lib==0.5.0+pt27cu128 \
    torch-scatter==2.1.2+pt27cu128 \
    torch-sparse==0.6.18+pt27cu128 \
    torch-cluster==1.6.3+pt27cu128 \
    torch-spline-conv==1.2.2+pt27cu128 \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html


pip install \
    tqdm \
    pyyaml \
    requests \
    prettytable \
    pyecharts \
    rich \
    swanlab \
    lmdb \
    protobuf \
    simplejson \
    pyaml \
    nvidia-ml-py

echo ""
echo "Verify installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

echo ""
echo "=========================================="
echo "✅ MMREC Environment installation Finished!"
echo "=========================================="
echo ""
echo "you can run this command:"
echo "  conda activate MMREC"
echo ""