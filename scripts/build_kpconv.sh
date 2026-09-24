#!/bin/bash
set -e
cd "$(dirname "$0")/../src/KPConv-PyTorch/cpp_wrappers"
export TORCH_CUDA_ARCH_LIST="8.6"
sh compile_wrappers.sh
echo "KPConv extensions built."
