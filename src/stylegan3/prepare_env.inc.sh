export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=/usr/local/cuda/targets/x86_64-linux/include
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib
export OPENCV_IO_ENABLE_OPENEXR=1
export CC=/usr/bin/gcc
export CACHE_DIR=./
export MESA_GL_VERSION_OVERRIDE=3.3

. ~/miniconda3/etc/profile.d/conda.sh
#conda init bash

conda activate stylegan3
