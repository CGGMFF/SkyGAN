#export CUDA_HOME=/usr/local/cuda
#export CPLUS_INCLUDE_PATH=/usr/local/cuda/targets/x86_64-linux/include
#export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib
source /opt/intel/oneapi/mkl/2024.2/env/vars.sh
source /opt/intel/oneapi/compiler/2024.2/env/vars.sh
source /opt/intel/oneapi/tbb/2021.13/env/vars.sh
export OPENCV_IO_ENABLE_OPENEXR=1
export CC=/usr/bin/gcc
export CACHE_DIR=/tmp
export MESA_GL_VERSION_OVERRIDE=3.3

. ~/miniconda3/etc/profile.d/conda.sh
#conda init bash

conda activate skygan_2_1_40_xpu  # the environment name should match the one in environment_intel.yml
