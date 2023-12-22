#!/bin/bash
#PBS -N skygan
#PBS -q qnvidia
#PBS -l select=1,walltime=48:00:00
#PBS -A OPEN-26-1

cd /home/martinmcgg/skygan-docker_1024

. "/mnt/proj2/open-26-1/miniconda3/etc/profile.d/conda.sh"
conda init bash
conda activate stylegan3

export PY=python3.9
export CACHE_DIR=/scratch/project/open-26-1/martinmcgg/skygan-diskcache
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_HOME='/usr/local/cuda'

nvidia-smi

cd src/stylegan3/

$PY train.py \
--cfg=stylegan3-t --gpus=8 \
--data /mnt/proj2/open-26-1/projects/SkyGAN/clouds_fisheye/auto_processed/auto_processed_20230405_1727.csv \
--resume=/mnt/proj2/open-26-1/out/skygan256/00103-stylegan3-t-auto_processed_20230405_1727-gpus8-batch32-gamma2/network-snapshot-008733.pkl \
--resume-augment-pipe=True \
--resolution=256 --gamma=2 \
--batch=32 --batch-gpu=4 --tick=10 --snap=2 \
--outdir=/mnt/proj2/open-26-1/out/skygan256_2 \
--metrics=none \
--mirror=0 \
--aug-ada-xflip=0 \
--aug-ada-rotate90=0 \
--aug-ada-xint=0 \
--aug-ada-scale=0 \
--aug-ada-rotate=1 \
--aug-ada-aniso=0 \
--aug-ada-xfrac=0 \
--normalize-azimuth=True \
--use-encoder=True
