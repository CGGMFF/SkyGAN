#!/bin/sh

export UID
export GID

#ocker-compose run gpu-jupyter python3 work/JanSpacek-thesis-code/stylegan/main.py run_128debug
#docker-compose run stylegan3 python train.py --data /projects/SkyGAN/skygan-docker/src/JanSpacek-thesis-code/stylegan/light_cloud_cover.csv --cfg=stylegan3-t --gpus=2 --batch=32 --gamma=32 --batch-gpu=16 --tick 1 --snap=5 --outdir=/local/stylegan3-encoder

docker-compose up stylegan3
