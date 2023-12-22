#!/bin/bash

. prepare_env.inc.sh

pkl_dir='/mnt/c/Users/CGG/Downloads'

set -e # stop on error
set -x # print executed commands

mkdir -p results

for pkl_name in \
    k00134t_Baseline_FID25.4@28.5M_network-snapshot-003665 \
    k00136t_Aug_FID20.8@29.2M_network-snapshot-002203 \
    k00133t_Ours_FID14.6@28.9M_network-snapshot-002343 \
    k00135t_Clear_FID21.5@26.5M_network-snapshot-000641 \
    ; do
#pkl_name='k00133t_Ours_FID14.6@28.9M_network-snapshot-002343'
    echo ',t,sun_elevation,img_fname' > results/$pkl_name.csv

    outdir="$pkl_name"

    #elevations="-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,40,45,50,55,60,65"

    elevtions="$(seq -5 65 | tr '\n' ',' | sed 's/,$//')"
    
    python gen_images.py \
        --network "$pkl_dir"/"$pkl_name".pkl \
        --normalize-azimuth=True --seeds='elevations+1000' --outdir=results/"$outdir" --azimuth=180 --elevations=$elevations

    (
        IFS=,; for el in $elevations; do
        #for el in 25; do
            #for seed in {0001..0005}; do
                seed=$(printf '%04d\n' $(expr $el + 1000))
                echo ",1970-01-01 00:00:00,$el,$outdir/fake_seed$seed.png" >> "results/$pkl_name.csv"
                seed2=$(printf '%04d\n' $(expr $el + 2000))
                echo ",1970-01-01 00:00:00,$el,$outdir/fake_seed$seed2.png" >> "results/$pkl_name.csv"
            #done
        done
    )
done

tar czvf results.tar.gz results/*.csv results/*/fake_seed*.png
