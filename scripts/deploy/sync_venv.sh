#!/bin/bash

# to execute from ziz
dir_path="`dirname \"$0\"`"
source $dir_path/config.sh

for NODE_ID in '1' '2' '3' '5'
do
    srun --clusters=srf_gpu_01 --partition=zizgpu0$NODE_ID-debug mkdir -p $parent_dir/$venv &
    srun --clusters=srf_gpu_01 --partition=zizgpu0$NODE_ID-debug rsync -e "ssh -i ~/.ssh/id_rsa_ziz" -a --info=progress2 ziz.stats.ox.ac.uk:$parent_dir/$venv/* $parent_dir/$venv/. &
done
