#!/bin/bash

# to execute from ziz
dir_path="`dirname \"$0\"`"
source $dir_path/config.sh

for NODE_ID in '4' '3' '1'
do
    srun --partition=ziz-gpu0$NODE_ID-debug mkdir -p $parent_dir/$venv &
    srun --partition=ziz-gpu0$NODE_ID-debug rsync -e "ssh -i ~/.ssh/id_rsa_ziz" -a --info=progress2 ziz.stats.ox.ac.uk:$parent_dir/$venv/* $parent_dir/$venv/. &
done
