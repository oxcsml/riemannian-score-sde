#!/bin/bash

# to execute from ziz
dir_path="`dirname \"$0\"`"
source $dir_path/config.sh

for NODE_ID in '4' '3' '1'
do
    srun --pty --partition=ziz-gpu0$NODE_ID-debug ssh-copy-id -i ~/.ssh/$rsa_key zizgpu02.cpu.stats.ox.ac.uk
done