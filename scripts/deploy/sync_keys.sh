#!/bin/bash

# to execute from ziz
dir_path="`dirname \"$0\"`"
source $dir_path/config.sh

for NODE_ID in '2' '3' '1' '5'
do
    srun --pty --partition=zizgpu0$NODE_ID-debug ssh-copy-id -i ~/.ssh/$rsa_key zizgpu04.cpu.stats.ox.ac.uk
done