#!/bin/bash

# This script will create ssh keys on zizgpu0x and ssh-copy-id them into ziz, 
# so that you can easily ssh or rsync from zizgpu0x to ziz within the SLURM jobs.
# This script will prompt you to enter your statistics departmental password for 
# each of the 4 nodes zizgpu0x when executing ssh-copy-id.

# to execute from ziz
dir_path="`dirname \"$0\"`"
source $dir_path/config.sh

ssh-keygen -q -t rsa -b 2048 -N "" -f ~/.ssh/$rsa_key <<< y

for NODE_ID in '1' '2' '3' '5'
do
	srun  --clusters=srf_gpu_01 --partition=zizgpu0$NODE_ID-debug ssh-keygen -q -t rsa -b 2048 -N "" -f ~/.ssh/$rsa_key <<< y
	srun --pty --clusters=srf_gpu_01 --partition=zizgpu0$NODE_ID-debug ssh-copy-id -i ~/.ssh/$rsa_key ziz.stats.ox.ac.uk
	# srun --pty --clusters=srf_gpu_01 --partition=zizgpu0$NODE_ID-debug scp ziz.stats.ox.ac.uk:~/.ssh/$rsa_key.pub ~/.ssh/
	# srun --clusters=srf_gpu_01 --partition=zizgpu0$NODE_ID-debug cat ~/.ssh/$rsa_key.pub >> ~/.ssh/authorized_keys
done