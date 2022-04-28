#!/bin/bash

# to execute from zizgpu02

dir_path="`dirname \"$0\"`"
source $dir_path/config.sh

deactivate
rm -rf $parent_dir
mkdir $parent_dir

virtualenv -p python3.9 $parent_dir/$venv
source $parent_dir/$venv/bin/activate
pip install -r requirements.txt