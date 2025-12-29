#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate spot
python twirler.py "$1"