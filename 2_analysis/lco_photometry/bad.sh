#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate spot
python badfiles.py