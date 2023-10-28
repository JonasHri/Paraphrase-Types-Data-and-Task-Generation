#!/bin/bash
srun -p grete:shared --pty -n 1 -c 32 -G V100:1 bash
module load anaconda3
module load nvitop
conda activate nlp