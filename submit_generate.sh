#!/bin/bash
#SBATCH -p grete:shared              # the partition
#SBATCH -G V100:2                    # For requesting 1 GPU.
#SBATCH -c 40                        # Requestion 8 CPU cores.

module load anaconda3
source activate nlp

python generation.py