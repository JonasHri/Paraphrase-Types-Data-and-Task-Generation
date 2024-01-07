#!/bin/bash
#SBATCH --time 01:00:00
#SBATCH --job-name jupyter-notebook

#SBATCH -p grete:interactive         # the partition
#SBATCH -G V100:1                    # For requesting GPU.
#SBATCH -c 40                        # Requestion CPU cores.


module load anaconda3
module load nvitop

source activate nlp

# run jupyter notebook
jupyter notebook --ip localhost --port 3001 --no-browser