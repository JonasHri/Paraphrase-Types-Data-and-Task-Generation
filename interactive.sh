#!/bin/bash
srun --pty -p grete:interactive  -G 2g.10gb:2 /bin/bash
module load anaconda3
module load nvitop
# conda activate nlp