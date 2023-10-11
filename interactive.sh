#!/bin/bash
srun -p grete:interactive --pty -n 1 -c 32 -G V100:1 bash
