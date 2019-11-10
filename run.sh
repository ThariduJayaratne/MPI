#!/bin/bash
module load languages/intel
make -B
sbatch stencil.job

