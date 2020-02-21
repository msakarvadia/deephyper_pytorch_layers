#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q debug-cache-quad
#COBALT -A datascience
#COBALT --jobname pytorch-deephyper

module load miniconda-3/2019-11

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export KMP_HW_SUBSET=1s,${OMP_NUM_THREADS}c,2t
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

env > ${COBALT_JOBID}.env
cp $0 ${COBALT_JOBID}.sh

#export MKLDNN_VERBOSE=1
#export MKL_VERBOSE=1
#aprun -n 1 -N 1 -d 64 -j 2 python -m deephyper.search.hps.ambs --problem problem.py --run model_run.py --n-jobs=1

cd /projects/datascience/parton/deephyper/deephyper_pytorch_layers

aprun -n 1 -N 1 --cc none python -m deephyper.search.hps.ambs --problem problem.py --run model_run.py --n-jobs=1 --evaluator=subprocess

mv results.csv ${COBALT_JOBID}.csv
