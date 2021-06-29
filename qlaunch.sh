#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q debug-cache-quad
#COBALT -A datascience
#COBALT --jobname pytorch-deephyper

module load miniconda-3/2019-11

# all of these environment variables are again set within the *_run.py. AND:
# os.environ['MKLDNN_VERBOSE'] = str(1)
# os.environ['MKL_VERBOSE'] = str(1)

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

# help="number of cores to use for the 'learner', if n_jobs=-1 then it will use all cores available.")
aprun -n 1 -N 1 --cc none python -m deephyper.search.hps.ambs --problem problem.py --run model_run.py --n-jobs=1 --evaluator=subprocess

# recall, -n vs. -N convention is the OPPOSITE of Slurm's convention. Cray aprun documentation:
# [‐n | ‐‐pes width ]
# [‐N | ‐‐pes‐per‐node pes_per_node ]

mv results.csv ${COBALT_JOBID}.csv
