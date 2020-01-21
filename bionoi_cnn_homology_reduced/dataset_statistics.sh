#!/bin/bash
#PBS -q workq
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -N bionoi_statistics
#PBS -A loni_omics01
#PBS -j oe

module purge
source activate pytorch
cd /work/wshi6/deeplearning-data/bionoi_prj/BionoiNet/bionoi_cnn_homology_reduced/
python dataset_statistics.py  > ./dataset_statistics.log 2>&1






