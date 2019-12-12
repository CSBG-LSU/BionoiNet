#!/bin/bash
#PBS -q k40
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -N bionoi_resnet18
#PBS -A loni_omics01
#PBS -j oe

module purge
source activate pytorch
cd /work/wshi6/deeplearning-data/BionoiNet-prj/BionoiNet/bionoi_cnn_homology_reduced/
python homology_reduced_cnn_cv_resnet18.py -op control_vs_nucleotide -batch_size 16 -result_file_suffix 13thrun > ./log/control_vs_nucleotide_res18_13th_run.log 2>&1





