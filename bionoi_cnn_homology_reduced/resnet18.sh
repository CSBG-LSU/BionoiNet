#!/bin/bash
#PBS -q workq
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -N bionoi_resnet18
#PBS -A loni_omics01
#PBS -j oe

module purge
source activate pytorch
cd /work/wshi6/deeplearning-data/BionoiNet-prj/BionoiNet/bionoi_cnn_homology_reduced/
python homology_reduced_cnn_cv_resnet18.py -op control_vs_heme -batch_size 16 -result_file_suffix 12thrun > ./log/control_vs_heme_res18_12th_run.log 2>&1
#python homology_reduced_cnn_cv_resnet18.py -op heme_vs_nucleotide -batch_size 32 -result_file_suffix 1thrun > ./log/heme_vs_nucleotide_res18_1th_run.log 2>&1






