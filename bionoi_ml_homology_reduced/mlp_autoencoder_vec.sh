#!/bin/bash
#PBS -q workq
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -N bionoi_autoencoder_mlp
#PBS -A loni_omics01
#PBS -j oe

module purge
source activate pytorch
cd /work/wshi6/deeplearning-data/BionoiNet-prj/BionoiNet/bionoi_ml_homology_reduced/
#python mlp_autoencoder_vec.py -op control_vs_nucleotide -result_file_suffix 1th_run > ./mlp_autoencoder_vec_log/control_vs_nucleotide_mlp_autoencoder_1th_run.log 2>&1
python mlp_autoencoder_vec.py -op control_vs_heme -result_file_suffix 1th_run > ./mlp_autoencoder_vec_log/control_vs_heme_mlp_autoencoder_1th_run.log 2>&1






