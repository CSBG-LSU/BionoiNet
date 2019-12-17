#!/bin/bash
#PBS -q workq
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -N bionoi_autoencoder_rf
#PBS -A loni_omics01
#PBS -j oe

module purge
source activate pytorch
cd /work/wshi6/deeplearning-data/BionoiNet-prj/BionoiNet/bionoi_ml_homology_reduced/
#python rf_autoencoder_vec.py -op control_vs_heme -result_file_suffix 7th_run > ./rf_autoencoder_vec_log/control_vs_heme_rf_autoencoder_7th_run.log 2>&1
python rf_autoencoder_vec.py -op control_vs_nucleotide -result_file_suffix 7th_run > ./rf_autoencoder_vec_log/control_vs_nucleotide_rf_autoencoder_7th_run.log 2>&1






