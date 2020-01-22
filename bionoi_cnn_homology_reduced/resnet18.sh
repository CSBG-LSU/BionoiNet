#!/bin/bash
#PBS -q workq
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -N seq_entropy
#PBS -A loni_omics01
#PBS -j oe

module purge
source activate pytorch
cd /work/wshi6/deeplearning-data/bionoi_prj/BionoiNet/bionoi_cnn_homology_reduced/
#python homology_reduced_cnn_cv_resnet18.py -op control_vs_heme -root_dir ../../bionoi_output/seq_entropy/ -result_file_suffix seq_entropy > ./log/control_vs_heme_res18_seq_entropy.log 2>&1
python homology_reduced_cnn_cv_resnet18.py -op control_vs_nucleotide -root_dir ../../bionoi_output/seq_entropy/ -result_file_suffix seq_entropy > ./log/control_vs_nucleotide_res18_seq_entropy.log 2>&1






