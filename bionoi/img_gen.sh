#!/bin/bash
#PBS -q workq
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -N bionoi_residue_type
#PBS -A loni_omics01
#PBS -j oe

module purge
source activate pytorch
cd /work/wshi6/deeplearning-data/bionoi_prj/bionoi

colorby="residue_type"
mol_folder="../homology_reduced_mols"/*/*
pop_folder="../pop_folder/"
profile_folder="../profile_folder/"
output_folder="../bionoi_output/${colorby}/"


img_gen_folder(){
    local mol_folder=$1
    local pop_folder=$2
    local profile_folder=$3
    local output_folder=$4   
    local colorby=$5

    for mol_path in $mol_folder
    do

        # mol file
        mol=${mol_path##*/}
        echo $mol
        
        # protein
        #protein=${mol::-9}
        protein=${mol::${#mol}-9}

        # pop file
        pop="${protein}.out"

        # pop path
        pop_path="${pop_folder}${pop}"

        # profile file
        profile="${protein}.profile"

        # profile path
        profile_path="${profile_folder}${profile}"

        #echo "$mol_path"
        #echo $mol
        #echo $protein
        #echo $pop
        #echo $pop_path
        #echo $profile
        #echo $profile_path

        # execute
        python bionoi.py -mol "$mol_path" -pop "$pop_path" -profile "$profile_path" -out "$output_folder" -colorby "$colorby" -direction 0 -rot_angle 0 -flip 0 
    done
}


for input_dir in $mol_folder
do
    echo "colorby:"
    echo $colorby
    
    echo "input directory:"
    echo $input_dir
    input_files="$input_dir"/*
    
    IFS='/' read -r -a array <<< "$input_dir"
    output_dir="${output_folder}${array[2]}/${array[3]}/"
    echo "output directory:"
    echo $output_dir
    
    pop_dir="${pop_folder}${array[2]}/"
    echo "pop directory:"
    echo $pop_dir

    profile_dir="${profile_folder}${array[2]}/"
    echo "profile directory:"
    echo $profile_dir

    echo
    img_gen_folder "$input_files" "$pop_dir" "$profile_dir" "$output_dir" "$colorby"
done











