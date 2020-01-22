"""
Copy the scores of atoms and paste into the last column of a .pdb file of a pocket.
Note that order of atoms are different in the csv file containing scores.
"""
import os
import argparse
import csv
import pickle
from pickle import load, dump
import pandas as pd
from biopandas.pdb import PandasPdb
from shutil import copyfile
from os import listdir

def pocket_visual_single(pdb_dir, source_pdb, scores_dir, scores_source, out_dir):
    """
    take the source pdb file and its corresponding score, then add the scores 
    to the last column of the pdb file, save it to out_dir.
    """
    ppdb = PandasPdb()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    
    pdb_file = pdb_dir + source_pdb
    scores_file = scores_dir + scores_source
    out_file = out_dir + source_pdb

    # copy file to out_dir
    copyfile(pdb_file, out_file)

    # read the scores as a list
    df = pd.read_csv(scores_file)
    scores = df['total']
    row_nums = df.iloc[:,0]
    #print(scores)
    #print(row_nums) # row numbers in original pdb file

    # modify the copied pdb file
    pocket = ppdb.read_pdb(out_file)
    i = 0
    for row_num in row_nums:
        ppdb.df['ATOM']['b_factor'][row_num] = scores[i]
        #print(ppdb.df['ATOM']['b_factor'][row_num])
        i = i + 1
    #print(ppdb.df['ATOM']['b_factor'])
    ppdb.to_pdb(path = out_file)

def pocket_visual_folder(pdb_dir, scores_dir, out_dir, pocket_list):
    """
    Apply pocket_visual_folder() function to entire folders of pdb files, and
    entire folders of corresponding final scores
    """
    for f in pocket_list:
        f_pdb = f + '.pdb'
        f_score = f + '-1' + '.csv'
        #print(f_pdb)
        pocket_visual_single(pdb_dir, f_pdb, scores_dir, f_score, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',   
                        required = False,
                        default = 'control_vs_heme',
                        choices = ['control_vs_heme', 'control_vs_nucleotide'],
                        help='operation modes')
    parser.add_argument('-correct_predict_dir',   
                        required = False,
                        default = '../../bionoi_transfer_learning/log/',
                        help='final scores directory for heme')                            
    parser.add_argument('-final_scores_dir_heme',   
                        required = False,
                        default = '../../analyse/final_scores/control_vs_heme/test/heme/',
                        help='final scores directory for heme')
    parser.add_argument('-final_scores_dir_nucleotide',   
                        required = False,
                        default = '../../analyse/final_scores/control_vs_nucleotide/test/nucleotide/',
                        help='final scores directory for nucleotide')
    parser.add_argument('-pdb_dir_heme',   
                        required = False,
                        default = '../../analyse/pocket-lpc-heme/',
                        help='heme pdb directory')
    parser.add_argument('-pdb_dir_nucleotide',   
                        required = False,
                        default = '../../analyse/pocket-lpc-nucleotide/',
                        help='nucleotide pdb directory')                    
    parser.add_argument('-out_dir_heme',   
                        required = False,
                        default = '../../analyse/pdb_visual_heme/',
                        help='pdb output directory for heme')
    parser.add_argument('-out_dir_nucleotide',   
                        required = False,
                        default = '../../analyse/pdb_visual_nucleotide/',
                        help='pdb output directory for nucleotide')
    args = parser.parse_args()
    op = args.op
    correct_predict_dir = args.correct_predict_dir
    final_scores_dir_heme = args.final_scores_dir_heme
    final_scores_dir_nucleotide = args.final_scores_dir_nucleotide
    pdb_dir_heme = args.pdb_dir_heme
    pdb_dir_nucleotide = args.pdb_dir_nucleotide
    out_dir_heme = args.out_dir_heme
    out_dir_nucleotide = args.out_dir_nucleotide
    if op == 'control_vs_heme':
        final_scores_dir = final_scores_dir_heme
        pdb_dir = pdb_dir_heme
        out_dir = out_dir_heme
    elif op == 'control_vs_nucleotide':
        final_scores_dir = final_scores_dir_nucleotide
        pdb_dir = pdb_dir_nucleotide
        out_dir = out_dir_nucleotide

    # load the lists of correctly predicted pockets
    with open(correct_predict_dir + op + '_correct_preds_6_dirs_vote.pickle', 'rb') as f:
        dict = load(f)
    #print(dict)
    class_1_list = dict[1]
    pocket_list = []
    for pocket in class_1_list:
        pocket_list.append(pocket[0])
    #print(pocket_list)
    print('total number of correct predictions:', len(pocket_list))

    # modify the pdb file to visualize atoms' scores
    pocket_visual_folder(pdb_dir, final_scores_dir, out_dir, pocket_list)