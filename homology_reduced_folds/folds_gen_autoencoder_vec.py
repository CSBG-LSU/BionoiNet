"""
Generate a folder containing .mol2 files arranged as the provided folds files
"""
import os
import glob
from shutil import copy2


def get_folds(fold_dir):
    """
    Get kth line of the file fold_dir and parse this line and return a list of .mol2 file names.
    """
    fp = open(fold_dir)
    lines = fp.readlines()
    #print(lines)
    num_folds = len(lines) # each line contains files of a fold
    print('there are {} folds for {}'.format(num_folds, fold_dir))
    folds = []
    for i in range(num_folds):
        line = lines[i]
        line = line[0:-1] # remove the '\n' at the end of string
        line = line.split(' ')
        #line = [file + '-1.mol2' for file in line]
        #print(line)
        folds.append(line)
    return folds


def copy_selected_files(src, dst, files):
    """
    Copy a list of files from src to dst.
    """
    src_list = []
    for file in files:
        src_list.extend(glob.glob(src + file + '*-1*.pickle'))
    #print(src_list)
    for file in src_list:
        copy2(file, dst)    


if __name__ == "__main__":
    src_root = '../../data/autoencoder_encoded_vectors/'
    dst_root = '../../data_homology_reduced/autoencoder_vectors/'
    control_folds = './control-folds'
    nucleotide_folds = './atp-0.2-folds'
    heme_folds = './heme-0.2-folds'
    control_files = get_folds(control_folds)
    nucleotide_files = get_folds(nucleotide_folds)
    heme_files = get_folds(heme_folds)
    files_dict = {'control': control_files, 'nucleotide': nucleotide_files, 'heme':heme_files}
    #print(heme_files)
    pockets_folds = [control_folds, nucleotide_folds, heme_folds]
    pockets = ['control', 'nucleotide', 'heme']
    num_folds = 5

    for pocket in pockets:
        src_dir = src_root + pocket + '/'
        file_list = files_dict[pocket]
        for i in range(num_folds):
            # create folder for this fold
            dst_dir = dst_root + pocket + '/' + 'fold_' + str(i+1) + '/'
            try:
                os.makedirs(dst_dir)
            except OSError:
                print ("Creation of the directory %s failed" % dst_dir)
            else:
                print ("Successfully created the directory %s" % dst_dir)

            # copy corresponding files to folder      
            copy_selected_files(src_dir, dst_dir, file_list[i])     

            
            
