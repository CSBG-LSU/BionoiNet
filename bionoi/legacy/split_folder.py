"""
Split a folder into train, val and test for classification
"""
import os
import numpy as np
from shutil import copyfile
from os import listdir
import argparse

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-src_dir',
                        default='../mols_extract/',
                        required=False,
                        help='directory containing folders of images')

    parser.add_argument('-dst_dir',
                        default='../heme_vs_nucleotide_mols/',
                        required=False,
                        help='destination directory')

    parser.add_argument('-target_name',
                        default='heme',
                        required=False,
                        help='directory containing folders of images')

    parser.add_argument('-target_num',
                        default='0',
                        required=False,
                        help='number of the target class')

    parser.add_argument('-val_frac',
                        type=float,
                        default=0.1,
                        required=False,
                        help='the fraction of validation data')
    return parser.parse_args()

# save the splitted datasets to folders
def save_to_folder(fileCollection, sourceFolder, targetFolder):
    m = len(fileCollection)
    for i in range(m):
        source = sourceFolder + '/' + fileCollection[i]
        target = targetFolder + '/' + fileCollection[i]
        copyfile(source, target)

def split_bc(src_dir, dst_dir, val_frac, target_name='cats', target_num='0'):
    src_target = src_dir + target_name

    # a list of files in target class
    list_target = [f for f in listdir(src_target)]
    num_target = len(list_target)
    # shuffle the collection to create a new one
    idx_target = np.arange(num_target)
    permu_target = np.random.permutation(idx_target)
    shuffled_target = [list_target[i] for i in permu_target]

    # calculate the length of each new folder of target
    len_target_test = int(num_target*val_frac)
    len_target_val = len_target_test
    print('lenth of val data of {}:{}'.format(target_name, len_target_val))
    print('lenth of test data of {}:{}'.format(target_name, len_target_test))
    len_target_train = num_target - (len_target_test + len_target_val)
    print('length of train data of {}:{}'.format(target_name, len_target_train))

    # create 2 new lists of images of target
    target_train_set = shuffled_target[0:len_target_train]
    target_val_set = shuffled_target[len_target_train : len_target_train + len_target_val]
    target_test_set = shuffled_target[len_target_train + len_target_val : num_target]

    # create sub-directory in dst_dir of nucleotide
    target_train_dir = dst_dir + 'train/' + target_num + '-' + target_name + '/'
    if not os.path.exists(target_train_dir):
        os.makedirs(target_train_dir)
    target_val_dir = dst_dir + 'val/' + target_num + '-' + target_name + '/'
    if not os.path.exists(target_val_dir):
        os.makedirs(target_val_dir)
    target_test_dir = dst_dir + 'test/' + target_num + '-' + target_name + '/'
    if not os.path.exists(target_test_dir):
        os.makedirs(target_test_dir)

    # copy files to corresponding new folder
    save_to_folder(target_train_set, src_target, target_train_dir)
    save_to_folder(target_val_set, src_target, target_val_dir)    
    save_to_folder(target_test_set, src_target, target_test_dir)

if __name__ == "__main__":
    args = get_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    target_name = args.target_name
    target_num = args.target_num
    val_frac = args.val_frac

    split_bc(src_dir, dst_dir, val_frac, target_name, target_num)