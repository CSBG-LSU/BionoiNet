'''
Generating files for finial analysis.
mol_2_info --- Take source folder's .mol2 files, then generate 2 target folders, one for the pickle files
containing 'vor' and 'atoms' objects, one for the images.
'''
import sys
import os
import argparse
import pickle
import skimage
from bionoi import Bionoi
from skimage.io import imshow
import numpy as np
from os import listdir
from os.path import isfile, join

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-dpi',
                        default=256,
                        required=False,
                        help='image quality in dpi')
    parser.add_argument('-size', default=256,
                        required=False,
                        help='image size in pixels, eg: 128')
    parser.add_argument('-alpha',
                        default=0.5,
                        required=False,
                        help='alpha for color of cells')
    parser.add_argument('-colorby',
                        default="residue_type",
                        choices=["atom_type", "residue_type", "residue_num"],
                        required=False,
                        help='color the voronoi cells according to {atom_type, residue_type, residue_num}')
    parser.add_argument('-imageType',
                        default=".jpg",
                        choices=[".jpg", ".png"],
                        required=False,
                        help='the type of image {.jpg, .png}')
    return parser.parse_args()

def mol_2_info(mol, sourceFolder, targetFolderPkl, targetFolderImg, args):
    """
    mol2File: a path of a sinle .mol2 file as input
    args: input arguments
    targetFolderPkl: path to pickle file of a dictionary containing vor objects and
    atoms objects of the .mol2 file from 6 different directions. The dirctions are already predefined.
    targetFolderImg: path to save the images of 6 different directions
    """
    dpi = args.dpi
    size = args.size
    alpha = args.alpha
    colorby = args.colorby
    imageType = args.imageType

    if not os.path.exists(targetFolderPkl):
        os.makedirs(targetFolderPkl)
    if not os.path.exists(targetFolderImg):
        os.makedirs(targetFolderImg)

    # path of the source .mol2 file
    molPath = sourceFolder + mol
    #print(molPath)

    # remove file extension
    molName = mol.split('.')[0]

    # the dictionary containing 6 sub-dictionaries for 6 dirctions:
    # xoy_p, xoy_n, yoz_p, yoz_n, zox_p, zox_n
    dict ={}
    # the directions tuple
    dirs = ('xoy_+', 'xoy_-', 'yoz_+', 'yoz_-', 'zox_+', 'zox_-')
    i = 0
    for dir in dirs:
        atoms, vor, image = Bionoi(mol=molPath,bs_out='',size=size,dpi=dpi,alpha=alpha,colorby=colorby,proj_direction=i+1)
        dict[dirs[i]] = {'atoms':atoms, 'vor':vor}
        #save the images
        skimage.io.imsave(targetFolderImg+molName+dir+imageType, image)
        # increment direction index
        i = i + 1
        #imshow(image)
    #print('dict:',dict)

    # dump the dictionary into a pickle file
    pklFile = open(targetFolderPkl+molName+'.pickle','wb')
    pickle.dump(dict, pklFile)


def mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args):
    """
    Run mol_2_info() on entire folder
    """
    fileList = [f for f in listdir(sourceFolder) if isfile(join(sourceFolder, f))]
    print('converting {} of files in folder {}'.format(len(fileList),sourceFolder))
    print('pickle files saved in folder {}'.format(targetFolderPkl))
    print('images saved in folder {}'.format(targetFolderImg))
    for mol in fileList:
        mol_2_info(mol, sourceFolder, targetFolderPkl, targetFolderImg, args)
        #print('processed:', mol)

def control_vs_heme_gen(args):
    """
    Generate files for control_vs_heme task.
    """
    sourceFolder = '../control_vs_heme_mols/train/control/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_heme/train/control/'
    targetFolderImg = '../analyse/images/control_vs_heme/train/control/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_heme_mols/train/heme/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_heme/train/heme/'
    targetFolderImg = '../analyse/images/control_vs_heme/train/heme/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_heme_mols/val/control/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_heme/val/control/'
    targetFolderImg = '../analyse/images/control_vs_heme/val/control/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_heme_mols/val/heme/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_heme/val/heme/'
    targetFolderImg = '../analyse/images/control_vs_heme/val/heme/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_heme_mols/test/control/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_heme/test/control/'
    targetFolderImg = '../analyse/images/control_vs_heme/test/control/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_heme_mols/test/heme/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_heme/test/heme/'
    targetFolderImg = '../analyse/images/control_vs_heme/test/heme/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)

def control_vs_nucleotide_gen(args):
    """
    Generate files for control_vs_nucleotide task.
    """
    sourceFolder = '../control_vs_nucleotide_mols/train/control/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_nucleotide/train/control/'
    targetFolderImg = '../analyse/images/control_vs_nucleotide/train/control/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_nucleotide_mols/train/nucleotide/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_nucleotide/train/nucleotide/'
    targetFolderImg = '../analyse/images/control_vs_nucleotide/train/nucleotide/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_nucleotide_mols/val/control/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_nucleotide/val/control/'
    targetFolderImg = '../analyse/images/control_vs_nucleotide/val/control/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_nucleotide_mols/val/nucleotide/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_nucleotide/val/nucleotide/'
    targetFolderImg = '../analyse/images/control_vs_nucleotide/val/nucleotide/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_nucleotide_mols/test/control/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_nucleotide/test/control/'
    targetFolderImg = '../analyse/images/control_vs_nucleotide/test/control/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../control_vs_nucleotide_mols/test/nucleotide/'
    targetFolderPkl = '../analyse/mols_pkl/control_vs_nucleotide/test/nucleotide/'
    targetFolderImg = '../analyse/images/control_vs_nucleotide/test/nucleotide/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)

def heme_vs_nucleotide_gen(args):
    """
    Generate files for heme_vs_nucleotide task.
    """
    sourceFolder = '../heme_vs_nucleotide_mols/train/0-heme/'
    targetFolderPkl = '../analyse/mols_pkl/heme_vs_nucleotide/train/0-heme/'
    targetFolderImg = '../analyse/images/heme_vs_nucleotide/train/0-heme/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../heme_vs_nucleotide_mols/train/1-nucleotide/'
    targetFolderPkl = '../analyse/mols_pkl/heme_vs_nucleotide/train/1-nucleotide/'
    targetFolderImg = '../analyse/images/heme_vs_nucleotide/train/1-nucleotide/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../heme_vs_nucleotide_mols/val/0-heme/'
    targetFolderPkl = '../analyse/mols_pkl/heme_vs_nucleotide/val/0-heme/'
    targetFolderImg = '../analyse/images/heme_vs_nucleotide/val/0-heme/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../heme_vs_nucleotide_mols/val/1-nucleotide/'
    targetFolderPkl = '../analyse/mols_pkl/heme_vs_nucleotide/val/1-nucleotide/'
    targetFolderImg = '../analyse/images/heme_vs_nucleotide/val/1-nucleotide/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../heme_vs_nucleotide_mols/test/0-heme/'
    targetFolderPkl = '../analyse/mols_pkl/heme_vs_nucleotide/test/0-heme/'
    targetFolderImg = '../analyse/images/heme_vs_nucleotide/test/0-heme/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
    sourceFolder = '../heme_vs_nucleotide_mols/test/1-nucleotide/'
    targetFolderPkl = '../analyse/mols_pkl/heme_vs_nucleotide/test/1-nucleotide/'
    targetFolderImg = '../analyse/images/heme_vs_nucleotide/test/1-nucleotide/'
    mol_2_info_folder(sourceFolder, targetFolderPkl, targetFolderImg, args)
#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    #control_vs_heme_gen(args)
    #control_vs_nucleotide_gen(args)
    heme_vs_nucleotide_gen(args)