'''
Take the .mol2 files and generate 48 images for each of them.
'''
import argparse
from bionoi import Bionoi
import os
import skimage
from skimage.io import imshow
from skimage.transform import rotate
import numpy as np
import shutil
from os import listdir
from os.path import isfile, join
from shutil import copyfile


def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-opMode',
                        default='control_vs_heme',
                        required=False,
                        help='control_vs_heme, control_vs_nucleotide, control_vs_heme_cv, control_vs_nucleotide_cv')
    parser.add_argument('-proDirect',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        required=False,
                        help='the direction of projection(0 from all, xy+,xy-,yz+,yz-,zx+,zx-)')
    parser.add_argument('-rotAngle2D',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4],
                        required=False,
                        help='the angle of rotation(0:(0,90,180,270), 1:0 2:90 3:180 4:270)')
    parser.add_argument('-flip',
                        type=int,
                        default=0,
                        choices=[0, 1, 2],
                        required=False,
                        help='the type of flipping(0:(original and up-down), 1:original, 2:up-down')
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
    parser.add_argument('-save_fig',
                        default=False,
                        choices=[True, False],
                        required=False,
                        help='whether the original image needs save (True, False)')
    return parser.parse_args()


def gen_output_filename_list(dirct, rotAngle, flip):
    f_p_list = []
    f_r_list = []
    f_f_list = []

    if dirct != 0:
        name = ''
        if dirct == 1:
            name = 'XOY+'
        elif dirct == 2:
            name = 'XOY-'
        elif dirct == 3:
            name = 'YOZ+'
        elif dirct == 4:
            name = 'YOZ-'
        elif dirct == 5:
            name = 'ZOX+'
        elif dirct == 6:
            name = 'ZOX-'
        f_p_list.append(name)
    elif dirct == 0:
        f_p_list = ['XOY+', 'XOY-', 'YOZ+', 'YOZ-', 'ZOX+', 'ZOX-']

    if rotAngle != 0:
        name = ''
        if rotAngle == 1:
            name = '_r0'
        elif rotAngle == 2:
            name = '_r90'
        elif rotAngle == 3:
            name = '_r180'
        elif rotAngle == 4:
            name = '_r270'
        f_r_list.append(name)
    else:
        f_r_list = ['_r0', '_r90', '_r180', '_r270']

    if flip != 0:
        name = ''
        if flip == 1:
            name = '_OO'
        elif flip == 2:
            name = '_ud'

        f_f_list.append(name)
    else:
        f_f_list = ['_OO', '_ud']

    return f_p_list, f_r_list, f_f_list

# generate 48 images for one .mol2 file
def one_gen_48(mol, out_folder, args):

    basepath = os.path.basename(mol)
    basename = os.path.splitext(basepath)[0]
    size = args.size
    dpi = args.dpi
    alpha = args.alpha
    imgtype = args.imageType
    colorby = args.colorby
    proDirect = args.proDirect
    rotAngle = args.rotAngle2D
    flip = args.flip

    f_p_list, f_r_list, f_f_list = gen_output_filename_list(proDirect, rotAngle, flip)
    len_list = len(f_p_list)
    proj_img_list = []
    rotate_img_list = []
    flip_img_list = []

    # ===================================== Projection ===============================================
    if proDirect != 0:
        atoms, vor, img = Bionoi(mol=mol,
                                 bs_out='',
                                 size=size,
                                 dpi=dpi,
                                 alpha=alpha,
                                 colorby=colorby,
                                 proj_direction=proDirect)
        # imshow(img)
        proj_img_list.append(img)
    else:
        for i in range(len_list):
            atoms, vor, img = Bionoi(mol=mol,
                                     bs_out='',
                                     size=size,
                                     dpi=dpi,
                                     alpha=alpha,
                                     colorby=colorby,
                                     proj_direction=i+1)
            proj_img_list.append(img)
    # ---------------------------------- rotate -----------------------------------------

    col = proj_img_list
    m = len(col)

    for i in range(m):
        img = col[i]
        if rotAngle == 0:
            rotate_img_list.append(img)
            rotate_img_list.append(rotate(img, angle=90))
            rotate_img_list.append(rotate(img, angle=180))
            rotate_img_list.append(rotate(img, angle=270))
        elif rotAngle == 1:
            rotate_img_list.append(rotate(img, angle=0))
        elif rotAngle == 2:
            rotate_img_list.append(rotate(img, angle=90))
        elif rotAngle == 3:
            rotate_img_list.append(rotate(img, angle=180))
        elif rotAngle == 4:
            rotate_img_list.append(rotate(img, angle=270))
    # ---------------------------------- flip  -----------------------------------------

    for i in range(len(rotate_img_list)):
        img = rotate_img_list[i]
        if flip == 0:
            flip_img_list.append(img)
            flip_img_list.append(np.flipud(img))
        if flip == 1:
            flip_img_list.append(img)
        if flip == 2:
            img = np.flipud(img)
            flip_img_list.append(img)

    filename_list = []

    for i in range(len(f_p_list)):
        for j in range(len(f_r_list)):
            for k in range(len(f_f_list)):

                saveFile = f_p_list[i] + f_r_list[j] + f_f_list[k] + imgtype
                filename_list.append(saveFile)

    # assert len(filename_list) == len(flip_img_list)
    for i in range(len(filename_list)):
        path_base = os.path.join(out_folder, basename + '_')
        skimage.io.imsave(path_base + filename_list[i], flip_img_list[i])

    del filename_list
    del flip_img_list
    del rotate_img_list
    del proj_img_list
    del f_p_list, f_r_list, f_f_list

# generate 48 images based on .mol2 files in the sourceFolder
def gen_48(sourceFolder, targetFolder):
    if os.path.exists(targetFolder):
        shutil.rmtree(targetFolder, ignore_errors=True)
    os.makedirs(targetFolder)

    # a list of files in target folder
    fileList = [f for f in listdir(sourceFolder) if isfile(join(sourceFolder, f))]
    m = len(fileList)
    num = 0
    for mol in fileList:
        one_gen_48(sourceFolder+mol, targetFolder, args)
        num = num+1
    assert(m==num)

# generate images for control_vs_nucleotide
def gen_48_control_vs_nucleotide():
    gen_48(sourceFolder='../control_vs_nucleotide_mols/train/control/',targetFolder='../control_vs_nucleotide/train/control/')
    gen_48(sourceFolder='../control_vs_nucleotide_mols/train/nucleotide/',targetFolder='../control_vs_nucleotide/train/nucleotide/')
    gen_48(sourceFolder='../control_vs_nucleotide_mols/val/control/',targetFolder='../control_vs_nucleotide/val/control/')
    gen_48(sourceFolder='../control_vs_nucleotide_mols/val/nucleotide/',targetFolder='../control_vs_nucleotide/val/nucleotide/')
    gen_48(sourceFolder='../control_vs_nucleotide_mols/test/control/',targetFolder='../control_vs_nucleotide/test/control/')
    gen_48(sourceFolder='../control_vs_nucleotide_mols/test/nucleotide/',targetFolder='../control_vs_nucleotide/test/nucleotide/')

# generate images for control_vs_heme
def gen_48_control_vs_heme():
    gen_48(sourceFolder='../control_vs_heme_mols/train/control/',targetFolder='../control_vs_heme/train/control/')
    gen_48(sourceFolder='../control_vs_heme_mols/train/heme/',targetFolder='../control_vs_heme/train/heme/')
    gen_48(sourceFolder='../control_vs_heme_mols/val/control/',targetFolder='../control_vs_heme/val/control/')
    gen_48(sourceFolder='../control_vs_heme_mols/val/heme/',targetFolder='../control_vs_heme/val/heme/')
    gen_48(sourceFolder='../control_vs_heme_mols/test/control/',targetFolder='../control_vs_heme/test/control/')
    gen_48(sourceFolder='../control_vs_heme_mols/test/heme/',targetFolder='../control_vs_heme/test/heme/')

# generate images for control_vs_nucleotide for 10-fold cross-validation
def gen_48_control_vs_nucleotide_cv(k):
    for i in range(k):
        # from 0 to 9, folder from 1 to 10
        cvFoder = 'cv'+str(i+1)
        gen_48(sourceFolder='../control_vs_nucleotide_mols_cv/'+cvFoder+'/train/control/',targetFolder='../control_vs_nucleotide_cv/'+cvFoder+'/train/control/')
        gen_48(sourceFolder='../control_vs_nucleotide_mols_cv/'+cvFoder+'/train/nucleotide/',targetFolder='../control_vs_nucleotide_cv/'+cvFoder+'/train/nucleotide/')
        gen_48(sourceFolder='../control_vs_nucleotide_mols_cv/'+cvFoder+'/val/control/',targetFolder='../control_vs_nucleotide_cv/'+cvFoder+'/val/control/')
        gen_48(sourceFolder='../control_vs_nucleotide_mols_cv/'+cvFoder+'/val/nucleotide/',targetFolder='../control_vs_nucleotide_cv/'+cvFoder+'/val/nucleotide/')

# generate images for control_vs_nucleotide for 10-fold cross-validation
def gen_48_control_vs_heme_cv(k):
    for i in range(k):
        # from 0 to 9, folder from 1 to 10
        cvFoder = 'cv'+str(i+1)
        gen_48(sourceFolder='../control_vs_heme_mols_cv/'+cvFoder+'/train/control/',targetFolder='../control_vs_heme_cv/'+cvFoder+'/train/control/')
        gen_48(sourceFolder='../control_vs_heme_mols_cv/'+cvFoder+'/train/heme/',targetFolder='../control_vs_heme_cv/'+cvFoder+'/train/heme/')
        gen_48(sourceFolder='../control_vs_heme_mols_cv/'+cvFoder+'/val/control/',targetFolder='../control_vs_heme_cv/'+cvFoder+'/val/control/')
        gen_48(sourceFolder='../control_vs_heme_mols_cv/'+cvFoder+'/val/heme/',targetFolder='../control_vs_heme_cv/'+cvFoder+'/val/heme/')

# generate images for heme_vs_nucleotide
def gen_48_heme_vs_nucleotide():
    gen_48(sourceFolder='../heme_vs_nucleotide_mols/train/heme/',targetFolder='../heme_vs_nucleotide/train/heme/')
    gen_48(sourceFolder='../heme_vs_nucleotide_mols/train/nucleotide/',targetFolder='../heme_vs_nucleotide/train/nucleotide/')
    gen_48(sourceFolder='../heme_vs_nucleotide_mols/val/heme/',targetFolder='../heme_vs_nucleotide/val/heme/')
    gen_48(sourceFolder='../heme_vs_nucleotide_mols/val/nucleotide/',targetFolder='../heme_vs_nucleotide/val/nucleotide/')
    gen_48(sourceFolder='../heme_vs_nucleotide_mols/test/heme/',targetFolder='../heme_vs_nucleotide/test/heme/')
    gen_48(sourceFolder='../heme_vs_nucleotide_mols/test/nucleotide/',targetFolder='../heme_vs_nucleotide/test/nucleotide/')

# generate images for heme_vs_nucleotide for 10-fold cross-validation
def gen_48_heme_vs_nucleotide_cv(k):
    for i in range(k):
        # from 0 to 9, folder from 1 to 10
        cvFoder = 'cv'+str(i+1)
        gen_48(sourceFolder='../heme_vs_nucleotide_mols_cv/'+cvFoder+'/train/0-heme/',targetFolder='../heme_vs_nucleotide_cv/'+cvFoder+'/train/0-heme/')
        gen_48(sourceFolder='../heme_vs_nucleotide_mols_cv/'+cvFoder+'/train/1-nucleotide/',targetFolder='../heme_vs_nucleotide_cv/'+cvFoder+'/train/1-nucleotide/')
        gen_48(sourceFolder='../heme_vs_nucleotide_mols_cv/'+cvFoder+'/val/0-heme/',targetFolder='../heme_vs_nucleotide_cv/'+cvFoder+'/val/0-heme/')
        gen_48(sourceFolder='../heme_vs_nucleotide_mols_cv/'+cvFoder+'/val/1-nucleotide/',targetFolder='../heme_vs_nucleotide_cv/'+cvFoder+'/val/1-nucleotide/')

# generate images for bionoi autoencoder. All the images will be in one single folder
def gen_48_bionoi_autoencoder():
    gen_48(sourceFolder='../bae-data-mol2/',targetFolder='../bae-data-images/')

#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    args = getArgs()
    opMode = args.opMode

    if opMode == 'control_vs_nucleotide':
        gen_48_control_vs_nucleotide()
    elif opMode == 'control_vs_heme':
        gen_48_control_vs_heme()
    elif opMode == 'control_vs_nucleotide_cv':
        gen_48_control_vs_nucleotide_cv(10)
    elif opMode == 'control_vs_heme_cv':
        gen_48_control_vs_heme_cv(10)
    elif opMode == 'heme_vs_nucleotide':
        gen_48_heme_vs_nucleotide()
    elif opMode == 'heme_vs_nucleotide_cv':
        gen_48_heme_vs_nucleotide_cv(10)
    elif opMode == 'bionoi_autoencoder':
        gen_48_bionoi_autoencoder()
    else:
        print('error: invalid value of opMode.')