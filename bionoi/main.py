import argparse
from bionoi import Bionoi
import os
import skimage
from skimage.io import imshow

from skimage.transform import rotate as skrotate
import numpy as np


def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-mol',
                        default="./examples/4v94E.mol2",
                        required=False,
                        help='the protein/ligand mol2 file')
    parser.add_argument('-out',
                        default="./output/",
                        required=False,
                        help='the folder of output images file')
    parser.add_argument('-dpi',
                        default=256,
                        required=False,
                        help='image quality in dpi')
    parser.add_argument('-size', default=256,
                        required=False,
                        help='image size in pixels, eg: 128')
    parser.add_argument('-alpha',
                        default=1.0,
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
    parser.add_argument('-direction',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        required=False,
                        help='The direction of projection. Input the index [0-6] for direction(s) of interest \
                             in this tuple: (ALL, xy+, xy-, yz+, yz-, zx+, zx-)')
    parser.add_argument('-rot_angle',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3, 4],
                        required=False,
                        help='The angle of rotation. Input the index [0-4] for direction(s) of interest \
                              in this tuple: (ALL, 0, 90, 180, 270)')
    parser.add_argument('-flip',
                        type=int,
                        default=0,
                        choices=[0, 1, 2, 3],
                        required=False,
                        help='The type of flipping. Input the index [0-3] for flip of interest \
                              in this tuple: (ALL, original, up-down, left-right)')
    parser.add_argument('-save_fig',
                        default=False,
                        choices=[True, False],
                        required=False,
                        help='Whether or not the original image needs saving.')
    return parser.parse_args()


def rotate(proj_img_list, rotation_angle):
    rotate_img_list = []
    for img in proj_img_list:
        if rotation_angle == 0:

            rotate_img_list.append(img)
            rotate_img_list.append(skrotate(img, angle=90))
            rotate_img_list.append(skrotate(img, angle=180))
            rotate_img_list.append(skrotate(img, angle=270))
        elif rotation_angle == 1:
            rotate_img_list.append(skrotate(img, angle=0))
        elif rotation_angle == 2:
            rotate_img_list.append(skrotate(img, angle=90))
        elif rotation_angle == 3:
            rotate_img_list.append(skrotate(img, angle=180))
        elif rotation_angle == 4:
            rotate_img_list.append(skrotate(img, angle=270))
    return rotate_img_list


def flip(rotate_img_list, flip):
    flip_img_list = []
    for img in rotate_img_list:
        if flip == 0:
            flip_img_list.append(img)
            flip_img_list.append(np.flipud(img))
            flip_img_list.append(np.fliplr(img))
        if flip == 1:
            flip_img_list.append(img)
        if flip == 2:
            img = np.flipud(img)
            flip_img_list.append(img)
        if flip == 3:
            img = np.fliplr(img)
            flip_img_list.append(img)

    return flip_img_list


def gen_output_filenames(direction, rotation_angle, flip):
    proj_names = []
    rot_names = []
    flip_names = []

    if direction != 0:
        name = ''
        if direction == 1:
            name = 'XOY+'
        elif direction == 2:
            name = 'XOY-'
        elif direction == 3:
            name = 'YOZ+'
        elif direction == 4:
            name = 'YOZ-'
        elif direction == 5:
            name = 'ZOX+'
        elif direction == 6:
            name = 'ZOX-'
        proj_names.append(name)
    elif direction == 0:
        proj_names = ['XOY+', 'XOY-', 'YOZ+', 'YOZ-', 'ZOX+', 'ZOX-']

    if rotation_angle != 0:
        name = ''
        if rotation_angle == 1:
            name = '_r0'
        elif rotation_angle == 2:
            name = '_r90'
        elif rotation_angle == 3:
            name = '_r180'
        elif rotation_angle == 4:
            name = '_r270'
        rot_names.append(name)
    else:
        rot_names = ['_r0', '_r90', '_r180', '_r270']

    if flip != 0:
        name = ''
        if flip == 1:
            name = '_OO'
        elif flip == 2:
            name = '_ud'
        elif flip == 3:
            name = '_lr'
        flip_names.append(name)
    else:
        flip_names = ['_OO', '_ud', '_lr']

    return proj_names, rot_names, flip_names


if __name__ == "__main__":

    args = getArgs()

    mol = args.mol
    out_folder = args.out
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Alias args
    size = args.size
    dpi = args.dpi
    alpha = args.alpha
    imgtype = args.imageType
    colorby = args.colorby
    proj_direction = args.direction
    rotation_angle = args.rot_angle
    flip_type = args.flip

    # 
    proj_names, rot_names, flip_names = gen_output_filenames(proj_direction, rotation_angle, flip_type)
    len_list = len(proj_names)
    proj_img_list = []

    # Project
    for i, proj_name in enumerate(proj_names):
        atoms, vor, img = Bionoi(mol=mol,
                                 bs_out=out_folder + proj_name,
                                 size=size,
                                 dpi=dpi,
                                 alpha=alpha,
                                 colorby=colorby,
                                 proj_direction=i + 1 if proj_direction==0 else proj_direction)
        proj_img_list.append(img)

    # Rotate
    rotate_img_list = rotate(proj_img_list, rotation_angle)

    # Flip
    flip_img_list = flip(rotate_img_list, flip_type)

    assert len(proj_img_list) == len(proj_names)
    assert len(rotate_img_list) == len(rot_names)*len(proj_names)
    assert len(flip_img_list) == len(flip_names)*len(rot_names)*len(proj_names)

    # Setup output folder 
    filenames = []
    for file in list(os.listdir(out_folder)):
        if os.path.isfile(file):
            os.remove(file)

    # Make file names 
    for pname in proj_names:
        for rname in rot_names:
            for fname in flip_names:
                saveFile = os.path.join(out_folder, pname + rname + fname + imgtype)
                filenames.append(saveFile)

    assert len(filenames) == len(flip_img_list)
    
    for i in range(len(filenames)):
        imshow(flip_img_list[i])
        skimage.io.imsave(filenames[i], flip_img_list[i])
