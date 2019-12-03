"""
Generate feature vectors of images for cross-validating using HOG algorithm. 
The vectors are used as structured data for machine learning classifiers. 
"""
import os
import pickle
import matplotlib.pyplot as plt
from os import listdir
from skimage.feature import hog
from skimage import data, exposure
from skimage.io import imread
import argparse

def get_args():
    """
    The parser function to set the default vaules of the program parameters or read the new value set by user.
    """    
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-op',
                        default='control_vs_heme',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help='control_vs_heme, control_vs_nucleotide, heme_vs_nucleotide')

    parser.add_argument('-src_dir',
                        default='../../bionois_data_prep/',
                        required=False,
                        help='directory to load data for cross validation.')

    parser.add_argument('-dst_dir',
                        default='../',
                        required=False,
                        help='directory to save data for cross validation.')

    return parser.parse_args()

def vec_gen(img):
    """
    take one input image and convert it to feature vector.
    img --- input image
    vec --- output vector, if input size is 256*256, the output size is 2048
    """
    vec = hog(img, 
              orientations=8, 
              pixels_per_cell=(16, 16),
              cells_per_block=(1, 1), 
              visualize=False, 
              multichannel=True)	
    return vec

def folder_vec_gen(src_dir, dst_dir):
    """
    convert entire folder of images to feature vector, and store 
    in a pickle file.
    src_dir ---- source image folder
    dst_dir ---- destination folder to store the h5 files
    """
    print('generateing feature vectors for images in {}, destination:{}'.format(src_dir, dst_dir))
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)    
    src_list = [f for f in listdir(src_dir)] # a list of files in src_dir
    for f in src_list:
        src_file = src_dir + f
        img = imread(src_file) # load image
        vec = vec_gen(img) # convert to hog vector
        mol_name = f.split('.')[0] # get rid of extension
        pkl_file = open(dst_dir + mol_name +'.pickle','wb') # save as pickle with the same file name
        dict = {'X':vec} # put vector in a dictionary
        pickle.dump(dict, pkl_file) # save the pickle file


if __name__ == "__main__":
    args = get_args()
    op = args.op
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    if op == 'control_vs_heme':
        src_dir = src_dir + 'control_vs_heme_cv' + '/'
        dst_dir = dst_dir + 'control_vs_heme_ft_cv' + '/'
        classes = ('control', 'heme')
    elif op == 'control_vs_nucleotide':
        src_dir = src_dir + 'control_vs_nucleotide_cv' + '/'
        dst_dir = dst_dir + 'control_vs_nucleotide_ft_cv' + '/'
        classes = ('control', 'nucleotide')
    elif op == 'heme_vs_nucleotide':
        src_dir = src_dir + 'heme_vs_nucleotide_cv' + '/'
        dst_dir = dst_dir + 'heme_vs_nucleotide_ft_cv' + '/'        
        classes = ('0-heme', '1-nucleotide')
    tasks = ('train', 'val')
    k = 10 # number of folds for cross validation
    print('source folder:', src_dir)
    print('destination folder:', dst_dir)

    for i in range(k):
        for task in tasks:
            for l in range(2): # two classes
                type = classes[l]
                print('generating {} files of {} for {}th fold'.format(task, type, i+1))
                src_dir_cv = src_dir + 'cv' + str(i+1) + '/' + task + '/' + type + '/'
                dst_dir_cv = dst_dir + 'cv' + str(i+1) + '/' + task + '/' + type + '/'
                folder_vec_gen(src_dir_cv, dst_dir_cv)

    #src_dir = '../mols_img/'
    #dst_dir = '../mols_hog_vec/'
    #classes = ('control','heme','nucleotide','steroid')
    #for i in range(4):
    #    folder_vec_gen(src_dir + classes[i] + '/', dst_dir + classes[i] + '/', i)

    #pkl = '../mols_hog_vec/control/154lA00-1_XOY+_r0_OO.pickle'
    #f = open(pkl, 'rb')
    #dict = pickle.load(f)
    #print(dict['X'].shape)
    #print(dict['y'])
    #f.close()

    #img_file = '../mols_img/steroid/1a28A00-1_XOY-_r0_OO.jpg'
    #img = imread(img_file)
    #print(img)  
    #vec = vec_gen(img)    
    #print(vec.shape)
