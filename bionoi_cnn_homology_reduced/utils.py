"""
Utility functions and modules to build a convolutional neural network (CNN) for the reduced homolgy dataset for binding site classification.
"""

import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class BionoiTrainDataset(Dataset):
    """
    Training dataset for bionoi.
    """
    def __init__(self, op, root_dir, folds, transform=None):
        """
        Args:
            op: operation mode, heme_vs_nucleotide, control_vs_heme or control_vs_nucleotide. 
            root_dir: folder containing all images.
            folds: a list containing folds to generate training data, for example [1,2,3,4].
            transform: transform to be applied to images.
        """
        self.op = op
        self.root_dir = root_dir
        self.folds = folds
        self.transform = transform
        self.pocket_classes = self.op.split('_vs_') # two classes of binding sites
        assert(len(self.pocket_classes)==2)

        # directory of folders containing training images
        self.folder_dirs = [] 
        for pocket_class in self.pocket_classes:
            for fold in self.folds:
                self.folder_dirs.append(self.root_dir + pocket_class + '/fold_' + str(fold))
        print(self.folder_dirs)
        
        # file names of the folders, [[filenames for folder 1], [filenames for folder 2], ...]
        self.list_of_files_list = []
        for folder_dir in self.folder_dirs:
            self.list_of_files_list.append([os.path.join(folder_dir, name) for name in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, name))])
        print(self.file_names)

        # lengths of the folders
        self.folder_dirs_lengths = []
        for files_list in self.list_of_files_list:
            self.folder_dirs_lengths.append(len(files_list))
        print(self.folder_dirs_lengths)

    def __len__(self):
        return sum(self.folder_dirs_lengths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

    def __locate_file(self, idx):
        """
        Function to locate the directory of file
        """
        low = 0
        up = 0
        for i in range(len(self.folder_dirs_lengths)):
            up += self.folder_dirs_lengths[i]
            if idx >= low and idx <up:
                return i, sub_idx
            low = up  

class BionoiValDataset(Dataset):
    """
    Validation dataset for bionoi
    """
    def __init__(self, op, root_dir, fold, transform=None):
        """
        Args:
            op: operation mode, heme_vs_nucleotide, control_vs_heme or control_vs_nucleotide. 
            root_dir: folder containing all images.
            fold: int, to represent validation fold, for example 5.
            transform: transform to be applied to images.
        """
        self.op = op
        self.root_dir = root_dir
        self.folds = folds
        self.transform = transform

if __name__ == "__main__":
    op = 'control_vs_nucleotide'
    root_dir = '../../data_homology_reduced/images/'
    folds = [1,2,3,4]
    training_set = BionoiTrainDataset(op=op, root_dir=root_dir, folds=folds)