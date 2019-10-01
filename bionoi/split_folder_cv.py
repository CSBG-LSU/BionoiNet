"""
Split the data in a k-fold cross validation manner for binary classification.
"""
import numpy as np
import os
import sys
import argparse
from os import listdir
from os.path import isfile, join
from shutil import copyfile
# save the splitted datasets to folders
def saveToFolder(fileCollection, sourceFolder, targetFolder):
    m = len(fileCollection)
    for i in range(m):
        source = sourceFolder + '/' + fileCollection[i]
        target = targetFolder + '/' + fileCollection[i]
        copyfile(source, target)

def split2CVFolder(k, sourceFolder, targetFolder, type, type_num):
    """
    type: 'cats/', 'dogs/' or 'control'
    """
    sourceFolder = sourceFolder + type

    # a list of files in target folder
    fileList = [f for f in listdir(sourceFolder) if isfile(join(sourceFolder, f))]
    m = len(fileList)

    # shuffle the collection to create a new one
    idx = np.arange(m)
    idx_permu = np.random.permutation(idx)
    colShuffle = [fileList[i] for i in idx_permu]

    # calculate the length of each new folder
    lenVal = int(m/k)
    print('lenth of val data:', lenVal)
    lenTrain = m - lenVal
    print('length of train data:',lenTrain)

    # put cats and dogs in class 1 and control in class 0
    type = type_num + '-' + type

    # split the collection in a k-fold cross validation manner
    for i in range (k):
        print('i:',i)
        colVal = colShuffle[i*lenVal:(i+1)*lenVal]
        print('val idx:', i*lenVal,' to ', (i+1)*lenVal-1)
        colTrain = colShuffle[0:i*lenVal] + colShuffle[(i+1)*lenVal:m]
        print('length of train:',len(colTrain))
        print('length of val:',len(colVal))

        # create sub-directory in targetFolder for current fold
        trainDir = targetFolder+'/cv'+str(i+1)+'/train/'+type
        if not os.path.exists(trainDir):
            os.makedirs(trainDir)
        valDir = targetFolder+'/cv'+str(i+1)+'/val/'+type
        if not os.path.exists(valDir):
            os.makedirs(valDir)

        # copy files to corresponding new folder
        saveToFolder(colTrain, sourceFolder, trainDir)
        saveToFolder(colVal, sourceFolder, valDir)
        print('-------------------------------------')

if __name__ == "__main__":
    k = 10
    src_dir = '../mols_extract/'

    #dst_dir = './cats_vs_control_cv/'
    #dst_dir = './dogs_vs_control_cv/'
    dst_dir = '../heme_vs_nucleotide_mols_cv/'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    #target_type = 'cats/'
    #target_type = 'dogs/'

    #split2CVFolder(k,src_dir,dst_dir,'control/')
    #split2CVFolder(k,src_dir,dst_dir,target_type)
    split2CVFolder(k, src_dir, dst_dir, 'heme/', '0')
    split2CVFolder(k, src_dir, dst_dir, 'nucleotide/', '1')