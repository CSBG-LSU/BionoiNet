"""
Combine 2 folders of pickle files into one folder of pickle files
"""
import pickle
import argparse
import os
from os import listdir
from os.path import isfile, join

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-op',
                        default='control_vs_heme',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help= 'control_vs_heme, control_vs_nucleotide, heme_vs_nucleotide')
    parser.add_argument('-mol_pickle_dir',
                        default='../analyse/mols_pkl/',
                        required=False,
                        help='root folder of original pickles')
    parser.add_argument('-map_pickle_dir',
                        default='../analyse/map_pickle/',
                        required=False,
                        help='root folder of original pickles')
    parser.add_argument('-combine_pickle_dir',
                        default='../analyse/combine_pickle/',
                        required=False,
                        help='output folder containing results for combining with files in mols_pkl.')
    return parser.parse_args()

def loadPickle(pkl):
    """
    pkl: input pickle file path
    """
    f = open(pkl, 'rb')
    dict = pickle.load(f)
    f.close()
    return dict

def combineDict(molName, molDict, mapDicts, dirs, combineTargetFolder):
    """
    Combine a .mol2 dictionary with 6 corresponding saliency
    map dictionaries.

    molDict             ----- .mol2 dictionary
    mapDicts            ----- a tuple containing 6 dictionaries of 6 directions
    dirs                ----- a tuple containing 6 directions
    combineTargetFolder ----- target folder to save the new pickle file
    """
    dict = {}
    i = 0
    for dir in dirs:
        # merge 2 dictionaries
        dict[dir] = {**molDict[dir], **mapDicts[i]}
        i = i+1

    # dump dict into a new pickle file
    pklFile = open(combineTargetFolder+molName+'.pickle','wb')
    pickle.dump(dict, pklFile)


def combineFolder(molSourceFolder, mapSourceFolder, combineTargetFolder):
    """
    process one .mol2 pickle folder and its corresponding image pickle folder
    """
    if not os.path.exists(combineTargetFolder):
        os.makedirs(combineTargetFolder)

    molPickleList = [f for f in listdir(molSourceFolder) if isfile(join(molSourceFolder, f))]
    mapPickleList = [f for f in listdir(mapSourceFolder) if isfile(join(mapSourceFolder, f))]
    print('processing {} of files in folder {}'.format(len(molPickleList), molSourceFolder))
    print('processing {} of files in folder {}'.format(len(mapPickleList), mapSourceFolder))
    print('target folder for combined pickle files:', combineTargetFolder)

    # Convert map pickle list to dictionary with file names as keys.
    # Using dictionary hashing is the fastest indexing method for this task
    # as I'm aware of.
    mapPickleDict = dict(zip(mapPickleList,mapPickleList))

    dirs = ('xoy_+', 'xoy_-', 'yoz_+', 'yoz_-', 'zox_+', 'zox_-')

    # loop through each pickle file
    for molPickle in molPickleList:
        # load .mol2 pickle
        # print(molSourceFolder+molPickle)
        molDict = loadPickle(molSourceFolder+molPickle)

        # remove suffix
        molName = molPickle.split('.')[0]

        # find 6 corresponding map pickles
        keys = (molName+'xoy_+.pickle',molName+'xoy_-.pickle',
                molName+'yoz_+.pickle',molName+'yoz_-.pickle',
                molName+'zox_+.pickle',molName+'zox_-.pickle')
        '''
        hashedMapPickles = (mapPickleDict[keys[0]],mapPickleDict[keys[1]],
                            mapPickleDict[keys[2]],mapPickleDict[keys[3]],
                            mapPickleDict[keys[4]],mapPickleDict[keys[5]])
        '''
        # a tuple of dictionaries
        hashedMapDicts = (loadPickle(mapSourceFolder+mapPickleDict[keys[0]]),
                          loadPickle(mapSourceFolder+mapPickleDict[keys[1]]),
                          loadPickle(mapSourceFolder+mapPickleDict[keys[2]]),
                          loadPickle(mapSourceFolder+mapPickleDict[keys[3]]),
                          loadPickle(mapSourceFolder+mapPickleDict[keys[4]]),
                          loadPickle(mapSourceFolder+mapPickleDict[keys[5]]))

        # combine the dictionaries
        combineDict(molName, molDict, hashedMapDicts, dirs, combineTargetFolder)

if __name__ == "__main__":
    tasks = ('train', 'val', 'test')
    args = get_args()
    op = args.op
    mol_pickle_dir = args.mol_pickle_dir
    map_pickle_dir = args.map_pickle_dir
    combine_pickle_dir = args.combine_pickle_dir
    print('Combining pickles for ', op)
    if op == 'control_vs_heme':
        classes = ('control','heme')
    elif op == 'control_vs_nucleotide':
        classes = ('control','nucleotide')
    elif op == 'heme_vs_nucleotide':
        classes = ('heme', 'nucleotide')

    for task in tasks:
        for data_class in classes:
            molSourceFolder = mol_pickle_dir + op + '/' + task + '/' + data_class + '/'
            mapSourceFolder = map_pickle_dir + op + '/' + task + '/' + data_class + '/'
            combineTargetFolder = combine_pickle_dir + op + '/' + task + '/' + data_class + '/'
            combineFolder(molSourceFolder, mapSourceFolder, combineTargetFolder)
