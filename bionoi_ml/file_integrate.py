"""
Load the files and combine the features into one matrix, labels into one vector.
"""
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join

def load_pkl(pkl):
    """
    load and unpack the pickle file.
    """
    f = open(pkl, 'rb')
    dict = pickle.load(f)
    f.close()
    ft = np.array(dict['X'])
    return ft

def file_integrate(data_dir, label):
    """
    integrate files for classification
    data_dir: folders containing .pikle files. Each file contains a feature matrix and a label
    combined_matrix: each row is a flattened feature matrix.
    combined_labels: a vector contains all the labels.
    """
    # get the list of file names
    print('integrating files of ', data_dir)
    file_list = [f for f in listdir(data_dir)]

    # size of list
    num_files = len(file_list)
    print('number of files:', num_files)

    # size of feature vector
    pickle_0 = data_dir + file_list[0]
    ft_0 = load_pkl(pickle_0)
    feature_size = ft_0.shape
    feature_size = feature_size[0]
    print('size of feature vector:', feature_size)
    
    # feature matrix
    combined_matrix = np.empty(shape=(num_files, feature_size))
    print('size of feature matrix:',combined_matrix.shape)

    # label vector
    combined_labels = np.empty(shape=(num_files))
    print('size of label vector:',combined_labels.shape)

    i = 0
    for f in file_list:
        # load the file and unpack
        pickle_file = data_dir + f
        ft = load_pkl(pickle_file)

        # flatten the feature matrix and append to the list
        #print('size of flattened feature vector:', matrix.flatten().shape)
        combined_matrix[i,:]  = ft.flatten()

        # append the label to the label list
        #print('size of label vector:', label.shape)
        combined_labels[i] = label

        i = i + 1
    #---------------------------------------------------------------------------
    # enf of for loop

    # test the correctness of the code
    '''
    i = 0
    for f in file_list:
        # load the file and unpack
        pickle_file = data_dir + f
        ft = load_pkl(pickle_file)
        assert((combined_matrix[i,:] == ft.flatten()).all())      
        assert(combined_labels[i] == label)
        i = i + 1
    '''
    #---------------------------------------------------------------------------
    # end of correctness testing
    return combined_matrix, combined_labels, num_files

if __name__ == "__main__":
    ft = load_pkl('../control_vs_nucleotide_ft_cv/cv1/val/nucleotide/2j1lA00-1_ZOX-_r180_ud.pickle')
    print(*ft, sep='\n')