"""
A convolutional neural network (CNN) for the reduced homolgy dataset for binding site classification.
"""
import argparse
import os
import numpy as np
import pickle
import copy
import time
import sklearn.metrics as metrics
import json

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',
                        default='heme_vs_nucleotide',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help="'control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'")
    parser.add_argument('-root_dir',
                        default='../../data_homology_reduced/autoencoder_vectors/',
                        required=False,
                        help='directory to load data for 5-fold cross-validation.')   
    parser.add_argument('-result_file_suffix',
                        required=True,
                        help='suffix to result file')                        
    parser.add_argument('-num_control',
                        default=74784,
                        required=False,
                        help='number of control data points, used to calculate the positive weight for the loss function')
    parser.add_argument('-num_heme',
                        default=22944,
                        required=False,
                        help='number of heme data points, used to calculate the positive weight for the loss function.')
    parser.add_argument('-num_nucleotide',
                        default=59664,
                        required=False,
                        help='number of num_nucleotide data points, used to calculate the positive weight for the loss function.')
    return parser.parse_args()


"""
Funcitons and modules to generate dataloaders for training and validation
"""
class BionoiDatasetCV():
    """
    Dataset for bionoi, cam be used to load multiple folds for training or single fold for validation and testing
    """
    def __init__(self, op, root_dir, folds):
        """
        Args:
            op: operation mode, heme_vs_nucleotide, control_vs_heme or control_vs_nucleotide. 
            root_dir: folder containing all images.
            folds: a list containing folds to generate training data, for example [1,2,3,4], [5].
            transform: transform to be applied to images.
        """
        self.op = op
        self.root_dir = root_dir
        self.folds = folds
        print('--------------------------------------------------------')
        if len(folds) == 1:
            print('generating validation dataset...')
        else:
            print('generating training dataset...')
        print('folds: ', self.folds)
        
        self.pocket_classes = self.op.split('_vs_') # two classes of binding sites
        assert(len(self.pocket_classes)==2)
        self.class_name_to_int = {}
        for i in range(len(self.pocket_classes)):
            self.class_name_to_int[self.pocket_classes[i]] = i
        print('class name to integer map: ', self.class_name_to_int)

        self.folder_dirs = [] # directory of folders containing training images                                 
        self.folder_classes = [] # list of classes of each folder, integer        
        for pocket_class in self.pocket_classes:
            for fold in self.folds:
                self.folder_classes.append(self.class_name_to_int[pocket_class])
                self.folder_dirs.append(self.root_dir + pocket_class + '/fold_' + str(fold))
        print('getting training data from following folders: ', self.folder_dirs)
        print('folder classes: ', self.folder_classes)

        self.list_of_files_list = [] # directory of files for all folds, [[files for fold 1], [files for fold 2], ...]
        for folder_dir in self.folder_dirs:
            self.list_of_files_list.append([os.path.join(folder_dir, name) for name in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, name))])
        #print(self.list_of_files_list)

        self.folder_dirs_lengths = [] # lengths of the folders
        for files_list in self.list_of_files_list:
            self.folder_dirs_lengths.append(len(files_list))
        print('lengths of the folders: ', self.folder_dirs_lengths)

    def __len__(self):
        return sum(self.folder_dirs_lengths)

    def __getitem__(self, idx):
        """
        Built-in function to retrieve item(s) in the dataset.
        Args:
            idx: index of the image
        """
        # get directory of file
        folder_idx, sub_idx = self.__locate_file(idx)
        vec_dir = self.list_of_files_list[folder_idx][sub_idx]
        
        # read autoencoder vector
        vector = self.__load_pkl(vec_dir)

        # get label 
        label = self.__get_class_int(folder_idx)

        return vector, label

    def __locate_file(self, idx):
        """
        Function to locate the directory of file
        Args:
            idx: an integer which is the index of the file.
        """
        low = 0
        up = 0
        for i in range(len(self.folder_dirs_lengths)):
            up += self.folder_dirs_lengths[i]
            if idx >= low and idx <up:
                sub_idx = idx - low
                #print('folder:', i)
                #print('sub_idx:', sub_idx)
                #print('low:', low)
                #print('up', up)
                return i, sub_idx
            low = up  

    def __get_class_int(self, folder_idx):
        """
        Function to get the label of an image as integer
        """
        return self.folder_classes[folder_idx]


    def __load_pkl(self, pkl):
        """
        load and unpack the pickle file.
        """
        f = open(pkl, 'rb')
        dict = pickle.load(f)
        f.close()
        ft = np.array(dict['X'], dtype='f')
        return ft


def load_data_to_memory(dataset):
    """
    Load the dataset into memory in the form of numpy array.
    """
    m = len(dataset) # number of data
    print('number of data: ', m)
    data = np.empty([m, 512])# allocate memory
    print('loading data...')
    for i in range(m):
        print(dataset[i])
        break 

def calc_metrics(label, out):
    """
    Both label and out should be numpy arrays containing 0s and 1s.
    This is used for training/validating/testing.
    """
    acc = metrics.accuracy_score(label, out)
    precision = metrics.precision_score(label,out)
    recall = metrics.recall_score(label,out)
    f1 = metrics.f1_score(label,out)
    mcc = metrics.matthews_corrcoef(label, out)
    return acc, precision, recall, f1, mcc


def report_avg_metrics(best_val_loss_metrics_over_folds):
    """
    Report the averaged metrics over 5 folds when valication loss reaches minimum 
    """
    avg_loss = 0
    avg_acc = 0
    avg_precision = 0
    avg_recall = 0
    avg_mcc = 0 
    for i in range(5):
        dict = best_val_loss_metrics_over_folds[i]
        avg_loss += dict['loss']
        avg_acc += dict['acc']
        avg_precision += dict['precision']
        avg_recall += dict['recall']
        avg_mcc += dict['mcc']
    avg_loss = avg_loss/5
    avg_acc = avg_acc/5
    avg_precision = avg_precision /5
    avg_recall = avg_recall/5
    avg_mcc = avg_mcc/5
    print('average loss: ', avg_loss)
    print('average accuracy: ', avg_acc)
    print('average precision: ', avg_precision)
    print('average recall: ', avg_recall)
    print('average mcc: ', avg_mcc)


if __name__ == "__main__":
    args = get_args()
    op = args.op    
    root_dir = args.root_dir
    result_file_suffix = args.result_file_suffix
    batch_size = args.batch_size
    print('data directory:', root_dir)
    print('batch size: '+str(batch_size))
    num_control = args.num_control
    num_heme = args.num_heme
    num_nucleotide = args.num_nucleotide

    # calculate positive weight for loss function
    if op == 'control_vs_heme':
        print('performing control_vs_heme cross-validation task.')
        loss_fn_pos_weight = num_control/num_heme
    elif op == 'control_vs_nucleotide':
        print('performing control_vs_nucleotide cross-validation task.')
        loss_fn_pos_weight = num_control/num_nucleotide
    elif op == 'heme_vs_nucleotide':
        print('performing heme_vs_nucleotide cross-validation task.')
        loss_fn_pos_weight = num_heme/num_nucleotide 

    for i in range(5):
        print('*********************************************************************')
        print('starting {}th fold cross-validation'.format(i+1))
        folds = [1, 2, 3, 4, 5]
        val_fold = i+1
        folds.remove(val_fold)
        
        training_set = BionoiDatasetCV(op, root_dir, folds)
        val_set = BionoiDatasetCV(op, root_dir, val_fold)
        #print(training_set[0])
        load_data_to_memory(val_set)
        break


    """
    # lists of the results of 5 folds, each element of the list is a list containing the
    # corresponding result of 1 fold
    history_over_folds=[]
    best_val_loss_metrics_over_folds=[]
    best_val_loss_roc_over_folds=[]

    # perfrom 5-fold cross-validation
    for i in range(5):
        print('*********************************************************************')
        print('starting {}th fold cross-validation'.format(i+1))
        folds = [1, 2, 3, 4, 5]
        val_fold = i+1
        folds.remove(val_fold)


        # import the model and training configurations
        if op == 'control_vs_heme':
            net, loss_fn, optimizer_dict, num_epochs = control_vs_heme_config(device)
        elif op == 'control_vs_nucleotide':
            net, loss_fn, optimizer_dict, num_epochs = control_vs_nucleotide_config(device)
        elif op == 'heme_vs_nucleotide':
            net, loss_fn, optimizer_dict, num_epochs = heme_vs_nucleotide_config(device)
        
        
        # train the neural network
        trained_model, best_val_loss_roc, best_val_loss_metrics, history = train_model(net,
                                                        device,
                                                        dataloaders_dict,
                                                        loss_fn,
                                                        optimizer_dict,
                                                        loss_fn_pos_weight = loss_fn_pos_weight,
                                                        num_epochs = num_epochs
                                                        )

        # put the histories into the lists
        best_val_loss_roc_over_folds.append(best_val_loss_roc)
        best_val_loss_metrics_over_folds.append(best_val_loss_metrics)
        history_over_folds.append(history)
        
        break
    #----------------------------------end of training---------------------------------------

    
    dict_to_save = {'loss':best_val_loss_metrics_over_folds, 'roc':best_val_loss_roc_over_folds, 'history':history_over_folds}
    result_file = './mlp_autoencoder_vec_result/' + op + '_cv_' + result_file_suffix + '.json'
    with open(result_file, 'w') as fp:
        json.dump(dict_to_save, fp)
    
    # print results of the folds when validation loss reaches minimum.    
    print('------------------------------------')
    print('metrics for each fold when validation loss reaches minimum:')
    print(best_val_loss_metrics_over_folds)
    print('Averge metrics over 5 folds:')
    report_avg_metrics(best_val_loss_metrics_over_folds)
    print('cross-validation finished, end of program')
    """