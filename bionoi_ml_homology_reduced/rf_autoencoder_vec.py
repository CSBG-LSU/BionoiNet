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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',
                        default='control_vs_nucleotide',
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
    data = np.empty([m, 512]) # allocate memory for feature matrix
    labels = np.empty([m]) # allocate memory for labels
    since = time.time()
    print('loading data...')
    for i in range(m):
        #data[i] = dataset[i] # put each element of dataset as a row in the data matrix
        row, label = dataset[i]
        data[i] = row
        labels[i] = label
        #print('row:',row)
        #print('label:',label)
        #print(data)
        #print(labels)
        #break 
    time_elapsed = time.time() - since
    print( 'Loading data takes {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return data, labels


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


def report_avg_metrics(val_metrics_over_folds):
    """
    Report the averaged metrics over 5 folds when valication loss reaches minimum 
    """
    avg_acc = 0
    avg_precision = 0
    avg_recall = 0
    avg_mcc = 0 
    for i in range(5):
        dict = val_metrics_over_folds[i]
        avg_acc += dict['acc']
        avg_precision += dict['precision']
        avg_recall += dict['recall']
        avg_mcc += dict['mcc']
    avg_acc = avg_acc/5
    avg_precision = avg_precision /5
    avg_recall = avg_recall/5
    avg_mcc = avg_mcc/5
    print('average accuracy: ', avg_acc)
    print('average precision: ', avg_precision)
    print('average recall: ', avg_recall)
    print('average mcc: ', avg_mcc)


if __name__ == "__main__":
    args = get_args()
    op = args.op    
    root_dir = args.root_dir
    result_file_suffix = args.result_file_suffix
    print('data directory:', root_dir)
    num_control = args.num_control
    num_heme = args.num_heme
    num_nucleotide = args.num_nucleotide

    if op == 'control_vs_heme':
        classes = ('control','heme')
        class_weight = {0:22944, 1:74784}
    elif op == 'control_vs_nucleotide':
        classes = ('control','nucleotide')
        class_weight = {0:59664, 1:74784}
    elif op == 'heme_vs_nucleotide':
        classes = ('0-heme','1-nucleotide')
        class_weight = {0:59664, 1:22944}

    val_metrics_over_folds=[]
    val_roc_over_folds=[]
    
    for i in range(5):
        print('*********************************************************************')
        print('starting {}th fold cross-validation'.format(i+1))
        folds = [1, 2, 3, 4, 5]
        val_fold = i+1
        folds.remove(val_fold)
        
        train_set = BionoiDatasetCV(op, root_dir, folds)
        val_set = BionoiDatasetCV(op, root_dir, [val_fold])
        #print(training_set[0])
        X_train, y_train = load_data_to_memory(train_set)
        X_val, y_val = load_data_to_memory(val_set)
        
        # random forest model
        rf = RandomForestClassifier(n_estimators=1000,
                                    max_depth=None,
                                    min_samples_split = 0.001,
                                    min_samples_leaf = 0.001,
                                    random_state=42,
                                    max_features = "auto",
                                    criterion = "gini",
                                    class_weight = class_weight,
                                    n_jobs = -1)

        # train the rf on the training data
        print('training the random forest...')
        rf.fit(X_train, y_train)
        print('training finished.')

        # training performance
        train_acc = rf.score(X_train, y_train)
        print('training accuracy:', train_acc)

        # validation performance
        predictions = rf.predict(X_val)
        num_correct = np.sum(predictions == y_val)
        print('number of correct validation predictions:',num_correct)
        
        val_acc = rf.score(X_val, y_val)
        print('validation accuracy:', val_acc)

        val_precision = precision_score(y_val,predictions)      
        print('validation precision:', val_precision)

        val_recall = recall_score(y_val,predictions)       
        print('validation recall:', val_recall)

        val_mcc = matthews_corrcoef(y_val, predictions)
        print('validation mcc:', val_mcc)

        val_f1 = f1_score(y_val, predictions)
        print('validation f1:', val_f1)        
        
        val_cm = confusion_matrix(y_val, predictions)
        print('validation confusion matrix:', val_cm)

        val_prob = rf.predict_proba(X_val) # output probabilities for val data
        fpr, tpr, thresholds = roc_curve(y_val, val_prob[:, 1])
        fpr = fpr.tolist()
        tpr = tpr.tolist()
        thresholds = thresholds.tolist()
        val_roc = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds}
        val_metrics = {'acc': val_acc, 'precision': val_precision, 'recall': val_recall, 'mcc': val_mcc}
        val_metrics_over_folds.append(val_metrics)
        val_roc_over_folds.append(val_roc)

    print('--------------------------------------------------')
    report_avg_metrics(val_metrics_over_folds)

    dict_to_save = {'metrics':val_metrics_over_folds, 'roc':val_roc_over_folds}
    result_file = './rf_autoencoder_vec_result/' + op + '_cv_' + result_file_suffix + '.json'
    with open(result_file, 'w') as fp:
        json.dump(dict_to_save, fp)