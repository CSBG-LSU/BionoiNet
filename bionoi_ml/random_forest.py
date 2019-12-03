"""
Run random forest on the bionoi data, performing 10-fold cross validation,
plot the ROC curve
"""
import argparse
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix
from file_integrate import file_integrate

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

    parser.add_argument('-src_dir_control_vs_heme_cv',
                        default='../../bionoi_autoencoder_prj/autoencoder_vec_control_vs_heme/',
                        required=False,
                        help='directory to load data for cross validation.')

    parser.add_argument('-src_dir_control_vs_nucleotide_cv',
                        default='../../bionoi_autoencoder_prj/autoencoder_vec_control_vs_nucleotide/',
                        required=False,
                        help='directory to load data for cross validation.')

    parser.add_argument('-src_dir_heme_vs_nucleotide_cv',
                        default='../../bionoi_autoencoder_prj/ft_vec_heme_vs_nucleotide_cv/',
                        required=False,
                        help='directory to load data for cross validation.')

    parser.add_argument('-log_dir',
                        default='./log/',
                        required=False,
                        help='directory to save data for cross validation.')

    return parser.parse_args()


def dataset_cv(data_dir, classes, k):
    """
    Return train and validation data, loaded in RAM.
    data_dir --- root of the data directory
    classes  --- names of the two classes
    k        --- which fold of data to use
    """
    # load class 0 for training
    data_dir_sub = data_dir + 'cv' + str(k) + '/' + 'train/' + classes[0] + '/'
    X_train_0, y_train_0, num_train_0 = file_integrate(data_dir_sub, 0)
    print('size of training data of {} class:{}'.format(classes[0], X_train_0.shape))
    assert(y_train_0.shape[0]==num_train_0)
    print('number of training data points of {}:{}'.format(classes[0],num_train_0))

    # load class 1 for training
    data_dir_sub = data_dir + 'cv' + str(k) + '/' + 'train/' + classes[1] + '/'
    X_train_1, y_train_1, num_train_1 = file_integrate(data_dir_sub, 1)
    print('size of training data of {} class:{}'.format(classes[1], X_train_1.shape))
    assert(y_train_1.shape[0]==num_train_1)
    print('number of training data points of {}:{}'.format(classes[1],num_train_1))

    # load class 0 for validation
    data_dir_sub = data_dir + 'cv' + str(k) + '/' + 'val/' + classes[0] + '/'
    X_val_0, y_val_0, num_val_0 = file_integrate(data_dir_sub, 0)
    print('size of val data of {} class:{}'.format(classes[0], X_val_0.shape))
    assert(y_val_0.shape[0]==num_val_0)
    print('number of val data points of {}:{}'.format(classes[0],num_val_0))

    # load class 1 for validation
    data_dir_sub = data_dir + 'cv' + str(k) + '/' + 'val/' + classes[1] + '/'
    X_val_1, y_val_1, num_val_1 = file_integrate(data_dir_sub, 1)
    print('size of val data of {} class:{}'.format(classes[1], X_val_1.shape))
    assert(y_val_1.shape[0]==num_val_1)
    print('number of val data points of {}:{}'.format(classes[1],num_val_1))

    # concatenate data from class 0 and class 1
    X_train = np.vstack((X_train_0, X_train_1)) # vertically
    y_train = np.hstack((y_train_0, y_train_1)) # horizontally
    X_val = np.vstack((X_val_0, X_val_1)) # vertically
    y_val = np.hstack((y_val_0, y_val_1)) # horizontally
    print('shape of X_train:', X_train.shape)
    print('shape of y_train:', y_train.shape)
    print('shape of X_val:', X_val.shape)
    print('shape of y_val:', y_val.shape)

    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    args = get_args()
    op = args.op
    src_dir_control_vs_heme_cv = args.src_dir_control_vs_heme_cv
    src_dir_control_vs_nucleotide_cv = args.src_dir_control_vs_nucleotide_cv
    src_dir_heme_vs_nucleotide_cv = args.src_dir_heme_vs_nucleotide_cv
    log_dir = args.log_dir
    if op == 'control_vs_heme':
        data_dir = src_dir_control_vs_heme_cv
        classes = ('control','heme')
        class_weight = {0:22944, 1:74784}
    elif op == 'control_vs_nucleotide':
        data_dir = src_dir_control_vs_nucleotide_cv
        classes = ('control','nucleotide')
        class_weight = {0:59664, 1:74784}
    elif op == 'heme_vs_nucleotide':
        data_dir = src_dir_heme_vs_nucleotide_cv
        classes = ('0-heme','1-nucleotide')
        class_weight = {0:59664, 1:22944}

    # random forest model
    rf = RandomForestClassifier(n_estimators=500,
                                max_depth=None,
                                min_samples_split=200,
                                random_state=0,
                                max_features = "auto",
                                criterion = "gini",
                                class_weight = class_weight,
                                n_jobs = -1)

    # list contains dictionarys for each fold
    val_acc_records = []
    val_precision_records = []
    val_recall_records = []
    val_mcc_records = []
    fpr_records = []
    tpr_records = []
    thresholds_records = []

    # k-fold cross-validation
    for k in range(10):
        print('----------------------------------------------')
        print('{}th fold cross-validation'.format(k+1))
        # forming the dataset
        # previous memory space is release at this point
        X_train, y_train, X_val, y_val = dataset_cv(data_dir, classes, k+1)

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
        print('number of correct predictions:',num_correct)

        val_acc = rf.score(X_val, y_val)
        val_acc_records.append(val_acc)
        print('validation accuracy:', val_acc)

        val_precision = precision_score(y_val,predictions)
        val_precision_records.append(val_precision)        
        print('validation precision:', val_precision)

        val_recall = recall_score(y_val,predictions)
        val_recall_records.append(val_recall)        
        print('validation recall:', val_recall)

        val_mcc = matthews_corrcoef(y_val, predictions)
        val_mcc_records.append(val_mcc)
        print('validation mcc:', val_mcc)

        val_f1 = f1_score(y_val, predictions)
        print('validation f1:', val_f1)

        val_cm = confusion_matrix(y_val, predictions)
        print('validation confusion matrix:', val_cm)

        # output probabilities for val data
        val_prob = rf.predict_proba(X_val)
        fpr, tpr, thresholds = roc_curve(y_val, val_prob[:, 1])

        # convert to list so that can be saved in .json file
        fpr = fpr.tolist()
        tpr = tpr.tolist()
        thresholds = thresholds.tolist()
        fpr_records.append(fpr)
        tpr_records.append(tpr)
        thresholds_records.append(thresholds)
    #---------------------------end of for loop-------------------------------------
    records = {'val_acc_records':val_acc_records,
               'val_precision_records':val_precision_records,
               'val_recall_records':val_recall_records,    
               'val_mcc_records':val_mcc_records,
               'fpr_records':fpr_records,
               'tpr_records':tpr_records,
               'thresholds_records':thresholds_records}
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)               
    pkl_file = open(log_dir+'rf_'+op+'_cv'+'.pickle','wb') # save as pickle with the same file name
    pickle.dump(records, pkl_file) # save the pickle file