"""
Plot the mean ROC curve for 10-fold cross-validation of each algorithm.
Take the pickle files that contains results of cross-validation, then 
compute the final results and draw the mean ROC cure.
Display the ROC curves for both contro_vs_heme and control_vs_nucleotide
"""
import argparse
import pickle
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from statistics import mean 
def get_args():
    """
    The parser function to set the default vaules of the program parameters or read the new value set by user.
    """    
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-model',
                        default='rf',
                        required=False,
                        choices = ['rf', 'mlp', 'lr', 'mlp_img'],
                        help='random forest, multi-layer perceptron, logistic regression')

    parser.add_argument('-log_dir',
                        default='./log_autoencoder_mlp/',
                        required=False,
                        help='directory to save data for cross validation.')

    return parser.parse_args()

def mean_roc(pkl_file, color):
    """
    Plot the average ROC curve for 10-fold cross validation
    pik_file   ---- contains the dictionary of the validation metrics.
    indiv_plot ---- bool, whether to plot the ROC curve for each fold. 
    """
    # unpack data
    records = pickle.load(pkl_file)
    val_acc_records = records['val_acc_records']
    #val_precision_records = records['val_precision_records']
    #val_recall_records = records['val_recall_records']
    val_mcc_records = records['val_mcc_records']
    fpr_records = records['fpr_records']
    tpr_records = records['tpr_records']
    thresholds_records = records['thresholds_records']

    # mean accuracy and mean mcc
    print('mean validation accuracy:', mean(val_acc_records))
    #print('mean validation precision:', mean(val_precision_records))
    #print('mean validation recall:', mean(val_recall_records))
    print('mean validation MCC:', mean(val_mcc_records))

    # begin to plot, using code from 
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10): # loop through the folds
        # plot the ROC curve for current fold
        fpr = fpr_records[i]
        tpr = tpr_records[i]
        thresholds = thresholds_records[i]
        tprs.append(interp(mean_fpr, fpr, tpr)) # interpolate data for averaging
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
                            
    # mean ROC cure
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=color,
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)    
    #----------------------------end of mean_roc function-------------------------------------

if __name__ == "__main__":
    args = get_args()
    log_dir = args.log_dir
    model = args.model

    title = model + ' ROC curve'

    # unpack data
    if model == 'rf':
        pkl_file_control_vs_heme = open(log_dir+'rf_'+ 'control_vs_heme' +'_cv'+'.pickle','rb')
        pkl_file_control_vs_nucleotide = open(log_dir+'rf_'+ 'control_vs_nucleotide' +'_cv'+'.pickle','rb')
    elif model == 'mlp':
        pkl_file_control_vs_heme = open(log_dir+'mlp_'+ 'control_vs_heme' +'_cv'+'.pickle','rb')
        pkl_file_control_vs_nucleotide = open(log_dir+'mlp_'+ 'control_vs_nucleotide' +'_cv'+'.pickle','rb')
    elif model == 'lr':
        pkl_file_control_vs_heme = open(log_dir+'lr_'+ 'control_vs_heme' +'_cv'+'.pickle','rb')
        pkl_file_control_vs_nucleotide = open(log_dir+'lr_'+ 'control_vs_nucleotide' +'_cv'+'.pickle','rb')
    elif model == 'mlp_img':
        pkl_file_control_vs_heme = open(log_dir+'mlp_img_'+ 'control_vs_heme' +'_cv'+'.pickle','rb')
        pkl_file_control_vs_nucleotide = open(log_dir+'mlp_img_'+ 'control_vs_nucleotide' +'_cv'+'.pickle','rb')


    # plot ROC curve of random guess
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)    
    mean_roc(pkl_file_control_vs_heme, color='r')
    mean_roc(pkl_file_control_vs_nucleotide, color='b')
    # miscellaneous
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.show() 



    
   
