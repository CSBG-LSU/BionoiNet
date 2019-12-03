"""
Plot the mean ROC curve for 10-fold cross-validation of each algorithm
"""

"""
Take the pickle files that contains results of cross-validation, then 
compute the final results and draw the mean ROC cure.
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

    parser.add_argument('-op',
                        default='control_vs_heme',
                        required=False,
                        choices = ['control_vs_heme', 'control_vs_nucleotide', 'heme_vs_nucleotide'],
                        help='control_vs_heme, control_vs_nucleotide, heme_vs_nucleotide')

    parser.add_argument('-model',
                        default='rf',
                        required=False,
                        choices = ['rf', 'mlp', 'lr', 'mlp_img'],
                        help='random forest, multi-layer perceptron, logistic regression or mlp direclty on images')

    parser.add_argument('-log_dir',
                        default='./log/',
                        required=False,
                        help='directory to save data for cross validation.')

    return parser.parse_args()

def mean_roc(pkl_file, title, indiv_plot=False):
    """
    Plot the average ROC curve for 10-fold cross validation
    pik_file   ---- contains the dictionary of the validation metrics.
    indiv_plot ---- bool, whether to plot the ROC curve for each fold. 
    """
    # unpack data
    records = pickle.load(pkl_file)
    val_acc_records = records['val_acc_records']
    val_precision_records = records['val_precision_records']
    val_recall_records = records['val_recall_records']
    val_mcc_records = records['val_mcc_records']
    fpr_records = records['fpr_records']
    tpr_records = records['tpr_records']
    thresholds_records = records['thresholds_records']

    # mean accuracy and mean mcc
    print('mean validation accuracy:', mean(val_acc_records))
    print('mean validation precision:', mean(val_precision_records))
    print('mean validation recall:', mean(val_recall_records))
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
        if indiv_plot == True:
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.3f)' % (i+1, roc_auc))

    # plot ROC curve of random guess
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)                             
    
    # mean ROC cure
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)    
    
    # variance of the ROC curve
    if indiv_plot == True:
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')   
    
    # miscellaneous
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of {}'.format(op))
    plt.legend(loc="lower right")
    #----------------------------end of mean_roc function-------------------------------------

if __name__ == "__main__":
    args = get_args()
    op = args.op
    log_dir = args.log_dir
    model = args.model

    if op == 'control_vs_heme':
        classes = ('control','heme')   
    elif op == 'control_vs_nucleotide':
        classes = ('control','nucleotide')
    elif op == 'heme_vs_nucleotide':
        classes = ('heme','nucleotide')

    title = model + ' for ' + classes[0] + ' vs ' + classes[1]

    # unpack data
    if model == 'rf':
        pkl_file = open(log_dir+'rf_'+op+'_cv'+'.pickle','rb')
    elif model == 'mlp':
        pkl_file = open(log_dir+'mlp_'+op+'_cv'+'.pickle','rb')
    elif model == 'lr':
        pkl_file = open(log_dir+'lr_'+op+'_cv'+'.pickle','rb')
    elif model == 'mlp_img':
        pkl_file = open(log_dir+'mlp_img_'+op+'_cv'+'.pickle','rb')        

    mean_roc(pkl_file, title, indiv_plot=True)
    plt.show() 



    
   
