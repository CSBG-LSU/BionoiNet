
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
        avg_acc += ditc['acc']
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
a = [{'loss': 0.4428280530602831, 'acc': 0.824387779362815, 'precision': 0.7772517675991393, 'recall': 0.8468649517684887, 'mcc': 0.6495252289314343}, {'loss': 0.5724882885387965, 'acc': 0.8167559523809523, 'precision': 0.915490288962577, 'recall': 0.6473070739549839, 'mcc': 0.641790531523949}, {'loss': 0.39063987834112984, 'acc': 0.8587797619047619, 'precision': 0.906766797155868, 'recall': 0.7603161843515541, 'mcc': 0.7171680605901883}, {'loss': 0.5236937225746097, 'acc': 0.8414103481163567, 'precision': 0.8136772330511256, 'recall': 0.8331989247311828, 'mcc': 0.6796548413741098}, {'loss': 0.6801462696175037, 'acc': 0.7942596566523605, 'precision': 0.7076802915907315, 'recall': 0.9133736559139785, 'mcc': 0.6152673037367277}]

report_avg_metrics(a)