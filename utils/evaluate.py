
import numpy as np
import logging
from collections import OrderedDict
import torch
import math
from pdb import set_trace as stop
import os
from models.utils import custom_replace
from utils.metrics import *
import torch.nn.functional as F 
import warnings


LabelWeightDict = {"RB":1.00,"OB":0.5518,"PF":0.2896,"DE":0.1622,"FS":0.6419,"IS":0.1847,"RO":0.3559,"IN":0.3131,"AF":0.0811,"BE":0.2275,"FO":0.2477,"GR":0.0901,"PH":0.4167,"PB":0.4167,"OS":0.9009,"OP":0.3829,"OK":0.4396}
Labels = list(LabelWeightDict.keys())
LabelWeights = list(LabelWeightDict.values())

def compute_metrics(args,all_predictions,all_targets,all_masks,loss,loss_unk,elapsed,known_labels=0,verbose=True):
    
    assert all_predictions.shape == all_targets.shape, "The input and targets do not have the same shape: Input: {} - Targets: {}".format(all_predictions.shape, all_targets.shape)

    all_predictions = F.sigmoid(all_predictions)

    unknown_label_mask = custom_replace(all_masks,1,0,0)

    if known_labels > 0:
        meanAP = custom_mean_avg_precision(all_targets,all_predictions,unknown_label_mask)
    else:
        meanAP = metrics.average_precision_score(all_targets,all_predictions, average='macro', pos_label=1)

    optimal_threshold = 0.5 

    all_targets = all_targets.numpy()
    all_predictions = all_predictions.numpy()

    
    all_predictions_thresh = all_predictions.copy()
    all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
    all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1
    
    # This will cause the same CF1 and OF1
    CP = metrics.precision_score(all_targets, all_predictions_thresh, average='macro')
    CR = metrics.recall_score(all_targets, all_predictions_thresh, average='macro')
    CF1 = (2*CP*CR)/(CP+CR)
    
    OP = metrics.precision_score(all_targets, all_predictions_thresh, average='micro')
    OR = metrics.recall_score(all_targets, all_predictions_thresh, average='micro')
    OF1 = (2*OP*OR)/(OP+OR)

    MF1 = np.sum(CF1) / 18
    print(f'CP is {CP*100}')
    print(f'CR is {CR*100}')
    print(f'CF1 is {CF1*100}')
    print(f'OP is {OP*100}')
    print(f'OR is {OR*100}')
    print(f'OF1 is {OF1*100}')

    # Zero-One exact match accuracy
    EMAcc = exact_match_accuracy(all_predictions, all_targets, threshold=optimal_threshold)
    
    Nc = np.zeros(18) # Nc = Number of Correct Predictions  - True positives
    Np = np.zeros(18) # Np = Total number of Predictions    - True positives + False Positives
    Ng = np.zeros(18) # Ng = Total number of Ground Truth occurences

    '''
    False Positives = Np - Nc
    False Negatives = Ng - Nc
    True Positives = Nc
    True Negatives = n_examples - Np + (Ng - Nc)
    '''
    
    # Array to hold the average precision metric. only size n_class, since it is not possible to calculate for the implicit normal class
    ap = np.zeros(17)
    
    for k in range(17):
        tmp_scores = all_predictions[:, k]
        tmp_targets = all_targets[:, k]
        tmp_targets[tmp_targets == -1] = 0 # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss

        Ng[k] = np.sum(tmp_targets == 1)
        Np[k] = np.sum(tmp_scores >= optimal_threshold) # when >= 0 for the raw input, the sigmoid value will be >= 0.5
        Nc[k] = np.sum(tmp_targets * (tmp_scores >= optimal_threshold))
        
        ap[k] = average_precision(tmp_scores, tmp_targets)

    # Get values for the 'implicit' normal class
    tmp_scores = np.sum(all_predictions >= optimal_threshold, axis=1)
    tmp_scores[tmp_scores > 0] = 1
    tmp_scores = np.abs(tmp_scores - 1)
    
    tmp_targets = all_targets.copy()
    tmp_targets[all_targets == -1] = 0 # Necessary if using MultiLabelSoftMarginLoss, instead of BCEWithLogitsLoss
    tmp_targets = np.sum(tmp_targets, axis=1)
    tmp_targets[tmp_targets > 0] = 1
    tmp_targets = np.abs(tmp_targets - 1)

    Ng[-1] = np.sum(tmp_targets == 1)
    Np[-1] = np.sum(tmp_scores >= optimal_threshold)
    Nc[-1] = np.sum(tmp_targets * (tmp_scores >= optimal_threshold))

    print(f'Np are {Np}')
    print(f'Ng are {Ng}')
    print(f'Nc are {Nc}')

    Np[Np == 0] = 1

    # for all labels num_imgs*n_classes
    OP = np.sum(Nc) / np.sum(Np)        # precision: true_positive/positive
    OR = np.sum(Nc) / np.sum(Ng)        # recall:    true_positive/true
    
    OF1 = (2 * OP * OR) / (OP + OR)     # F1_score: harmonic mean of precision and recall
    
    # average by class
    CP = np.sum(Nc / Np) / 17
    
    # CP = np.sum(Nc / Np) / 17      # precision: true_positive/positive
    CR = np.sum(Nc / Ng) / 17      # recall:    true_positive/true

    if CP == 0 and CR == 0:
        CF1 = 0
    else:
        CF1 = (2 * CP * CR) / (CP + CR)     # F1_score: harmonic mean of precision and recall

    print(f'CP are {CP*100}')
    print(f'CR are {CR*100}')
    print(f'CF1 are {CF1*100}')
    print(f'OP are {OP*100}')
    print(f'OR are {OR*100}')
    print(f'OF1 are {OF1*100}')

    precision_k, recall_k, F1_k, MF1 = macro_f1(Ng, Np, Nc)
    F2_normal = (5 * precision_k[-1] * recall_k[-1])/(4*precision_k[-1] + recall_k[-1])

    F2, F2_k = class_weighted_f2(Ng[:-1], Np[:-1], Nc[:-1], LabelWeights)

    acc_ = list(subset_accuracy(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions_thresh, axis=1, per_sample=True))  # type: ignore
    acc = np.mean(acc_)
    hl = np.mean(hl_)
    exf1 = np.mean(exf1_)

    eval_ret = OrderedDict([('Subset accuracy', acc),
                        ('Hamming accuracy', 1 - hl),
                        ('Example-based F1', exf1),
                        ('Label-based Micro F1', OF1),
                        ('Label-based Macro F1', CF1)])

    
    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    OF1 = eval_ret['Label-based Micro F1']
    CF1 = eval_ret['Label-based Macro F1']

    if verbose:
        print('loss:  {:0.3f}'.format(loss))
        print('lossu: {:0.3f}'.format(loss_unk))
        print('----')
        print('mAP:   {:0.1f}'.format(meanAP*100))
        print('----')
        print('CP:    {:0.1f}'.format(CP*100))
        print('CR:    {:0.1f}'.format(CR*100))
        print('CF1:   {:0.1f}'.format(CF1*100))
        print('OP:    {:0.1f}'.format(OP*100))
        print('OR:    {:0.1f}'.format(OR*100))
        print('OF1:   {:0.1f}'.format(OF1*100))
        

    metrics_dict = {}
    metrics_dict['mAP'] = meanAP*100
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['loss'] = loss
    metrics_dict['lossu'] = loss_unk
    metrics_dict['time'] = elapsed
    metrics_dict['mF1'] = OF1*100
    metrics_dict['MF1'] = MF1*100
    metrics_dict['CP'] = CP*100
    metrics_dict['CR'] = CR*100
    metrics_dict['CF1'] = CF1*100
    metrics_dict['OP'] = OP*100
    metrics_dict['OR'] = OR*100
    metrics_dict['OF1'] = OF1*100
    metrics_dict['EMAcc'] = EMAcc*100
    metrics_dict['F2'] = F2*100
    metrics_dict['F2_class'] = [i*100 for i in F2_k] + [F2_normal*100]
    metrics_dict['F1_Normal'] = F1_k[-1]*100
    metrics_dict['P_class'] = precision_k*100
    metrics_dict['R_class'] = recall_k*100
    metrics_dict['F1_class'] = F1_k*100
    metrics_dict['ap'] = ap*100

    return metrics_dict
