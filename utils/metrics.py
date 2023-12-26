
"""
Helper functions to compute all metrics needed
"""
import numpy as np
import scipy.sparse as sp
import logging
from collections import OrderedDict
import sys
import pdb
from sklearn import metrics
from threading import Lock
from threading import Thread
import torch
import math
from pdb import set_trace as stop
import pandas as pd

def error_rate(true_targets,predictions):
    acc = metrics.accuracy_score(true_targets, predictions)
    error_rate = 1-acc
    return error_rate

def subset_accuracy(true_targets, predictions, per_sample=False, axis=0):
    # print(true_targets.shape)
    # print(predictions.shape)
    result = np.all(true_targets == predictions, axis=axis)

    if not per_sample:
        result = np.mean(result)

    return result


def hamming_loss(true_targets, predictions, per_sample=False, axis=0):

    result = np.mean(np.logical_xor(true_targets, predictions),
                        axis=axis)

    if not per_sample:
        result = np.mean(result)

    return result


def compute_tp_fp_fn(true_targets, predictions, axis=0):
    # axis: axis for instance
    tp = np.sum(true_targets * predictions, axis=axis).astype('float32')
    fp = np.sum(np.logical_not(true_targets) * predictions,
                   axis=axis).astype('float32')
    fn = np.sum(true_targets * np.logical_not(predictions),
                   axis=axis).astype('float32')

    return (tp, fp, fn)


def example_f1_score(true_targets, predictions, per_sample=False, axis=0):
    tp, _, _ = compute_tp_fp_fn(true_targets, predictions, axis=axis)

    numerator = 2*tp
    denominator = (np.sum(true_targets,axis=axis).astype('float32') + np.sum(predictions,axis=axis).astype('float32'))

    zeros = np.where(denominator == 0)[0]

    denominator = np.delete(denominator,zeros)
    numerator = np.delete(numerator,zeros)

    example_f1 = numerator/denominator


    if per_sample:
        f1 = example_f1
    else:
        f1 = np.mean(example_f1)

    return f1




def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    f1 = 0.0 
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    if average == 'micro':
        f1 = 2*np.sum(tp) / \
            float(2*np.sum(tp) + np.sum(fp) + np.sum(fn))

    elif average == 'macro':

        def safe_div(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
            return c[np.isfinite(c)]

        f1 = np.mean(safe_div(2*tp, 2*tp + fp + fn))

    return f1


def f1_score(true_targets, predictions, average='micro', axis=0):
    """
        average: str
            'micro' or 'macro'
        axis: 0 or 1
            label axis
    """
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)

    return f1


def compute_aupr_thread(all_targets,all_predictions):

    aupr_array = []
    lock = Lock()

    def compute_aupr_(start,end,all_targets,all_predictions):
        for i in range(all_targets.shape[1]):
            try:
                precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
                auPR = metrics.auc(recall,precision)
                lock.acquire()
                aupr_array.append(np.nan_to_num(auPR))
                lock.release()
            except Exception:
                pass

    t1 = Thread(target=compute_aupr_, args=(0,100,all_targets,all_predictions) )
    t2 = Thread(target=compute_aupr_, args=(100,200,all_targets,all_predictions) )
    t3 = Thread(target=compute_aupr_, args=(200,300,all_targets,all_predictions) )
    t4 = Thread(target=compute_aupr_, args=(300,400,all_targets,all_predictions) )
    t5 = Thread(target=compute_aupr_, args=(400,500,all_targets,all_predictions) )
    t6 = Thread(target=compute_aupr_, args=(500,600,all_targets,all_predictions) )
    t7 = Thread(target=compute_aupr_, args=(600,700,all_targets,all_predictions) )
    t8 = Thread(target=compute_aupr_, args=(700,800,all_targets,all_predictions) )
    t9 = Thread(target=compute_aupr_, args=(800,900,all_targets,all_predictions) )
    t10 = Thread(target=compute_aupr_, args=(900,919,all_targets,all_predictions) )
    t1.start();t2.start();t3.start();t4.start();t5.start();t6.start();t7.start();t8.start();t9.start();t10.start()
    t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();t7.join();t8.join();t9.join();t10.join()


    aupr_array = np.array(aupr_array)

    mean_aupr = np.mean(aupr_array)
    median_aupr = np.median(aupr_array)
    return mean_aupr,median_aupr,aupr_array

def compute_fdr(all_targets,all_predictions, fdr_cutoff=0.5):
    fdr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i],pos_label=1)
            fdr = 1- precision
            cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
            fdr_at_cutoff = recall[cutoff_index]
            if not math.isnan(fdr_at_cutoff):
                fdr_array.append(np.nan_to_num(fdr_at_cutoff))
        except:
            pass

    fdr_array = np.array(fdr_array)
    mean_fdr = np.mean(fdr_array)
    median_fdr = np.median(fdr_array)
    var_fdr = np.var(fdr_array)
    return mean_fdr,median_fdr,var_fdr,fdr_array


def compute_aupr(all_targets,all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
            auPR = metrics.auc(recall,precision)
            if not math.isnan(auPR):
                aupr_array.append(np.nan_to_num(auPR))
        except:
            pass

    aupr_array = np.array(aupr_array)
    mean_aupr = np.mean(aupr_array)
    median_aupr = np.median(aupr_array)
    var_aupr = np.var(aupr_array)
    return mean_aupr,median_aupr,var_aupr,aupr_array



def compute_auc_thread(all_targets,all_predictions):

    auc_array = []
    lock = Lock()

    def compute_auc_(start,end,all_targets,all_predictions):
        for i in range(start,end):
            try:
                auROC = metrics.roc_auc_score(all_targets[:,i], all_predictions[:,i])
                lock.acquire()
                if not math.isnan(auROC):
                    auc_array.append(auROC)
                lock.release()
            except ValueError:
                pass

    t1 = Thread(target=compute_auc_, args=(0,100,all_targets,all_predictions) )
    t2 = Thread(target=compute_auc_, args=(100,200,all_targets,all_predictions) )
    t3 = Thread(target=compute_auc_, args=(200,300,all_targets,all_predictions) )
    t4 = Thread(target=compute_auc_, args=(300,400,all_targets,all_predictions) )
    t5 = Thread(target=compute_auc_, args=(400,500,all_targets,all_predictions) )
    t6 = Thread(target=compute_auc_, args=(500,600,all_targets,all_predictions) )
    t7 = Thread(target=compute_auc_, args=(600,700,all_targets,all_predictions) )
    t8 = Thread(target=compute_auc_, args=(700,800,all_targets,all_predictions) )
    t9 = Thread(target=compute_auc_, args=(800,900,all_targets,all_predictions) )
    t10 = Thread(target=compute_auc_, args=(900,919,all_targets,all_predictions) )
    t1.start();t2.start();t3.start();t4.start();t5.start();t6.start();t7.start();t8.start();t9.start();t10.start()
    t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();t7.join();t8.join();t9.join();t10.join()

    auc_array = np.array(auc_array)

    mean_auc = np.mean(auc_array)
    median_auc = np.median(auc_array)
    return mean_auc,median_auc,auc_array


def compute_auc(all_targets,all_predictions):
    auc_array = []
    lock = Lock()

    for i in range(all_targets.shape[1]):
        try:
            auROC = metrics.roc_auc_score(all_targets[:,i], all_predictions[:,i])
            auc_array.append(auROC)
        except ValueError:
            pass

    auc_array = np.array(auc_array)
    mean_auc = np.mean(auc_array)
    median_auc = np.median(auc_array)
    var_auc = np.var(auc_array)
    return mean_auc,median_auc,var_auc,auc_array


def Find_Optimal_Cutoff(all_targets, all_predictions):
    thresh_array = []
    for j in range(all_targets.shape[1]):
        try:
            fpr, tpr, threshold = metrics.roc_curve(all_targets[:,j], all_predictions[:,j], pos_label=1)
            i = np.arange(len(tpr))
            roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
            roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
            thresh_array.append(list(roc_t['threshold'])[0])

        except:
            pass
    return thresh_array



def mean_avg_precision(true_targets, predictions, axis=0):
    meanAP = metrics.average_precision_score(true_targets, predictions, average='macro', pos_label=1)
    return meanAP

def custom_mean_avg_precision(all_targets, all_predictions, unknown_label_mask):
    APs = []
    all_predictions = torch.where(torch.isnan(all_predictions), torch.full_like(all_predictions, 0), all_predictions)
    for label_idx in range(all_targets.size(1)):
        all_targets_unk = torch.masked_select(all_targets[:,label_idx],unknown_label_mask[:,label_idx].type(torch.ByteTensor))
        all_predictions_unk = torch.masked_select(all_predictions[:,label_idx],unknown_label_mask[:,label_idx].type(torch.ByteTensor))
        if len(all_targets_unk)>0 and all_targets_unk.sum().item() > 0:
            AP = metrics.average_precision_score(all_targets_unk, all_predictions_unk, average=None, pos_label=1)
            APs.append(AP)
    meanAP = np.array(APs).mean()
    return meanAP



def class_weighted_f2(Ng, Np, Nc, weights):
    n_class = len(Ng)
    precision_k = Nc / Np
    recall_k = Nc / Ng
    F2_k = (5 * precision_k * recall_k)/(4*precision_k + recall_k)

    F2_k[np.isnan(F2_k)] = 0

    ciwF2 = F2_k * weights 
    ciwF2 = np.sum(ciwF2) / np.sum(weights)

    return ciwF2, F2_k

def exact_match_accuracy(scores, targets, threshold):
    n_examples, n_class = scores.shape

    binary_mat = np.equal(targets, (scores >= threshold))
    row_sums = binary_mat.sum(axis=1)

    perfect_match = np.zeros(row_sums.shape)
    perfect_match[row_sums == n_class] = 1

    EMAcc = np.sum(perfect_match) / n_examples

    return EMAcc


def macro_f1(Ng, Np, Nc):
    n_class = len(Ng)
    precision_k = Nc / Np
    precision_k[np.isnan(precision_k)] = 0
    
    recall_k = Nc / Ng
    F1_k = (2 * precision_k * recall_k)/(precision_k + recall_k)

    F1_k[np.isnan(F1_k)] = 0

    MF1 = np.sum(F1_k)/n_class

    return precision_k, recall_k, F1_k, MF1

def average_precision(scores, target, max_k = None):
    
    assert scores.shape == target.shape, "The input and targets do not have the same shape"
    assert scores.ndim == 1, "The input has dimension {}, but expected it to be 1D".format(scores.shape)

    # sort examples
    indices = np.argsort(scores, axis=0)[::-1]

    total_cases = np.sum(target)

    if max_k == None:
        max_k = len(indices)
    
    # Computes prec@i
    pos_count = 0.
    total_count = 0.
    precision_at_i = 0.

    for i in range(max_k):
        label = target[indices[i]]
        total_count += 1
        if label == 1:
            pos_count += 1
            precision_at_i += pos_count / total_count
        if pos_count == total_cases:
            break
        
    if pos_count > 0:
        precision_at_i /= pos_count
    else:
        precision_at_i = 0
    return precision_at_i