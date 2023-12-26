import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from models.utils import custom_replace
import random
from utils.Regularizations import Regularization

def run_epoch(args,model,data,optimizer,epoch,desc,train=False,warmup_scheduler=None, criterion=nn.BCEWithLogitsLoss):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_masks = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_image_ids = []

    max_samples = args.max_samples

    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0


    for batch in tqdm(data,mininterval=0.5,desc=desc,leave=False,ncols=50):
        if batch_idx == max_samples:
            break
        
        labels = batch['labels'].float() 
        images = batch['image'].float() 
        mask = batch['mask'].float() 
        
        unk_mask = custom_replace(mask,1,0,0)
        '''
        unk_mask 在mask的基础上将所有0和1都变成0，只有-1变成1
        '''
        all_image_ids += batch['imageIDs']
        
        mask_in = mask.clone()

        if train:
            pred = model(images.cuda(),mask_in.cuda()) 
        else:
            with torch.no_grad():
                pred = model(images.cuda(),mask_in.cuda())
        
        loss = criterion(pred, labels.cuda())
        
        if args.loss_labels == 'unk': 
            # only use unknown labels for loss
            loss_out = (unk_mask.cuda()*loss).sum()
            
        else: 
            # use all labels for loss
            loss_out = loss.sum() 
        

        if train:
            loss_out.backward() 
            # Grad Accumulation
            if ((batch_idx+1)%args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()
        
        ## Updates ##
        loss_total += loss.sum().item() # loss_total should not add loss_out.item()
        unk_loss_total += loss_out.item()
        start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)
        
        
        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0),-1)
        
        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        all_masks[start_idx:end_idx] = mask.data.cpu()
        batch_idx +=1

    loss_total = loss_total/float(all_predictions.size(0))
    unk_loss_total = unk_loss_total/float(all_predictions.size(0))

    return all_predictions,all_targets,all_masks,all_image_ids,loss_total,unk_loss_total


