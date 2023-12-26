import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse,math,numpy as np
from load_data import get_data
from models import CTranModel
from config_args import get_args
import utils.evaluate as evaluate
import utils.logger as logger
from pdb import set_trace as stop
from optim_schedule import WarmupLinearSchedule
from run_epoch import run_epoch
import os
import re
import time



if __name__ == '__main__':
    args = get_args(argparse.ArgumentParser())

    print('Labels: {}'.format(args.num_labels))
    print('Train Known: {}'.format(args.train_known_labels))
    print('Test Known:  {}'.format(args.test_known_labels))

    train_loader,valid_loader,test_loader, class_weights = get_data(args)

    model = CTranModel(args.num_labels,args.use_lmt,args.pos_emb,args.layers,args.heads,args.dropout,args.no_x_features)


    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), "GPUs!")
        
        model = nn.parallel.DataParallel(model.cuda(), device_ids=[0,1])
        '''
        将之前单卡运行时的所有.cuda()替换为.to(device)，会遇到问题，因为
        a = torch.nn.Parameter(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True)
        a = torch.autograd.Variable(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True)
        Variable和Parameter的区别在于前者需要将参数to(device)才能放到GPU上，而后者只需要将整个模型to(device)，而参数不需要to(device)
        '''
    else:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device = torch.device('cuda:0')
        model.to(device)

    if args.inference:
        checkpoint = torch.load(os.path.join('results', args.saved_model_name, 'best_model.pt'))
        model.load_state_dict(checkpoint['state_dict'])
        if test_loader is not None:
            data_loader =test_loader
        else:
            data_loader =valid_loader
        
        all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk = run_epoch(args,model,data_loader,None,1,'Testing')
        test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,args.test_known_labels)
        print('The Inference are: \n')
        print(test_metrics)
        exit(0)

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr, weight_decay=1e-4) 
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    if args.warmup_scheduler:
        step_scheduler = None
        scheduler_warmup = WarmupLinearSchedule(optimizer, 1, 300000)
    else:
        scheduler_warmup = None
        if args.scheduler_type == 'plateau':
            step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        elif args.scheduler_type == 'step':
            step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        else:
            step_scheduler = None


    metrics_logger = logger.Logger(args)
    loss_logger = logger.LossLogger(args.model_name)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(1,args.epochs+1):
        print('======================== {} ========================'.format(epoch))
        for param_group in optimizer.param_groups:
            print('LR: {}'.format(param_group['lr']))
        
        start = time.time()

        ################### Train #################
        all_preds,all_targs,all_masks,all_ids,train_loss,train_loss_unk = run_epoch(args,model,train_loader,optimizer,epoch,'Training',train=True, warmup_scheduler=scheduler_warmup, criterion=criterion)
        train_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,train_loss,train_loss_unk,0,args.train_known_labels)
        loss_logger.log_losses('train.log',epoch,train_loss,train_metrics,train_loss_unk, param_group['lr'], start) # type: ignore
        
        print(f'=== Epoch {epoch} training costs {(time.time() - start)/60} mins ===\n')
        middle = time.time()

        ################### Valid #################
        all_preds,all_targs,all_masks,all_ids,valid_loss,valid_loss_unk = run_epoch(args,model,valid_loader,None,epoch,'Validating', criterion=criterion)
        valid_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,valid_loss,valid_loss_unk,0,args.test_known_labels)
        loss_logger.log_losses('valid.log',epoch,valid_loss,valid_metrics,valid_loss_unk, param_group['lr'], middle) # type: ignore
        print(f'=== Epoch {epoch} validating costs {(time.time() - middle)/60} mins ===\n')

        ################### Test #################
        if test_loader is not None:
            all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk = run_epoch(args,model,test_loader,None,epoch,'Testing', criterion=criterion)
            test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,args.test_known_labels)
        else:
            test_loss,test_loss_unk,test_metrics = valid_loss,valid_loss_unk,valid_metrics
        loss_logger.log_losses('test.log',epoch,test_loss,test_metrics,test_loss_unk, param_group['lr'], 0) # type: ignore

        

        if step_scheduler is not None:
            if args.scheduler_type == 'step':
                step_scheduler.step(epoch)
            elif args.scheduler_type == 'plateau':
                step_scheduler.step(int(valid_loss_unk))

        ############## Log and Save ##############
        best_valid,best_test = metrics_logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch,0,model,valid_loss,test_loss,all_preds,all_targs,all_ids,args)

        print(args.model_name)
