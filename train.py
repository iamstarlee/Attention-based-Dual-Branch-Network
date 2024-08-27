import torch, torch.nn as nn
from pdb import set_trace as stop
from tqdm import tqdm
from models.utils import custom_replace
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from loguru import logger

def setup(rank, world_size):
    """ Initialize the process group for distributed training. """
    os.environ['MASTER_ADDR'] = 'localhost'  # Master address (can be IP if using multiple nodes)
    os.environ['MASTER_PORT'] = '12355'      # Master port (arbitrary open port number)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """ Clean up and destroy the process group. """
    dist.destroy_process_group()


def train(args,model,data,optimizer,desc,train=False,warmup_scheduler=None, criterion=nn.BCEWithLogitsLoss, rank=0, world_size=2):
    setup(rank, world_size)

    model = DDP(model, device_ids=[rank])
    model.train()
    optimizer.zero_grad()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_masks = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_image_ids = []

    max_samples = args.max_samples

    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0

    for epoch in range(1, args.epochs+1):
        
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
            pred = model(images.cuda(),mask_in.cuda()) 
            loss = criterion(pred, labels.cuda())
            
            if args.loss_labels == 'unk': 
                # only use unknown labels for loss
                loss_out = (unk_mask.cuda()*loss).sum()
            else: 
                # use all labels for loss
                loss_out = loss.sum() 

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

        train_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,train_loss,train_loss_unk,0,args.train_known_labels)
        loss_logger.log_losses('train.log',epoch,train_loss,train_metrics,train_loss_unk, param_group['lr'], start)
        

    cleanup()

