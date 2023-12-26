import torch
from skimage import transform
import numpy as np
from torch.utils.data import Dataset, DataLoader, distributed
from torchvision import transforms
from pdb import set_trace as stop
import os, random
from dataloaders.sewerml_dataset import MultiLabelDataset, MultiLabelDatasetInference


def get_data(args):
 
    workers=args.workers

    trainTransform = transforms.Compose([transforms.Resize((args.scale_size, args.scale_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue = 0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])])

    testTransform = transforms.Compose([transforms.Resize((args.scale_size, args.scale_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])])
    
    train_loader = None
    valid_loader = None
    drop_last = False
    train_dataset =None
    
    sewer_root = os.path.join(args.dataroot, 'Sewer-ML')
    train_dir = sewer_root
    val_dir = sewer_root
    anno_dir = 'Annotations/'

    if args.inference:
        valid_dataset = MultiLabelDatasetInference(
            annRoot=anno_dir,
            imgRoot=val_dir,
            split='Val',
            transform=testTransform,
            onlyDefects=False)
    else:
        train_dataset = MultiLabelDataset(
            img_dir=train_dir,
            image_transform=trainTransform,
            labels_path=anno_dir,
            known_labels=args.train_known_labels,
            testing=False,
            split='Train'
            )
        valid_dataset = MultiLabelDataset(
            img_dir=val_dir,
            image_transform=testTransform,
            labels_path=anno_dir,
            known_labels=args.test_known_labels,
            testing=True,
            split='Val')
        train_classweights = train_dataset.class_weights

    
    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, drop_last=drop_last, num_workers=workers) 
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size,shuffle=False, num_workers=workers, pin_memory=True) 
    
    
    return train_loader,valid_loader,None, train_classweights


if __name__ == '__main__':
    pass
