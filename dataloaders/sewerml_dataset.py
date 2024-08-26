import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from .data_utils import get_unk_mask_indices
import argparse


Labels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK", "VA", "ND"]

class MultiLabelDataset(Dataset):
    def __init__(self, img_dir, labels_path, image_transform=None, loader=default_loader, onlyDefects=False, known_labels=0,testing=False,split='Train'):
        super(MultiLabelDataset, self).__init__()
        self.img_dir = img_dir
        self.labels_path = labels_path
        self.testing = testing
        self.split = split

        self.image_transform = image_transform
        self.loader = loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)
        self.known_labels = known_labels
        
        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.labels_path, "SewerML_{}.csv".format(self.split))
        
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.LabelNames + ["Filename", "Defect"])
       
        if self.onlyDefects:
            gt = gt[gt["Defect"] == 1]

        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values
        

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]
    
        img = self.loader(os.path.join(self.img_dir, self.split, path)) 

        if self.image_transform is not None:
            img = self.image_transform(img)


        labels = torch.Tensor(self.labels[index, :]) 
        unk_mask_indices = get_unk_mask_indices(img,self.testing,self.num_classes,self.known_labels)

        mask = labels.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = img
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = str(path)

        return sample


    def getClassWeights(self):
        data_len = self.labels.shape[0]
        # print('The shape of labels is {}'.format(self.labels.shape)) # (31711,17)
        class_weights = []

        for defect in range(self.num_classes):
            pos_count = len(self.labels[self.labels[:,defect] == 1])
            '''
            self.labels[:,defect] == 1 表示defect列有哪些行是1
            self.labels[self.labels[:,defect] == 1] 表示选择为1的行
            len(self.labels[self.labels[:,defect] == 1]) 表示为1的行的个数
            '''
            neg_count = data_len - pos_count

            class_weight = neg_count/pos_count if pos_count > 0 else 0
            class_weights.append(np.asarray([class_weight]))
            '''
            UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
            Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
            '''
        return torch.as_tensor(np.array(class_weights)).squeeze()


class MultiLabelDatasetInference(Dataset):
    def __init__(self, annRoot, imgRoot, split="Val", transform=None, loader=default_loader, onlyDefects=False):
        super(MultiLabelDatasetInference, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)
        
        self.loadAnnotations()

    def loadAnnotations(self):
        
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.LabelNames + ["Filename", "Defect"])
        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values

        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, self.split, path))
        
        if self.transform is not None:
            img = self.transform(img)

        labels = torch.Tensor(self.labels[index, :]) 
        mask = torch.Tensor([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
        
        sample = {}
        sample['image'] = img
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = str(path)

        return sample


class MultiLableTwoStageInference(Dataset):
    def __init__(self, annRoot, imgRoot, split="Val", transform=None, loader=default_loader) -> None:
        super(MultiLableTwoStageInference, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        
        self.num_classes = len(self.LabelNames)
        
        self.loadAnnotations()
    
    def loadAnnotations(self):
        gtPath = 'binary_results/after.csv'
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.LabelNames + ["Filename", "Defect"])
        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, self.split, path)) 
       
        if self.transform is not None:
            img = self.transform(img)

        labels = torch.Tensor(self.labels[index, :]) 
        mask = torch.Tensor([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
        
        sample = {}
        sample['image'] = img
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = str(path)

        return sample
    

if __name__ == "__main__":
    pass
    # Train_Transform = transforms.Compose([transforms.Resize((640, 640)),
    #                                     transforms.RandomChoice([
    #                                     transforms.RandomCrop(640),
    #                                     transforms.RandomCrop(576),
    #                                     transforms.RandomCrop(512),
    #                                     transforms.RandomCrop(384),
    #                                     transforms.RandomCrop(320)
    #                                     ]),
    #                                     transforms.Resize((576, 576)),
    #                                     transforms.RandomHorizontalFlip(),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Test_Transform = transforms.Compose([transforms.Resize((640, 640)),
    #                                     transforms.CenterCrop(576),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
    # train_dataset = MultiLabelDataset(labels_path="./Annotations", img_dir="./Datasets/mini", split="Train", image_transform=Train_Transform)
    # val_dataset = MultiLabelDatasetInference(annRoot="./Annotations", imgRoot="./Datasets/mini", split="Val", transform=Test_Transform)
    # torch.set_printoptions(threshold=np.inf) # 设置打印不输出省略号
    # # print(len(train_dataset)) # 1040129
    
    # train_loader = DataLoader(train_dataset, batch_size=64,shuffle=True, drop_last=False, num_workers=8) 
    # val_loader = DataLoader(val_dataset, batch_size=64,shuffle=True, drop_last=False, num_workers=8)
    # # print(len(train_loader)) # 1040129/64 = 16253
    # # print(train_dataset.__dict__)
    # # print('==========================')
    # # print(val_dataset.__dict__)
    # from tqdm import tqdm
    # for batch in tqdm(val_loader, mininterval=0.5,desc='Inference',leave=False,ncols=50):
    #     print(batch)
    #     # with open("../sample.txt", "a") as f:
    #     #         f.write(str(sample))
    #     # print('The shape of images is {}'.format(sample['image'].shape)) # torch.Size([64, 3, 576, 576])
    # # print('the dataset are {}'.format(train_dataset.__dict__))