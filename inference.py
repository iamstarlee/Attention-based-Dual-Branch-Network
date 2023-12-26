import os
import numpy as np
from argparse import ArgumentParser
from torchvision import models as torch_models
from torchvision import transforms
from collections import OrderedDict
import pandas as pd
import torch
import torch.nn.functional as F

from dataloaders.sewerml_dataset import MultiLabelDatasetInference, MultiLableTwoStageInference
from torch.utils.data import DataLoader
import torch.nn as nn
from models import CTranModel

torch.set_printoptions(threshold=np.inf) # 设置打印不输出省略号

def evaluate(dataloader, model):
    model.eval()

    sigmoidPredictions = None
    imgPathsList = []

    # sigmoid = nn.Sigmoid()

    dataLen = len(dataloader)

    with torch.no_grad():
        
        for i, sample in enumerate(dataloader):
            if i % 10 == 0:
                print("{} / {}".format(i, dataLen))
            
            # with open("sample.txt", "a") as f:
            #     f.write(str(sample))

            pred = model(sample['image'].cuda(), sample['mask'].cuda())


            # diag_mask = torch.eye(pred.size(1)).unsqueeze(0).repeat(pred.size(0),1,1).cuda() 
            # pred = (pred*diag_mask).sum(-1)


            # with open("pred.txt","a") as f:
            #     f.write(str(pred))
            
            sigmoidOutput = F.sigmoid(pred).detach().cpu().numpy()
            
            # sigmoidOutput = pred.detach().cpu().numpy()
            
            # with open("sigmoid.txt","a") as f:
            #     f.write(str(sigmoidOutput))

            
            sigmoidOutput[sigmoidOutput < 0.5] = 0
            sigmoidOutput[sigmoidOutput >= 0.5] = 1

            if sigmoidPredictions is None:
                sigmoidPredictions = sigmoidOutput
            else:
                sigmoidPredictions = np.vstack((sigmoidPredictions, sigmoidOutput))

            imgPathsList.extend(list(sample['imageIDs']))

            # with open("sample.txt","a") as f:
            #     f.write(str(sigmoidPredictions))

    return sigmoidPredictions, imgPathsList


def load_model(model_path):

    best_model_state_dict = torch.load(model_path)["state_dict"]

    updated_state_dict = OrderedDict()
    for k,v in best_model_state_dict.items():
        name = k.replace("module.", "")
        if "criterion" in name:
            continue

        updated_state_dict[name] = v

    return updated_state_dict


def run_inference(args):

    ann_root = './Annotations'
    data_root = './Datasets/Sewer-ML'
    model_path = './results/bsz_64.sgd0.0001.lmt.unk_loss_w2v/best_model.pt'
    outputPath = './sigmoid_results'
    split = 'Val'
    
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
  
    updated_state_dict = load_model(model_path)

    model_version = 'C-Tran'
    
    # Init model
    model = CTranModel(args.num_labels,args.use_lmt,args.pos_emb,args.layers,args.heads,args.dropout,args.no_x_features)
    model.load_state_dict(updated_state_dict)
    
    # initialize dataloaders
    eval_transform= transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])])
    
    dataset = MultiLabelDatasetInference(ann_root, data_root, split=split, transform=eval_transform, onlyDefects=False)
    # dataset = MultiLableTwoStageInference(annRoot='', imgRoot=data_root, split=split, transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, num_workers = 4, pin_memory=True)

    labelNames = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # model = nn.parallel.DataParallel(model.cuda(), device_ids = [0,1])

    # Validation results
    print("Let's go!")
    sigmoid_predictions, val_imgPaths = evaluate(dataloader, model)

    sigmoid_dict = {}
    sigmoid_dict["Filename"] = val_imgPaths
    for idx, header in enumerate(labelNames):
        sigmoid_dict[header] = sigmoid_predictions[:,idx]

    
    sigmoid_df = pd.DataFrame(sigmoid_dict)
    sigmoid_df.to_csv(os.path.join(outputPath, "{}_{}_sigmoid.csv".format(model_version, split.lower())), sep=",", index=False)


if __name__ == "__main__":
    from config_args import get_args
    import argparse
    args = get_args(argparse.ArgumentParser())
    
    run_inference(args)