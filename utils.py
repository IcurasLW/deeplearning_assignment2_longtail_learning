from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from losses import *
from torch.nn import CrossEntropyLoss
from sklearn.metrics import auc, precision_recall_curve, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import label_binarize
import csv
import os
import pandas as pd
from dataset import *


class ProgressMeter:
    def __init__(self, args, mode, meters:dict, prefix="", ):
        self.meters = meters
        self.prefix = prefix
        self.args = args
        self.mode = mode
        
    def display(self):
        '''
        Will print [Epoch: i] --> loss: loss | Accuracy: acc |  
        '''
        keys = self.meters.keys()
        output = self.prefix
        
        for i in keys:
            output += f' |{i} : {round(self.meters[i], 4)}'

        print(output)
        self.save_results()
    
    
    def save_results(self):
        '''
        Each row is one epoch
        '''
        output_filename = self.args.save_path + f'{self.args.dataname}_{self.mode}_{self.args.modelname}_{self.args.loss}_results.csv'
        # save train results
        
        
        if not os.path.exists(self.args.save_path):
            # If it doesn't exist, create it
            os.mkdir(self.args.save_path)
        
        
        try:
            with open (output_filename, 'r') as log:
                pass
            
            print('file exists')
            with open(output_filename, 'a+', newline='') as log:
                writer = csv.DictWriter(log, fieldnames=self.meters.keys())
                writer.writerow(self.meters)
        except:
            print('file not exists')
            with open(output_filename, 'w', newline='') as log:
                writer = csv.DictWriter(log, fieldnames=self.meters.keys())
                writer.writeheader()
                writer.writerow(self.meters)




def prepare_loss(args,train_dataloader):
    if args.loss == 'focalloss':
        loss_fn = FocalLoss()
    elif args.loss == 'crossentropy':
        loss_fn = CrossEntropyLoss()
    elif args.loss == 'classdistance':
        loss_fn = ClassDistanceWeightedLoss(args.num_class)
    elif args.loss == 'focallossv1':
        threshold = int(0.3 * args.num_class)
        loss_fn = FocalLoss_v1(threshold=threshold)
    elif args.loss == 'cbloss':
        
        label_counts = dict()
        for _, labels in train_dataloader:
            for label in labels:
                label = label.item()
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
        
        label_counts = pd.Series(label_counts)
        loss_fn = CBLoss(label_counts)
        
    return loss_fn


def save_model(val_loss, best_loss, model, args):
    if val_loss <= best_loss:
        torch.save(model.state_dict(), f'best_{args.modelname}_{args.loss}.pth')
    return best_loss


def roc_aupr_score(args, y_true, y_score, average="macro"):
    
    '''
    y_score: the probality of prediction in shape: [num_sample, num_class]
    y_true: the model predcition in shape: [num_sample, num_class], it's one-hot reprensentation of class
    '''
    y_one_hot = label_binarize(y=y_true, classes=[i for i in range(args.num_class)])

    def _binary_roc_aupr_score(y_one_hot, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_one_hot, y_score)
        return auc(recall, precision) # Liangwei: I delete a re-order

    def _average_binary_score(binary_metric, y_one_hot, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_one_hot, y_score)
        if average == "micro":
            y_one_hot = y_one_hot.ravel()
            y_score = y_score.ravel()
        if y_one_hot.ndim == 1:
            y_one_hot = y_one_hot.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_one_hot.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_one_hot, y_score, average)


def evaluate(args, y_pred, y_true):
    # y_pred = F.softmax(y_pred, dim=1)
    pred_max_indices = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_pred=pred_max_indices, y_true=y_true)
    f1_macro = f1_score(y_pred=pred_max_indices, y_true=y_true, average='macro', zero_division=1)
    precision_macro = precision_score(y_pred=pred_max_indices, y_true=y_true, average='macro', zero_division=1)
    recall_macro = recall_score(y_pred=pred_max_indices, y_true=y_true, average='macro', zero_division=1)
    auc_score_macro = roc_auc_score(y_score=y_pred, y_true=y_true, multi_class='ovr', average='macro')
    aupr_score_macro = roc_aupr_score(args, y_true, y_pred, average='macro')
    
    output = {
            'acc':acc,
            'PR_macro': precision_macro,
            'RE_macro': recall_macro,
            'f1_macro': f1_macro,
            'AUC_macro': auc_score_macro,
            'AUPR_macro': aupr_score_macro
            }
    return output

