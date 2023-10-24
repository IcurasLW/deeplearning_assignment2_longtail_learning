import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch import Tensor
from collections import Counter
import pandas as pd



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ClassDistanceWeightedLoss(torch.nn.Module):
    """
    Instead of calculating the confidence of true class, this class takes into account the confidences of
    non-ground-truth classes and scales them with the neighboring distance.
    Paper: "Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation" (https://arxiv.org/abs/2202.05167)
    It is advised to experiment with different power terms. When searching for new power term, linearly increasing
    it works the best due to its exponential effect.

    """

    def __init__(self, class_size: int, power: float = 2., reduction: str = "mean"):
        super(ClassDistanceWeightedLoss, self).__init__()
        self.class_size = class_size
        self.power = power
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_sm = input.softmax(dim=1)

        weight_matrix = torch.zeros_like(input_sm)
        for i, target_item in enumerate(target):
            weight_matrix[i] = torch.tensor([abs(k - target_item) for k in range(self.class_size)])

        weight_matrix.pow_(self.power)

        # TODO check here, stop here if a nan value and debug it
        reverse_probs = (1 - input_sm).clamp_(min=1e-4)

        log_loss = -torch.log(reverse_probs)
        if torch.sum(torch.isnan(log_loss) == True) > 0:
            print("nan detected in forward pass")

        loss = log_loss * weight_matrix
        loss_sum = torch.sum(loss, dim=1)

        if self.reduction == "mean":
            loss_reduced = torch.mean(loss_sum)
        elif self.reduction == "sum":
            loss_reduced = torch.sum(loss_sum)
        else:
            raise Exception("Undefined reduction type: " + self.reduction)

        return loss_reduced



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction:str='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, probs, target):
        
        probs = F.softmax(probs, dim=1)
        Pt_index = target.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        loss = -((1-Pt) ** self.gamma) * torch.log2(Pt)
        
        return torch.mean(loss)




class FocalLoss_v1(nn.Module):
    def __init__(self, threshold, alphas:list=None, gamma=2, reduction:str='mean', ):
        super().__init__()
        self.alphas = alphas
        self.gamma = gamma
        self.reduction = reduction
        self.threshold = threshold
        
        
    def forward(self, probs, target):
        
        probs = F.softmax(probs, dim=1)
        Pt_index = target.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        gate = torch.where(Pt_index > self.threshold, 1, 0).to(DEVICE)
        Pt_dash = Pt**gate

        loss = -((1-Pt) ** self.gamma) * torch.log2(Pt) + (-torch.log2(Pt_dash**2))
        return torch.mean(loss)

    
    
class CBLoss(nn.Module):
    def __init__(self, train_data_frq:pd.Series, loss_type="softmax", beta=0.9999, gamma=1):
        super(CBLoss, self).__init__()  
        self.ny = train_data_frq
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self, probs, labels): 
        power_term = self.ny.loc[labels.cpu().detach().numpy()]
        denominator = 1.0 - np.power(self.beta, power_term)
        weights = torch.tensor((1.0 - self.beta) / np.array(denominator)).to(DEVICE)
        
        
        probs = F.softmax(probs, dim=1)
        Pt_index = labels.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        
        if self.loss_type == "focal":
            cb_loss = - weights * ((1-Pt) ** self.gamma) * torch.log2(Pt)
        
        elif self.loss_type == "softmax":
            cb_loss = - weights * torch.log2(Pt)
            
        return torch.mean(cb_loss)
    
    
