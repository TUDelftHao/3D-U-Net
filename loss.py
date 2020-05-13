import torch.nn.functional as F 
import torch 
import torch.nn as nn 
import numpy as np 
from utils import load_config


config_file = 'config.yaml'
config = load_config(config_file)
labels = config['PARAMETERS']['labels']

def to_one_hot(seg, labels=labels):

    assert len(seg.shape) == 4, 'segmentation shape must be in NxDxHxW'
    assert labels is not None

    seg[seg == 4] = 3
    seg = seg.cpu().unsqueeze(1).long()
    n_channels = len(labels)
    shape = np.array(seg.shape)
    shape[1] = n_channels
    one_hot_seg = torch.zeros(tuple(shape), dtype=torch.long)
    one_hot_seg = one_hot_seg.scatter_(1, seg, 1) #dim, index, src

    return one_hot_seg


class DiceLoss(nn.Module):
    def __init__(self, labels, epsilon=1e-5, ignore_index=None):
        super(DiceLoss, self).__init__()

        # smooth factor
        self.epsilon = epsilon
        self.labels = labels
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        targets = to_one_hot(targets, self.labels).to(logits.device)
        targets.requires_grad = False

        batch_size, channels = logits.size(0), logits.size(1)

        tot_loss = 0

        for i in range(channels):
            if not self.ignore_index or i != int(self.ignore_index):
                logit = logits[:, i].view(batch_size, -1)
                target = targets[:, i].view(batch_size, -1).type_as(logit)
                intersection = (logit*target).sum(-1)
                dice_score = 2. * intersection / ((logit + target).sum(-1) + self.epsilon)
                loss = torch.mean(1. - dice_score)

                tot_loss += loss

        return tot_loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, labels, ignore_index=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()

        self.labels = labels
        self.reduction = reduction
        self.ignore_index = ignore_index

        self.loss_f = nn.BCELoss(reduction=self.reduction)

    def forward(self, logits, targets):
        targets = to_one_hot(targets, self.labels).to(logits.device)
        targets.requires_grad = False

        batch_size, channels = logits.size(0), logits.size(1)

        BCE_loss = 0
        
        for i in range(channels):
            if not self.ignore_index or i != int(self.ignore_index):
                loss = self.loss_f(logits[:, i].view(batch_size, -1), targets[:, i].view(batch_size, -1).type_as(logits))
            
                BCE_loss += loss
        
        return BCE_loss


class FocalLoss(nn.Module):
    def __init__(self, labels, ignore_index=None, gamma=2):
        super(FocalLoss, self).__init__()

        self.labels = labels
        self.ignore_index = ignore_index
        self.gamma = gamma
        self._alpha = 0.25

    def forward(self, logits, targets):
        targets = to_one_hot(targets, self.labels).to(logits.device)
        targets.requires_grad = False
        
        batch_size, channels = logits.size(0), logits.size(1)

        tot_loss = 0
        for i in range(channels):
            if not self.ignore_index or i != int(self.ignore_index):
                pred = logits[:, i].view(batch_size, -1)
                target = targets[:, i].view(batch_size, -1).type_as(pred)

                loss = -target * torch.log(pred)
                loss = self._alpha * loss * (1 - pred) ** self.gamma
            
                tot_loss += torch.sum(loss)

        return tot_loss

        
class Dice_CE(nn.Module):
    def __init__(
        self, 
        labels,  
        ignore_index=None, 
        reduction='mean', 
        epsilon=1e-5,
    ):
        super(Dice_CE, self).__init__()

        self.labels = labels
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.epsilon = epsilon

        self.dl_fn = DiceLoss(self.labels, epsilon=self.epsilon, ignore_index=self.ignore_index)
        self.ce_fn = CrossEntropyLoss(self.labels, ignore_index=self.ignore_index, reduction=self.reduction)

    def forward(self, logits, targets):

        dl_loss = self.dl_fn(logits, targets)
        ce_loss = self.ce_fn(logits, targets)

        return dl_loss + ce_loss

class Dice_FL(nn.Module):
    def __init__(
        self,
        labels, 
        gamma=2,
        reduction='mean', 
        ignore_index=None, 
        epsilon=1e-5, 
    ):
        super(Dice_FL, self).__init__()

        self.labels = labels
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.epsilon = epsilon

        self.dl_fn = DiceLoss(self.labels, epsilon=self.epsilon, ignore_index=self.ignore_index)
        self.fl_fn = FocalLoss(self.labels, ignore_index=self.ignore_index, gamma=self.gamma)

    def forward(self, logits, targets):
        
        dl_loss = self.dl_fn(logits, targets)
        fl_loss = self.fl_fn(logits, targets)

        return dl_loss + fl_loss



if __name__ == "__main__":

    labels = [0, 1, 2, 4]
    yt = np.random.random(size=(2, 4, 3, 3, 3))
    yt = torch.from_numpy(yt).cuda()
    yp = np.zeros(shape=(2, 3, 3, 3))
    yp = yp + 1
    yp = torch.from_numpy(yp).cuda()
    # print(yp)
    
    # dl = DiceLoss(labels=labels)
    # dl = CrossEntropyLoss(labels=labels)
    dl = FocalLoss(labels=labels, gamma=2)
    # dl = Dice_CE(labels=labels)
    # dl = Dice_FL(labels=labels, gamma=3)
    print(dl(yt, yp).item())



    
    
