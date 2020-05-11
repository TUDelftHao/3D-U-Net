import torch.nn.functional as F 
import torch 
import torch.nn as nn 
import numpy as np 
from utils import load_config


# config_file = 'U-Net\\config.yaml'
config_file = 'config.yaml'

config = load_config(config_file)
labels = config['PARAMETERS']['labels']

def to_one_hot(seg, labels=labels):

    assert len(seg.shape) == 4, 'segmentation shape must be in NxDxHxW'
    assert labels is not None
    assert isinstance(labels, list), 'lables must be in list and cannot be empty'

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    seg[seg == 4] = 3
    seg = seg.cpu().unsqueeze(1).long()
    n_channels = len(labels)
    shape = np.array(seg.shape)
    shape[1] = n_channels
    one_hot_seg = torch.zeros(tuple(shape), dtype=torch.long)

    one_hot_seg = one_hot_seg.scatter_(1, seg, 1) #dim, index, src

    # print('one_hot shape: ', one_hot_seg.shape)

    return one_hot_seg

class BinaryDiceLoss(nn.Module):

    ''' Dice loss for binary class '''

    def __init__(self, smooth=1, p=1):
        super(BinaryDiceLoss, self).__init__()

        self.smooth = torch.Tensor([smooth])
        self.smooth.requires_grad = False 
        self.p = p

    def forward(self, predict, target):

        assert predict.shape == target.shape, 'prediction and target should share the same shape'

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = predict.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=-1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=-1) 
        
        return 1 - torch.mean(2*num / den)


class DiceLoss(nn.Module):
    def __init__(self, labels, epsilon=1e-5):
        super(DiceLoss, self).__init__()

        # smooth factor
        self.epsilon = epsilon
        self.labels = labels

    def forward(self, logits, targets):

        targets = to_one_hot(targets, self.labels).to(logits.device)
        targets.requires_grad = False

        batch_size = targets.size(0)
        tot_loss = 0

        for i in range(logits.shape[1]):
            logit = logits[:, i].view(batch_size, -1).type(torch.FloatTensor)
            target = targets[:, i].view(batch_size, -1).type(torch.FloatTensor)
            intersection = (logit*target).sum(-1)
            dice_score = 2. * intersection / ((logit + target).sum(-1) + self.epsilon)
            loss = torch.mean(1. - dice_score)
            tot_loss += loss

        return tot_loss/logits.shape[1]


if __name__ == "__main__":


    yt = np.random.random(size=(2, 3, 3, 3))
    # print(yt)
    yt = torch.from_numpy(yt).cuda()
    yp = np.zeros(shape=(2, 4, 3, 3, 3))
    yp = yp + 1
    yp = torch.from_numpy(yp).cuda()
    # print(yp)
    dl = DiceLoss(labels=[0, 1, 2, 4])
    print(dl(yp, yt).item())



    
    
