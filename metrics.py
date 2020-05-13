import numpy as np 
import torch
from loss import to_one_hot


def dice_coe(output, target, eps=1e-5):

    '''
    Used for accuracy evaluation during training and validation 
    '''

    target = to_one_hot(target)

    output = output.contiguous().view(output.shape[0], output.shape[1], -1)
    target = target.contiguous().view(target.shape[0], output.shape[1], -1).type_as(output)
    
    num = 2*torch.sum(output*target, dim=-1)
    den = torch.sum(output + target, dim=-1) + eps

    BG_dice_coe = torch.mean(num[:,0]/den[:,0]).numpy()
    NET_dice_coe = torch.mean(num[:,1]/den[:,1]).numpy()
    ED_dice_coe = torch.mean(num[:,2]/den[:,2]).numpy()
    ET_dice_coe = torch.mean(num[:,3]/den[:,3]).numpy()

    total_dice_coe = (NET_dice_coe + ED_dice_coe + ET_dice_coe)/3

    dice_coe = {}
    dice_coe['avg'] = total_dice_coe.item()
    dice_coe['BG'] = BG_dice_coe.item()
    dice_coe['NET'] = NET_dice_coe.item()
    dice_coe['ED'] = ED_dice_coe.item()
    dice_coe['ET'] = ET_dice_coe.item()
    
    return dice_coe


if __name__ == '__main__':
    
    yt = np.random.random(size=(2, 3, 3, 3))
    yt = torch.from_numpy(yt)
    yp = np.zeros(shape=(2, 4, 3, 3, 3))
    yp = yp + 1
    yp = torch.from_numpy(yp)
    coe = dice_coe(yp, yt)
    print(coe)

    
