import numpy as np 
import torch
from loss import to_one_hot


def dice_coe(output, target, eps=1e-5):

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
    dice_coe['avg'] = total_dice_coe
    dice_coe['BG'] = BG_dice_coe
    dice_coe['NET'] = NET_dice_coe
    dice_coe['ED'] = ED_dice_coe
    dice_coe['ET'] = ET_dice_coe
    
    return dice_coe

def softmax_output_dice(output, target):

    ''' generate a dict to show dice coefficient of each label'''

    score = []
    keys = ['tumour core', 'edema', 'enhancing tumour'] # 1, 2, 4

    # dice score of tumour core
    out = output == 1
    tar = target == 1
    score.append(dice_coe(out, tar))

    # dice score of edema
    out = output == 2
    tar = target == 2
    score.append(dice_coe(out, tar))

    # dice score of enhancing tumour
    out = output == 4
    tar = target == 4
    score.append(dice_coe(out, tar))

    dice_dict = {}
    for value,key in zip(score, keys):
        dice_dict[key] = value

    return dice_dict

if __name__ == '__main__':
    import time 
    now= time.time()
    
    yt = np.random.random(size=(2, 3, 3, 3))

    yt = torch.from_numpy(yt)
    yp = np.zeros(shape=(2, 4, 3, 3, 3))
    yp = yp + 1
    yp = torch.from_numpy(yp)
    coe = dice_coe(yp, yt)
    print(coe)
    
    end = time.time() - now
    print(end)
