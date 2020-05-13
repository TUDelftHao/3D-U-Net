from loss import DiceLoss, to_one_hot
from model import init_U_Net
from dataset_conversion import BraTSDataset, data_loader
from utils import load_config, load_ckp
import torch.optim as optim
import torch.nn.functional as F
import torch 
import os
from tqdm import tqdm 
import numpy as np
from utils import crop_index_gen, image_rebuild, image_crop, normalize, inference_output
import SimpleITK as sitk
import nibabel as nib 
from metrics import dice_coe
import matplotlib.pyplot as plt 
import seaborn as sns
import time
import argparse


def validation(trained_net, 
                val_set, 
                criterion, 
                device,
                batch_size):

    '''
    used for valuation during training phase

    params trained_net: trained U-net
    params val_set: validation dataset 
    params criterion: loss function
    params device: cpu or gpu
    '''

    n_val = len(val_set)
    val_loader = val_set.load()

    tot = 0 
    acc = 0

    with tqdm(total=n_val, desc='Validation round', unit='patch', leave=False) as pbar:
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                images, segs = sample['image'].to(device=device), sample['seg'].to(device=device)

                preds = trained_net(images)
                val_loss = criterion(preds, segs)
                dice_score = dice_coe(preds.detach().cpu(), segs.detach().cpu())

                tot += val_loss.detach().item() 
                acc += dice_score['avg']

                pbar.set_postfix(**{'validation loss (images)': val_loss.detach().item(), 'val_acc':dice_score['avg']})
                pbar.update(images.shape[0])

    return tot/(np.ceil(n_val/batch_size)), acc/(np.ceil(n_val/batch_size))


def predict(trained_net, 
            model_name,
            test_patient, 
            root,
            crop_size=98, 
            overlap_size=0, 
            save_mask=True):

    ''' used for predicting image segmentation after training '''

    # load test image
    patient_name = os.path.basename(test_patient)
    modality_dir = os.listdir(test_patient)
    image = []
    for modality in modality_dir:
        if modality != patient_name + '_seg.nii.gz':
            path = os.path.join(test_patient, modality)
            img = sitk.GetArrayFromImage(sitk.ReadImage(path))
            image.append(img)
        else:
            path = os.path.join(test_patient, modality)
            seg = sitk.ReadImage(path)
            seg_arr = sitk.GetArrayFromImage(seg)

    image = np.stack(image) # C*D*H*W

    # model inference
    trained_net.eval()
    image_shape = image.shape[-3:]
    crop_info = crop_index_gen(image_shape=image_shape, crop_size=crop_size, overlap_size=overlap_size)

    image_patches = image_crop(image, crop_info, norm=True, ToTensor=True)

    cropped_image_list = np.zeros_like(image_patches.cpu().numpy())

    with torch.no_grad():
        with tqdm(total=len(cropped_image_list), desc='inference test image', unit='patch') as pbar:
            for i, image in enumerate(image_patches):
                image = image.unsqueeze(dim=0)
                preds = trained_net(image)

                cropped_image_list[i, ...] = preds.squeeze(0).detach().cpu().numpy()
                pbar.update(1)

    crop_index = crop_info['index_array']
    rebuild_four_channels = image_rebuild(crop_index, cropped_image_list)
    inferenced_mask = inference_output(rebuild_four_channels) 

    # calcualte DSC
    target = torch.from_numpy(seg_arr).unsqueeze(0)
    pred = torch.from_numpy(inferenced_mask).unsqueeze(0)
    pred = to_one_hot(pred)
    dsc = dice_coe(pred, target)
    print('DSC by label of this image is: ', dsc)

    # plot predicted segmentation
    plt.figure(figsize=(20, 10))
    ground_truth = seg_arr[image_shape[0]//2]
    predicted = inferenced_mask[image_shape[0]//2]
    image_list = [ground_truth, predicted]

    subtitles = ['ground truth', 'predicted']
    plt.subplots_adjust(wspace=0.3)

    for i in range(1,3):
        ax = plt.subplot(1,2,i)
        ax.set_title(subtitles[i-1])
        sns.heatmap(image_list[i-1], vmin=0, vmax=4, xticklabels=False, yticklabels=False, square=True, cmap='coolwarm', cbar=True)

    # save prediction
    save_path = os.path.join(root, 'prediction_results', patient_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    mask = sitk.GetImageFromArray(inferenced_mask.astype(np.int16))
    if save_mask:
        plt.savefig(os.path.join(save_path, '{}_2D_prediction_{}.png'.format(patient_name, model_name)))

        mask.CopyInformation(seg)
        sitk.WriteImage(mask, os.path.join(save_path, '{}_2D_prediction_{}.nii.gz'.format(patient_name, model_name)))


def predict_use(args):

    model_name = args.model_name
    patient_path = args.patient_path

    config_file = 'config.yaml'
    cfg = load_config(config_file)
    input_modalites = int(cfg['PARAMETERS']['input_modalites'])
    output_channels = int(cfg['PARAMETERS']['output_channels'])
    base_channels = int(cfg['PARAMETERS']['base_channels'])
    patience = int(cfg['PARAMETERS']['patience'])

    ROOT = cfg['PATH']['root'] 
    best_dir = cfg['PATH']['best_model_path']
    best_model_dir = os.path.join(ROOT, best_dir)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load best trained model
    net = init_U_Net(input_modalites, output_channels, base_channels)
    net.to(device)
    
    optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=patience)
    ckp_path = os.path.join(best_model_dir, model_name + '_best_model.pth.tar')
    net, _, _, _, _, _ = load_ckp(ckp_path, net, optimizer, scheduler)

    # predict
    predict(net, model_name, patient_path, ROOT, save_mask=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--model_name', default='baseline_local', type=str, help='which model to be used')
    parser.add_argument('-p', '--patient_path', type=str, help='patient dir')
    args = parser.parse_args()

    predict_use(args)

if __name__ == '__main__':
    
    main()
    

    



    
    

    
    


