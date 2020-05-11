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
from utils import crop_index_gen, image_rebuild, image_crop, normalize, inference_output, heatmap_plot_pred
import SimpleITK as sitk
import nibabel as nib 
from metrics import dice_coe
import matplotlib.pyplot as plt 
import seaborn as sns
import time


def validation(trained_net, val_set, criterion, device, batch_size):

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
            test_patient, 
            crop_size=98, 
            overlap_size=0, 
            save_mask=True):

    ''' used for predicting image segmentation after training '''

    modality_dir = os.listdir(test_patient)
    image = []
    for modality in modality_dir:
        if modality != 'Brats18_2013_9_1_seg.nii.gz':
            path = os.path.join(test_patient, modality)
            img = sitk.GetArrayFromImage(sitk.ReadImage(path))
            image.append(img)

        else:
            path = os.path.join(test_patient, modality)
            seg = sitk.ReadImage(path)
            seg_arr = sitk.GetArrayFromImage(seg)
            # print(np.unique(seg_arr))

    image = np.stack(image) # C*D*H*W

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

                # plot heatmap per patch
                in_images = image.detach().cpu().numpy()[:, 0, ...]
                in_segs = np.expand_dims(seg_arr, 0)
                in_pred = preds.detach().cpu().numpy()
                heatmap_plot_pred(image=in_images, mask=in_segs, pred=in_pred, epoch='prediction')

                cropped_image_list[i, ...] = preds.squeeze(0).detach().cpu().numpy()
                pbar.update(1)

    crop_index = crop_info['index_array']
    rebuild_four_channels = image_rebuild(crop_index, cropped_image_list)
    inferenced_mask = inference_output(rebuild_four_channels) 


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

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    plt.savefig(current_time+'result.png')

    mask = sitk.GetImageFromArray(inferenced_mask.astype(np.int16))
    if save_mask:
        mask.CopyInformation(seg)
        sitk.WriteImage(mask, 'test.nii.gz')


if __name__ == '__main__':
    
    config_file = 'config.yaml'
    cfg = load_config(config_file)
    input_modalites = int(cfg['PARAMETERS']['input_modalites'])
    output_channels = int(cfg['PARAMETERS']['output_channels'])
    base_channels = int(cfg['PARAMETERS']['base_channels'])
    patience = int(cfg['PARAMETERS']['patience'])

    data_root = cfg['PATH']['data_root'] # F:\\TU Delft\\thesis\\sample_images
    data_class = cfg['PATH']['data_class'] # MICCAI_BraTS_2018_Data_Training
    ROOT = cfg['PATH']['root'] # E:\\VSpythonCode\\Deep_Learning\\U-Net 
    best_dir = cfg['PATH']['best_model_path']
    best_model_dir = os.path.join(ROOT, best_dir)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    
    # load best trained model
    net = init_U_Net(input_modalites, output_channels, base_channels)
    net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=patience)

    ckp_path = os.path.join(best_model_dir, 'best_model.pth')
    net, optimizer,scheduler, start_epoch, min_loss, start_loss = load_ckp(ckp_path, net, optimizer, scheduler)

    # predict
    test_patient = 'F:\\TU Delft\\thesis\\sample_images\\MICCAI_BraTS_2018_Data_Training\\LGG\\Brats18_2013_9_1'
    predict(net, test_patient, save_mask=True)
    

    '''
    #######################################################
    test_patient = 'F:\\TU Delft\\thesis\\sample_images\\MICCAI_BraTS_2018_Data_Training\\LGG\\Brats18_2013_6_1'
    
    modality_dir = os.listdir(test_patient)
    image = []
    for modality in modality_dir:
        if modality != 'Brats18_2013_6_1_seg.nii.gz':
            path = os.path.join(test_patient, modality)
            img = sitk.GetArrayFromImage(sitk.ReadImage(path))
            image.append(img)

        else:
            path = os.path.join(test_patient, modality)
            seg = sitk.ReadImage(path)
            seg_arr = sitk.GetArrayFromImage(seg)

    plt.imshow(seg_arr[80])
    plt.show()
    

    image = np.stack(image) 
    test_seg = np.expand_dims(seg_arr, 0) # (1, 155, 240, 240)
    
    test_seg_onehot = to_one_hot(torch.from_numpy(test_seg).cuda())
    test_seg_onehot = test_seg_onehot.squeeze(0).numpy()
    # print(test_seg_onehot.shape)

    image_shape = test_seg.shape[-3:]
    crop_info = crop_index_gen(image_shape=image_shape, crop_size=100, overlap_size=0)
    # print(crop_info['index_array'])
    
    image_patches = image_crop(test_seg_onehot, crop_info, norm=False, ToTensor=False)
    # print(image_patches.shape)

    cropped_image_list = np.zeros_like(image_patches)
    # print(cropped_image_list.shape)

    
    for i, image in enumerate(image_patches):

        cropped_image_list[i, ...] = image
    
    # print(cropped_image_list.shape)
    print(np.unique(cropped_image_list))
    
    crop_index = crop_info['index_array']
    rebuild_four_channels = image_rebuild(crop_index, cropped_image_list)
    # print(rebuild_four_channels.shape)
    print(np.unique(rebuild_four_channels))

    
    inferenced_mask = inference_output(rebuild_four_channels)
    print(np.unique(inferenced_mask))
    plt.imshow(inferenced_mask[80])
    plt.show()


    mask = sitk.GetImageFromArray(inferenced_mask.astype(np.int16))
    
    mask.CopyInformation(seg)
    sitk.WriteImage(mask, 'test_seg.nii.gz')
    '''
    
    
    
    
    
  





    
    

    
    


