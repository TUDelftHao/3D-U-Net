import torch 
import numpy as np 
import torch.nn as nn 
import yaml 
import SimpleITK as sitk 
import shutil 
import os
from collections import OrderedDict 
import matplotlib
matplotlib.use('Agg')
import logging
import json
import matplotlib.pyplot as plt 
import time
import seaborn as sns
import warnings

def logfile(path, level='debug'):
    
    if os.path.exists(path):
        os.remove(path)
        
    # set up log    file

    logger = logging.getLogger(__name__)
    if level == 'debug':
        logger.setLevel(level=logging.DEBUG)   
    elif level == 'info':
        logger.setLevel(level=logging.INFO) 
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # FileHandler
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def image_rebuild(crop_index_array, cropped_image_list):

    '''
    concatenate all the pateches and rebuild a new image with the same size as orignal one, the final image has four channels in one-hot format 
    
    params crop_index_array: crop start corner index, saved in order
    params cropped_image_list: predict patches with the same order of crop index
    
    '''

    if isinstance(cropped_image_list, list):
        cropped_image_list = np.asarray(cropped_image_list)

    assert crop_index_array.shape[0] == cropped_image_list.shape[0], 'The number of index array should equal image number'

    cropped_image_shape = cropped_image_list[0].shape
    target_imagesize_z = max(np.unique(crop_index_array[:,0])) + cropped_image_shape[1]
    target_imagesize_y = max(np.unique(crop_index_array[:,1])) + cropped_image_shape[2]
    target_imagesize_x = max(np.unique(crop_index_array[:,2])) + cropped_image_shape[3]

    target_imagechannel = cropped_image_shape[0]
    total_mask = np.zeros((target_imagechannel, target_imagesize_z, target_imagesize_y, target_imagesize_x), dtype='float_')


    for i in range(target_imagechannel):

        overlap_mask = np.zeros((target_imagesize_z, target_imagesize_y, target_imagesize_x), dtype='float_') # used to count the overlap 

        for (crop_index, cropped_image) in zip(crop_index_array, cropped_image_list):

            image_channel = cropped_image[i] # (D,H,W)
            
            total_mask[i, crop_index[0]:crop_index[0]+cropped_image_shape[1], crop_index[1]:crop_index[1]+cropped_image_shape[2], crop_index[2]:crop_index[2]+cropped_image_shape[3]] += image_channel

            overlap_mask[crop_index[0]:crop_index[0]+cropped_image_shape[1], crop_index[1]:crop_index[1]+cropped_image_shape[2], crop_index[2]:crop_index[2]+cropped_image_shape[3]] += 1


        total_mask_channel = total_mask[i, ...]
        total_mask_channel /= overlap_mask

        total_mask_channel = np.where(total_mask_channel>0.8, 1, 0)
        total_mask[i, ...] = total_mask_channel


    return total_mask

def inference_output(output_image):

    shape = output_image.shape
    channel = shape[0]
    inferenced_image = np.zeros(shape[-3:])

    for i in range(1, channel):
        if i == 1:
            inferenced_image[output_image[i]==1] = 1
        if i == 2:
            inferenced_image[output_image[i]==1] = 2
        if i == 3:
            inferenced_image[output_image[i]==1] = 4
    return inferenced_image

def crop_index_gen(image_shape, crop_size=98, overlap_size=30):

    ''' return a dict containing the sorted cropping start index, crop size and patch number '''

    if overlap_size is None:
        overlap_size = 0

    assert overlap_size >= 0, 'overlap must be a non-negative value'
    assert overlap_size < crop_size, 'overlap must be smaller to crop size'

    if isinstance(crop_size, int):
        crop_size = np.asarray([crop_size]*len(image_shape))

    if isinstance(overlap_size, int):
        overlap_size = np.asarray([overlap_size]*len(image_shape))

    num_block_per_dim = (image_shape - overlap_size) // (crop_size - overlap_size)

    index_per_axis_dict = {}
    for j, num in enumerate(num_block_per_dim):
        
        initial_point_dim = [i*(crop_size[j]-overlap_size[j]) for i in range(num)]
        initial_point_dim.append(image_shape[j]-crop_size[j])
        index_per_axis_dict[j] = initial_point_dim

    index_axis_z = index_per_axis_dict[0]
    index_axis_y = index_per_axis_dict[1]
    index_axis_x = index_per_axis_dict[2]
    index_array = []

    for val_z in index_axis_z:
        for val_y in index_axis_y:
            for val_x in index_axis_x:
                index_array.append([val_z, val_y, val_x])
    
    index_array = np.asarray(index_array).reshape(-1,3)

    crop_info = {}
    crop_info['index_array'] = index_array
    crop_info['crop_size'] = crop_size
    crop_info['crop_number'] = index_array.shape[0]

    return crop_info

def image_crop(image, crop_info, norm=False, ToTensor=False):
    
    '''return a list of cropped image patches according to crop index'''
    '''return: patches * channels * D * H * W''' 

    assert image.ndim == 4 # C*W*H*D

    crop_index, crop_size, crop_num = crop_info['index_array'], crop_info['crop_size'], crop_info['crop_number']
    cropped_images = np.zeros((crop_num, image.shape[0], crop_size[0], crop_size[1], crop_size[2]))

    for i, index in enumerate(crop_index):
        cp_img = image.copy()
        img = cp_img[:, index[0]:index[0]+crop_size[0], index[1]:index[1]+crop_size[1], index[2]:index[2]+crop_size[2]]

        if norm:
            img = normalize(img)
        cropped_images[i, ...] = img

    if ToTensor:
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        cropped_images = torch.from_numpy(cropped_images).type(dtype)
    return cropped_images

def normalize(image):

    assert len(image.shape) == 4, 'image must be in form of C*D*W*H'

    for i in range(image.shape[0]):
        img = np.asarray(image[i], dtype='float_')
        nonzero_mask = img[np.nonzero(img)]

        if len(nonzero_mask) != 0:
            img -= nonzero_mask.mean()
            img /= (nonzero_mask.std()+1e-5) 
        else:
            img -= img.mean()
            img /= img.std()+1e-5

        image[i, ...] = img  

    return image

#---------------------------------------------------

def load_config(file_path):
    return yaml.safe_load(open(file_path, 'r'))

def save_ckp(state, is_best, save_model_dir, best_dir, name):
    f_path = os.path.join(save_model_dir, '{}_model_ckp.pth.tar'.format(name))
    torch.save(state, f_path)
    if is_best:
        best_path = os.path.join(best_dir, '{}_best_model.pth.tar'.format(name))
        shutil.copyfile(f_path, best_path)


def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['min_loss'], checkpoint['loss']


def loss_plot(train_info_file, name):

    #plt.cla()

    train_info = load_config(train_info_file)
    train_loss = train_info['train_loss']
    val_loss = train_info['val_loss']
    BG_acc = train_info['BG_acc']
    NET_acc = train_info['NET_acc']
    ED_acc = train_info['ED_acc']
    ET_acc = train_info['ET_acc']

    epoch = len(train_loss)
    x_axis = np.arange(epoch)

    figure = plt.figure(1, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(121)
    plt.title('loss')
    plt.plot(x_axis, train_loss, lw=3, color='black', label='training loss')
    plt.plot(x_axis, val_loss, lw=3, color='green', label='validation loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.subplot(122)
    plt.title('accuracy')
    plt.plot(x_axis, BG_acc, color='red', label='back ground accuracy')
    plt.plot(x_axis, NET_acc,  color='skyblue', label='NET accuracy')
    plt.plot(x_axis, ED_acc, color='blue', label='ED accuracy')
    plt.plot(x_axis, ET_acc, color='yellow', label='ET accuracy')
    plt.legend() 

    plt.xlabel('epochs')
    plt.ylabel('acc')

    # time_stamp = os.path.basename(train_info_file).split('.')[0]
    plt.savefig(os.path.join(os.path.dirname(train_info_file), '{}_loss_acc_plot.png'.format(name)))
    # plt.show()

def heatmap_plot_pred(image, mask, pred, epoch, name, save=True):

    warnings.filterwarnings("ignore")
    # plt.cla()

    current_path = os.getcwd()
    plot_dir = os.path.join(current_path, 'pre_temp_plot', name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    size = image.shape[2]
 
    ax1 = plt.subplot(1,6,1)
    ax1.axis('off')
    ax1.imshow(image[0, size//2])
    
    ax2 = plt.subplot(1,6,2)
    sns.heatmap(mask[0, size//2], vmin=0, vmax=3, xticklabels=False, yticklabels=False, square=True, cmap='coolwarm', cbar=False)
    
    ax3 = plt.subplot(1,6,3)
    sns.heatmap(pred[0, 0, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)
    
    ax4 = plt.subplot(1,6,4)
    sns.heatmap(pred[0, 1, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)
    
    ax5 = plt.subplot(1,6,5)
    sns.heatmap(pred[0, 2, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)
    
    ax6 = plt.subplot(1,6,6)
    sns.heatmap(pred[0, 3, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)

    if save:
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        plot_name = os.path.join(plot_dir, '{}-epoch-{}_{}.png'.format(name, epoch, current_time))
        plt.savefig(plot_name)

    # plt.show()

def heatmap_plot(image, mask, pred, epoch, name, save=True):
    # image, mask, pred should be numpy.array()
    warnings.filterwarnings("ignore")
    # plt.cla()

    current_path = os.getcwd()
    plot_dir = os.path.join(current_path, 'temp_plot', name)
    if not os.path.exists(plot_dir):
        # os.mkdir(plot_dir)
        os.makedirs(plot_dir)

    size = image.shape[2]
 
    ax1 = plt.subplot(2,6,1)
    ax1.axis('off')
    ax1.imshow(image[0, size//2])
    
    ax2 = plt.subplot(2,6,2)
    sns.heatmap(mask[0, size//2], vmin=0, vmax=3, xticklabels=False, yticklabels=False, square=True, cmap='coolwarm', cbar=False)
    
    ax3 = plt.subplot(2,6,3)
    sns.heatmap(pred[0, 0, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)
    
    ax4 = plt.subplot(2,6,4)
    sns.heatmap(pred[0, 1, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)
    
    ax5 = plt.subplot(2,6,5)
    sns.heatmap(pred[0, 2, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)
    
    ax6 = plt.subplot(2,6,6)
    sns.heatmap(pred[0, 3, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)
    


    ax7 = plt.subplot(2,6,7)
    plt.axis('off')
    plt.imshow(image[1, size//2])

    ax8 = plt.subplot(2,6,8)
    sns.heatmap(mask[1, size//2], vmin=0, vmax=3, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)

    ax9 = plt.subplot(2,6,9)
    sns.heatmap(pred[1, 0, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)

    ax10 = plt.subplot(2,6,10)
    sns.heatmap(pred[1, 1, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)

    ax11 = plt.subplot(2,6,11)
    sns.heatmap(pred[1, 2, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)

    ax12 = plt.subplot(2,6,12)
    sns.heatmap(pred[1, 3, size//2], vmin=0, vmax=1, xticklabels=False, yticklabels=False,square=True, cmap='coolwarm', cbar=False)

    if save:
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        plot_name = os.path.join(plot_dir, '{}-epoch-{}_{}.png'.format(name, epoch, current_time))
    
        plt.savefig(plot_name)


# ------------------------------------------------------------------------

def count_params(model):

    ''' print number of trainable parameters and its size of the model'''

    num_of_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model {} : params number {}, params size: {:4f}M'.format(model._get_name(), num_of_param, num_of_param*4/1000/1000))


def seg_conversion(InputFile, OutputFile):
    
    ''' convert segmentation file to our usecase '''

    image = sitk.ReadImage(InputFile)
    image_np = sitk.GetArrayFromImage(image)
    seg_new = np.zeros_like(image_np)
    seg_new[image_np==4] = 3
    seg_new[image_np==2] = 1
    seg_new[image_np==1] = 2
    img_new = sitk.GetImageFromArray(seg_new)
    img_new.CopyInformation(image)
    sitk.WriteImage(img_new, OutputFile)

def modalites_conversion():

    ''' copy original images to a new folder and convert their name '''

    # build up new target dataset
    target_dataset = os.path.join(ROOT, 'dataset'+'\\'+data_class)
    if not os.path.exists(target_dataset):
        os.makedirs(target_dataset)

    # build up sub dir to store images and labels, respectively

    ImageFolder = os.path.join(target_dataset, 'Images')
    if not os.path.exists(ImageFolder):
        os.mkdir(ImageFolder)
    LableFolder = os.path.join(target_dataset, 'Labels')
    if not os.path.exists(LableFolder):
        os.mkdir(LableFolder)

    # copy and rename original dataset to target dataset
    task = os.path.join(data_root, data_class)

    patient_newNames = []
    for sub_task in ['LGG', 'HGG']:
        sub_task_path = os.path.join(task, sub_task)

        print('{} folder is in process ...'.format(sub_task))

        for patient in os.listdir(sub_task_path):
            patient_path = os.path.join(sub_task_path, patient)
            patient_newName = sub_task + '_' + patient
            patient_newNames.append(patient_newName)

            modalities = ['_t1.nii.gz', '_t1ce.nii.gz', '_flair.nii.gz', '_t2.nii.gz']

            for modality in modalities:
                image_mode = os.path.join(patient_path, patient+modality)
                
                assert os.path.isfile(image_mode), '%s' %patient

                shutil.copyfile(image_mode, os.path.join(ImageFolder, patient_newName+modality))

            original_seg = os.path.join(patient_path, patient+'_seg.nii.gz')
            target_seg = os.path.join(LableFolder, patient_newName+'_seg.nii.gz')
            seg_conversion(original_seg, target_seg)
    
    json_dict = OrderedDict()
    json_dict['name'] = 'BraTS_2018'
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing",
    }
    json_dict['numTraining'] = len(patient_newNames)

    with open('%s\\data.json' %target_dataset, 'w') as f:
        json.dump(json_dict, f)

    print('Data transferring finished!')



if __name__ == '__main__':
    cfg = load_config('config.yaml')
    data_root = cfg['PATH']['data_root'] # F:\\TU Delft\\thesis\\sample_images
    data_class = cfg['PATH']['data_class'] # MICCAI_BraTS_2018_Data_Training
    ROOT = cfg['PATH']['root'] # E:\\VSpythonCode\\Deep_Learning\\U-Net

    
    # train_info = {'train_loss':np.random.random(10), 
    #             'val_loss':np.random.random(10),
    #             'BG_acc':np.random.random(10),
    #             'NET_acc':np.random.random(10),
    #             'ED_acc':np.random.random(10),
    #             'ET_acc':np.random.random(10)}
    # file_path = 'E:\\VSpythonCode\\Deep_Learning\\U-Net\\modeldata\\train_info_2020-04-21-21-22-43.json'
    
    # loss_plot(file_path)  
    
   

