
'''The tranformer functions mostly based on https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/augment/transforms.py'''

import SimpleITK as sitk 
import os 
import torch
import shutil 
from utils import load_config 
from collections import OrderedDict
from torchvision import transforms, utils
import json
import numpy as np 
import glob
import random
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from visulization import show_imageXY
import skimage
from scipy import ndimage
from functools import reduce 

# GLOBAL_RANDOM_STATE = np.random.RandomState(47)
config_file = 'config.yaml'
cfg = load_config(config_file)
data_root = cfg['PATH']['data_root'] # F:\\TU Delft\\thesis\\sample_images
data_class = cfg['PATH']['data_class'] # MICCAI_BraTS_2018_Data_Training
ROOT = cfg['PATH']['root'] # E:\\VSpythonCode\\Deep_Learning\\U-Net

def train_split(data_dir, split_percentage=0.8, shuffle=True):

    ''' split dataset into training and validation '''
    '''
    data_conetent--
                train_content--
                            HGG
                            LGG
                            merge
                validation_content--
                            HGG
                            LGG
                            merge
    '''

    train_content = {}
    train_content['HGG'] = []
    train_content['LGG'] = []
    validation_content = {}
    validation_content['HGG'] = []
    validation_content['LGG'] = []

    for category in ['HGG', 'LGG']:
        category_dir = os.path.join(data_dir, category)
        
        patient = os.listdir(category_dir)
        total_num_patient = len(patient)
        train_num = int(total_num_patient*split_percentage)

        validation_target_idx = list(range(total_num_patient))
        train_target_idx = np.random.choice(validation_target_idx, train_num, replace=False)

        for idx in train_target_idx:
            patient_dir = os.path.join(category_dir, patient[idx])
            train_content['%s' %category].append(patient_dir)
            validation_target_idx.remove(idx)

        if validation_target_idx:
            for idx in validation_target_idx:
                patient_dir = os.path.join(category_dir, patient[idx])
                validation_content['%s' %category].append(patient_dir)
    
    merged_train = train_content['HGG'] + train_content['LGG']
    merged_validation = validation_content['HGG'] + validation_content['LGG']

    if shuffle:
        random.shuffle(merged_train)
        random.shuffle(merged_validation)

    train_content.update({'merge': merged_train})
    validation_content.update({'merge': merged_validation})


    data_content = {'train': train_content, 'val': validation_content}
    return data_content

def data_obtain(data_conetent, key='train', form='merge'):

    ''' build up a dataframe to access individual modality dir per patient'''
    '''   name           flair          t1          t1ce          t2           seg    
        patient 1    path_to_flair  path_to_t1  path_to_t1ce   path_to_t2   path_to_seg
        patient 2
            .
            .
            .
        patient n
    '''
    assert key in ['train', 'val'], 'key value must be one of "train" and "val"'
    
    dictionary = data_conetent[key]
    merged_dataset = dictionary[form]
    patient_dict = {}
    patient_dict['name'] = []
    patient_dict['flair'] = []
    patient_dict['t1'] = []
    patient_dict['t1ce'] = []
    patient_dict['t2'] = []
    patient_dict['seg'] = []

    for patient in merged_dataset:

        patient_name = os.path.basename(patient)
        flair = os.path.join(patient, '{}_flair.nii.gz'.format(patient_name))
        t1    = os.path.join(patient, '{}_t1.nii.gz'.format(patient_name))
        t1ce  = os.path.join(patient, '{}_t1ce.nii.gz'.format(patient_name))
        t2    = os.path.join(patient, '{}_t2.nii.gz'.format(patient_name))

        if os.path.exists(os.path.join(patient, '{}_seg.nii.gz'.format(patient_name))):
            seg = os.path.join(patient, '{}_seg.nii.gz'.format(patient_name))
        else:
            seg = None

        patient_dict['name'].append(patient_name)
        patient_dict['flair'].append(flair)
        patient_dict['t1'].append(t1)
        patient_dict['t1ce'].append(t1ce)
        patient_dict['t2'].append(t2)
        patient_dict['seg'].append(seg)

    df = pd.DataFrame(patient_dict)
    # print(df)
    
    return df

class RandomRotation:

    ''' Randomly rotate the images within 10 degrees '''

    def __init__(self, max_angle=10, prob=0.1):

        # 90% probability to happen
        self.max_angle = max_angle
        self.axes = [(0,1), (1,2), (0,2)] # rotate along x, z, y, repsectively
        # self.random_state = GLOBAL_RANDOM_STATE
        self.prob = prob

    def __call__(self, sample):

        image, seg = sample['image'], sample['seg']

        rotate_angle = np.random.uniform(-self.max_angle, self.max_angle)
        rotate_axes  = self.axes[np.random.randint(len(self.axes))]

        # The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
        rotated_channels = [ndimage.rotate(image[i, ...], rotate_angle, rotate_axes, reshape=False, order=0, mode='constant', cval=0) for i in range(image.shape[0])]
        rotated_images = np.stack(rotated_channels, axis=0)
        rotated_seg = ndimage.rotate(seg, rotate_angle, rotate_axes, reshape=False, order=0, mode='constant', cval=0)

        return {'image':rotated_images, 'seg':rotated_seg}


class RandomGaussianNoise:

    ''' Randomly add gaussian noise to images '''

    def __init__(self, prob=0.5):

        # 50% probability to happen
        self.prob = prob
        # self.random_state = GLOBAL_RANDOM_STATE

    def __call__(self, sample):
        image, seg = sample['image'], sample['seg']

        # if self.random_state.uniform() > self.prob:
        std = np.random.uniform(0, 0.5)
        noise = np.random.normal(0, std, size=image.shape)
        noised_image = image + noise

        return {'image':noised_image, 'seg':seg}

        
    
class RandomFlip:

    ''' Randomly flip the axes '''

    def __init__(self, prob=0.1):

        # 90% probability to happen
        self.axes = [0, 1, 2]
        self.prob = prob
        # self.random_state = GLOBAL_RANDOM_STATE
    
    def __call__(self, sample):
        image, seg = sample['image'], sample['seg']

        # if self.random_state.uniform() > self.prob:

        for axis in self.axes:
            flipped_image = [np.flip(image[i, ...], axis) for i in range(image.shape[0])]
            flipped_image = np.stack(flipped_image, axis=0)
            flipped_seg = np.flip(seg, axis)

        return {'image':flipped_image, 'seg':flipped_seg}

         


class RandomContrast:
    
    ''' Randomly adjust the intensitiy of image '''

    def __init__(self, mean=0, alpha=[0.5, 1.5], prob=0.2):

        # 80% probability to happen
        assert len(alpha) == 2, 'the lence of alpha should be 2'

        self.alpha = alpha
        self.prob = prob
        self.new_mean = mean
        # self.random_state = GLOBAL_RANDOM_STATE

    def __call__(self, sample):

        image, seg = sample['image'], sample['seg']

        # if self.random_state.uniform() > self.prob:
        alpha = np.random.uniform(self.alpha[0], self.alpha[1])
        channel_means = [image[i].mean() for i in range(image.shape[0])]
        adjusted_image = [self.new_mean + alpha*(image[i]-channel_means[i]) for i in range(image.shape[0])]
        adjusted_image = np.stack(adjusted_image, axis=0)

        return {'image': adjusted_image, 'seg':seg}

        
            
class RandomCrop:

    ''' randomly crop the patch to (crop_size, crop_size, crop_size) in xy plane'''

    def __init__(self, crop_size=98):

        
        if isinstance(crop_size, int):
            self.crop_size = [crop_size]*3
        else:
            self.crop_size = crop_size
    
        # self.random_state = GLOBAL_RANDOM_STATE

    def __call__(self, sample):

        image, seg = sample['image'], sample['seg']

        x, y, z = image.shape[-1], image.shape[-2], image.shape[-3]
        new_x, new_y, new_z = self.crop_size[0], self.crop_size[1], self.crop_size[2]

        size = [(x, new_x), (y, new_y), (z, new_z)]
        pads = [abs(val-new_val) for (val, new_val) in size]
        starts = [np.random.randint(0, pad) for pad in pads]

        
        image = image[:, starts[-1]:starts[-1]+new_z, starts[-2]:starts[-2]+new_y, starts[-3]:starts[-3]+new_x]
        seg = seg[starts[-1]:starts[-1]+new_z, starts[-2]:starts[-2]+new_y, starts[-3]:starts[-3]+new_x]

        return {'image':image, 'seg':seg}
        

class ToTensor:

    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample):
        # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        dtype = torch.FloatTensor
        image, seg = sample['image'], sample['seg']
        
        return {'image':torch.from_numpy(image).type(dtype), 'seg':torch.from_numpy(seg).type(dtype)}

class Normalize:

    '''Z-scoring Normalize the images '''

    def __call__(self, sample):
        image, seg = sample['image'], sample['seg']


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

        return {'image':image, 'seg':seg}

class BraTSDataset(Dataset):

    ''' BraTS18 DataSet '''

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe['name'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient = self.dataframe.iloc[idx]

        modalities = ['flair', 't1', 't1ce', 't2']
        seg = ['seg']
        image  = [sitk.GetArrayFromImage(sitk.ReadImage(patient[modality])) for modality in modalities]
        image = np.stack(image)

        seg = sitk.GetArrayFromImage(sitk.ReadImage(patient['seg']))
        sample = {'image':image, 'seg':seg}

        if self.transform:
            sample = self.transform(sample)

        return sample


class data_loader:

    def __init__(self, 
                data_content, 
                key='train', 
                form='merge', 
                crop_size=98, 
                overlap_size=0, 
                batch_size=2,  
                num_works=8, 
                dataset=BraTSDataset):
        
        '''
        params data_content: dictionary where the image dir are saved
        params key: generate training data or validation data, one of 'train' and 'val'
        params form: one of 'merge', 'HGG' and 'LGG'
        params crop_size: patch size, int or 3d list or tuple
        params overlap_size: cropping overlap between patches, only valid when crop_method is chosen as 'inorder'
        params batch_size: the number of input to be generated in each minibatch
        params num_works: the number of kernels used to load image simultanously, default to 0
        params dataset: instance of pytroch Dataset class
        '''

        assert key in ('train', 'val'), 'key value must be "train" or "val"'
        assert form in ('merge', 'HGG', 'LGG'), 'data form must be "merge", "HGG" or "LGG"'

        self.key = key
        self.batch_size = batch_size
        self.num_works = num_works
        self.dataset = dataset
        self.crop_size = crop_size
        self.overlap_size = overlap_size
        self.form = form
        self.crop_method = RandomCrop(crop_size=self.crop_size)

        self.df = data_obtain(data_content, key=self.key, form=self.form)

        if self.key == 'train':
            self.bratsdata = self.dataset(self.df, 
            transform=transforms.Compose([
                RandomRotation(),
                self.crop_method,
                RandomContrast(),
                RandomGaussianNoise(),
                Normalize(),
                ToTensor()
            ]))

        elif self.key == 'val':
            self.bratsdata = self.dataset(self.df, 
            transform=transforms.Compose([
                self.crop_method,
                Normalize(),
                ToTensor()
            ]))

    def __len__(self):

        return len(self.bratsdata)

    def load(self):

        dataloader = DataLoader(self.bratsdata, 
                                batch_size=self.batch_size, 
                                num_workers=self.num_works, 
                                shuffle=True, 
                                # worker_init_fn=lambda x: np.random.seed()
                                )

        return dataloader


            
if __name__ == '__main__':

    dir = os.path.join(data_root, data_class)
    data_content = train_split(dir)

    df = data_obtain(data_content)

    bratsdata = BraTSDataset(df)
    print(len(bratsdata))

    

    # for i in range(len(bratsdata)):
    #     sample = bratsdata[i]
    #     if i == 0:
    #         print(sample['image'].shape, sample['seg'].shape)
    #     else:
    #         break
    

    # print(len(dataloader))
    # for sample_batch in dataloader:
    #     image, seg = sample_batch['image'], sample_batch['seg']
    #     print(image.shape)
    #     print(np.unique(seg))

    data = data_loader(data_content, crop_size=98, batch_size=2, key='val', form='LGG')
    


    for _ in range(3):
        dataloader = data.load()
        for i, sample_batch in enumerate(dataloader):
            # print(i, sample_batch['image'].size(), sample_batch['seg'].size())
            if i < 1: # show the first three batches
                # for j in range(sample_batch['image'].size()[0]):

                image = sample_batch['image'][0].cpu()
                seg = sample_batch['seg'][0].cpu()
                
                # print('Image average intensities: {}, labels: {}'.format(np.unique(image), np.unique(seg)))
                # show_imageXY(image, seg)
            else:
                break
    
    

