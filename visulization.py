import numpy as np
import torch
import SimpleITK as sitk 
import matplotlib.pyplot as plt 
from torchvision import transforms, utils
from IPython import display
import time

GLOBAL_RANDOM_STATE = np.random.RandomState(47)


def sitk_show(img, title=None, margin=0.05, dpi=40):
    ''' show the 2D images'''
    nda = sitk.GetArrayFromImage(img)
    # (from (H, W, C) to (C, W, H))
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1-2*margin, 1-2*margin])

    # plt.set_cmap('gray')
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)
    
    plt.show()

def show_imageXY(image, seg, transform=None, slice_pos=2, title=None):

    z = image.shape[-3]
    sliceindex = z//slice_pos
   
    fig, axes = plt.subplots(1,5,figsize=(10,10))
    ax = axes.ravel()
    
    for i in range(4):
        ax[i].set_axis_off()
        ax[i].imshow(image[i, sliceindex])
        
    ax[4].set_axis_off()
    ax[4].imshow(seg[sliceindex])
    if title:
        plt.title(title)

    plt.show()
    



# def show_batch_images(sample_batched):
#     image_batch, seg_batch = sample_batched['image'], sample_batched['seg']
#     batch_size = len(image_batch)
#     im_size = image_batch.size(2)
#     grid_border_size = 2

#     grid = utils.make_grid(image_batch[:3])
#     plt.imshow(grid.numpy())

#     for i in range(batch_size):
#         plt.imshow



if __name__ == '__main__':

    flair_dir = 'F:\\TU Delft\\thesis\\sample_images\\MICCAI_BraTS_2018_Data_Training\\HGG\\Brats18_2013_3_1\\Brats18_2013_3_1_flair.nii'
    t1 = 'F:\\TU Delft\\thesis\\sample_images\\MICCAI_BraTS_2018_Data_Training\\HGG\\Brats18_2013_3_1\\Brats18_2013_3_1_t1.nii'
    t1ce = 'F:\\TU Delft\\thesis\\sample_images\\MICCAI_BraTS_2018_Data_Training\\HGG\\Brats18_2013_3_1\\Brats18_2013_3_1_t1ce.nii'
    t2 = 'F:\\TU Delft\\thesis\\sample_images\\MICCAI_BraTS_2018_Data_Training\\HGG\\Brats18_2013_3_1\\Brats18_2013_3_1_t2.nii'
    seg = 'F:\\TU Delft\\thesis\\sample_images\\MICCAI_BraTS_2018_Data_Training\\HGG\\Brats18_2013_3_1\\Brats18_2013_3_1_seg.nii'


    image_flair = sitk.ReadImage(flair_dir)
    image_t1   = sitk.ReadImage(t1)
    image_t1ce = sitk.ReadImage(t1ce)
    image_t2   = sitk.ReadImage(t2)
    image_seg  = sitk.ReadImage(seg)
    print(sitk.GetArrayFromImage(image_seg).shape)

    for i in range(2,5):
        show_imageXY(sitk.GetArrayFromImage(image_seg), slice_pos=i)

    # show orginal image 
    indexSlice = 70
    # sitk_show(sitk.Tile(image_flair[:,:,indexSlice], image_t1[:,:,indexSlice], image_t1ce[:,:,indexSlice], image_t2[:,:,indexSlice], (2,2,0)), title='Original MRI image at z slice %s' %indexSlice)

    # show rotated images
    # numpy_flair = sitk.GetArrayFromImage(image_flair)

    # t = RandomFlip(GLOBAL_RANDOM_STATE)
    # fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    # ax = axes.ravel()

    # # orginal image
    # ax[0].imshow(numpy_flair[70])
    # ax[0].set_axis_off()
    # # rotated images 
    # for i in range(1, 25):
    #     ax[i].set_axis_off()
    #     ax[i].imshow(t(numpy_flair)[70])
    # plt.show()

    # soomth image
    # image_flair_smooth = sitk.CurvatureFlow(image_flair, timeStep=0.125, numberOfIterations=5)
    # image_t1_soomth = sitk.CurvatureFlow(image_t1, timeStep=0.125, numberOfIterations=5)
    # image_t2_soomth = sitk.CurvatureFlow(image_t2, timeStep=0.125, numberOfIterations=5)
    # image_t1ce_soomth = sitk.CurvatureFlow(image_t1ce, timeStep=0.125, numberOfIterations=5)

    # sitk_show(sitk.Tile(image_flair_smooth[:,:,indexSlice], image_t1_soomth[:,:,indexSlice], image_t1ce_soomth[:,:,indexSlice], image_t2_soomth[:,:,indexSlice], (2,2,0)), title='smoothed MRI image at z slice %s' %indexSlice)

        
        
  
