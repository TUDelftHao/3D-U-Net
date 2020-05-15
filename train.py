from loss import DiceLoss, CrossEntropyLoss, FocalLoss, Dice_CE, Dice_FL
from model import init_U_Net
from dataset_conversion import BraTSDataset, data_loader, train_split
from utils import load_config, save_ckp, load_ckp, logfile, loss_plot, heatmap_plot
from metrics import dice_coe
import torch.optim as optim
import torch 
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
from test import validation
import json
import numpy as np 
import shutil
import warnings
import time
import argparse
from apex import amp 

def train(args):

    torch.cuda.manual_seed(1)
    torch.manual_seed(1)

    # user defined
    model_name = args.model_name 
    model_loss_fn = args.loss_fn

    config_file = 'config.yaml'

    config = load_config(config_file)
    data_root = config['PATH']['data_root'] 
    labels = config['PARAMETERS']['labels']
    root_path = config['PATH']['root']
    model_dir = config['PATH']['model_path']
    best_dir = config['PATH']['best_model_path']

    data_class = config['PATH']['data_class']
    input_modalites = int(config['PARAMETERS']['input_modalites'])
    output_channels = int(config['PARAMETERS']['output_channels'])
    base_channel = int(config['PARAMETERS']['base_channels'])
    crop_size = int(config['PARAMETERS']['crop_size'])
    batch_size = int(config['PARAMETERS']['batch_size'])
    epochs = int(config['PARAMETERS']['epoch'])
    is_best = bool(config['PARAMETERS']['is_best'])
    is_resume = bool(config['PARAMETERS']['resume'])
    patience = int(config['PARAMETERS']['patience'])
    ignore_idx = int(config['PARAMETERS']['ignore_index'])
    early_stop_patience = int(config['PARAMETERS']['early_stop_patience'])
    
    # build up dirs
    model_path = os.path.join(root_path, model_dir)
    best_path = os.path.join(root_path, best_dir)
    intermidiate_data_save = os.path.join(root_path, 'train_data', model_name)
    train_info_file = os.path.join(intermidiate_data_save, '{}_train_info.json'.format(model_name))
    log_path = os.path.join(root_path, 'logfiles')

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(best_path):
        os.mkdir(best_path)
    if not os.path.exists(intermidiate_data_save):
        os.makedirs(intermidiate_data_save)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_name = model_name + '_' + config['PATH']['log_file']
    logger = logfile(os.path.join(log_path, log_name))
    logger.info('Dataset is loading ...')
    # split dataset
    dir_ = os.path.join(data_root, data_class)
    data_content = train_split(dir_)

    # load training set and validation set 
    train_set = data_loader(data_content=data_content, 
                            key='train', 
                            form='LGG',
                            crop_size=crop_size,
                            batch_size=batch_size, 
                            num_works=8
                            )
    n_train = len(train_set)
    train_loader = train_set.load()

    val_set = data_loader(data_content=data_content,
                            key='val', 
                            form='LGG',
                            crop_size=crop_size, 
                            batch_size=batch_size, 
                            num_works=8
                            )

    logger.info('Dataset loading finished!')
    
    n_val = len(val_set)
    nb_batches = np.ceil(n_train/batch_size)
    n_total = n_train + n_val
    logger.info('{} images will be used in total, {} for trainning and {} for validation'.format(n_total, n_train, n_val))
    
    net = init_U_Net(input_modalites, output_channels, base_channel)
    

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.device_count() > 1:
        logger.info('{} GPUs available.'.format(torch.cuda.device_count()))
        net = nn.DataParallel(net)

    net.to(device)

    if model_loss_fn == 'Dice':
        criterion = DiceLoss(labels=labels, ignore_index=ignore_idx)
    elif model_loss_fn == 'CrossEntropy':
        criterion = CrossEntropyLoss(labels=labels, ignore_index=ignore_idx)
    elif model_loss_fn == 'FocalLoss':
        criterion = FocalLoss(labels=labels, ignore_index=ignore_idx)
    elif model_loss_fn == 'Dice_CE':
        criterion = Dice_CE(labels=labels, ignore_index=ignore_idx)
    elif model_loss_fn == 'Dice_FL':
        criterion = Dice_FL(labels=labels, ignore_index=ignore_idx)
    else:
        raise NotImplementedError()

    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=patience)

    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    min_loss = float('Inf')
    early_stop_count = 0
    global_step = 0
    start_epoch = 0
    start_loss = 0
    train_info = {'train_loss':[], 
                'val_loss':[],
                'BG_acc':[],
                'NET_acc':[],
                'ED_acc':[],
                'ET_acc':[]}

    if is_resume:
        try: 
            ckp_path = os.path.join(model_dir, '{}_model_ckp.pth.tar'.format(model_name))
            net, optimizer, scheduler, start_epoch, min_loss, start_loss = load_ckp(ckp_path, net, optimizer, scheduler)

            # open previous training records
            with open(train_info_file) as f:
                train_info = json.load(f)

            logger.info('Training loss from last time is {}'.format(start_loss) + '\n' + 'Mininum training loss from last time is {}'.format(min_loss))

        except:
            logger.warning('No checkpoint available, strat training from scratch')


    # start training
    for epoch in range(start_epoch, epochs):

        # setup to train mode
        net.train()
        running_loss = 0
        dice_coeff_bg = 0
        dice_coeff_net = 0
        dice_coeff_ed = 0
        dice_coeff_et = 0
        
        logger.info('Training epoch {} will begin'.format(epoch+1))

        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}', unit='patch') as pbar:

            for i, data in enumerate(train_loader, 0):
                images, segs = data['image'].to(device), data['seg'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = net(images)

                loss = criterion(outputs, segs)
                # loss.backward()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                
                optimizer.step()

                # save the output at the begining of each epoch to visulize it
                if i == 0:
                    in_images = images.detach().cpu().numpy()[:, 0, ...]
                    in_segs = segs.detach().cpu().numpy()
                    in_pred = outputs.detach().cpu().numpy()
                    heatmap_plot(image=in_images, mask=in_segs, pred=in_pred, name=model_name, epoch=epoch+1)
    
                running_loss += loss.detach().item()
                dice_score = dice_coe(outputs.detach().cpu(), segs.detach().cpu())
                dice_coeff_bg += dice_score['BG']
                dice_coeff_ed += dice_score['ED']
                dice_coeff_et += dice_score['ET']
                dice_coeff_net += dice_score['NET']

                # show progress bar
                pbar.set_postfix(**{'Training loss': loss.detach().item(), 'Training (avg) accuracy': dice_score['avg']})
                pbar.update(images.shape[0])

                global_step += 1
                if global_step % nb_batches == 0:
                    # validate 
                    net.eval()
                    val_loss, val_acc = validation(net, val_set, criterion, device, batch_size)
                   

        train_info['train_loss'].append(running_loss/nb_batches)
        train_info['val_loss'].append(val_loss)
        train_info['BG_acc'].append(dice_coeff_bg/nb_batches)
        train_info['NET_acc'].append(dice_coeff_net/nb_batches)
        train_info['ED_acc'].append(dice_coeff_ed/nb_batches)
        train_info['ET_acc'].append(dice_coeff_et/nb_batches)

        # save bast trained model
        scheduler.step(running_loss / nb_batches)

        if min_loss > val_loss:
            min_loss = val_loss
            is_best = True
            early_stop_count = 0
        else:
            is_best = False
            early_stop_count += 1

        state = {
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':running_loss / nb_batches,
            'min_loss': min_loss
        }
        verbose = save_ckp(state, is_best, early_stop_count=early_stop_count, early_stop_patience=early_stop_patience, save_model_dir=model_path, best_dir=best_path, name=model_name)
            
        logger.info('The average training loss for this epoch is {}'.format(running_loss / (np.ceil(n_train/batch_size))))
        logger.info('Validation dice loss: {}; Validation (avg) accuracy: {}'.format(val_loss, val_acc))
        logger.info('The best validation loss till now is {}'.format(min_loss))

        # save the training info every epoch
        logger.info('Writing the training info into file ...')
        with open(train_info_file, 'w') as fp:
            json.dump(train_info, fp)
    
        loss_plot(train_info_file, name=model_name)

        if verbose:
            logger.info('The validation loss has not improved for {} epochs, training will stop here.'.format(early_stop_patience))
            break

    logger.info('finish training!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '--model_name', default='baseline', type=str, help='model name')
    parser.add_argument('-loss', '--loss_fn', default='CrossEntropy', type=str, help='loss function, options: Dice, CrossEntropy, FocalLoss, Dice_CE, Dice_FL')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
