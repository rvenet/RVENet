import os
import re

import torch
from os import listdir
from os.path import isfile, join


def save_ckp(state: dict, checkpoint_path: str):
    """
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)

def load_checkpoint_weights(checkpoint_path: str, model: torch.nn.Module):
    
    model_dict = model.state_dict()

    checkpoint = torch.load(checkpoint_path)
    pretrained_dict = checkpoint['state_dict']

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # if 'FC2.weight' in pretrained_dict.keys() and 'FC2.weight' in model_dict.keys():
    #     pretrained_dict['FC2.weight'] = model_dict['FC2.weight']
    # if 'FC2.bias' in pretrained_dict.keys() and 'FC2.bias' in model_dict.keys():
    #     pretrained_dict['FC2.bias'] = model_dict['FC2.bias']

    # model_dict.update(pretrained_dict) 
    # model.load_state_dict(pretrained_dict)

    for layer_key in pretrained_dict.keys():
        if "features" in layer_key:
            model_dict[layer_key] = pretrained_dict[layer_key]

    model.load_state_dict(model_dict)

    return model

def load_ckp(checkpoint_fpath: str, model: torch.nn.Module, optimizer: torch.optim, start_epoch: int):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    only_files = [f for f in listdir(checkpoint_fpath) if isfile(join(checkpoint_fpath, f))]

    last_epoch = -1
    if start_epoch:
        last_epoch = start_epoch
    else:
        for f in only_files:
            epoch_nbr = int(re.split('[_.]', f)[1])
            if epoch_nbr > last_epoch:
                last_epoch = epoch_nbr

    # load check point
    checkpoint = torch.load(os.path.join(checkpoint_fpath, 'checkpoint_' + str(last_epoch) + '.pt'))
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min
