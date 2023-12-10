import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from RVEnet.data_loader import preprocess_EFNet, preprocess_r2plus1d
from RVEnet.utils.checkpoint_handler import save_ckp
from RVEnet.utils.EF_calulation_helper_functions import quantize_EF
from RVEnet.utils.fix_seed import fix_seed

from sklearn.metrics import r2_score


def save_confusion_matrix(conf_mat: np.ndarray, output_path: str):
    np.savetxt(output_path, conf_mat, delimiter='\t', fmt='%.0d')


def create_log_folders(experiment_folder: str):
    checkpoints_folder = os.path.join(experiment_folder, 'checkpoints')

    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    confusion_folder = os.path.join(experiment_folder, 'confusion_matrices')

    if not os.path.exists(confusion_folder):
        os.makedirs(confusion_folder)

    return checkpoints_folder, confusion_folder


def visualize_validation_loss(loss_list):
    plt.plot(loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


def train_calssification_model(model: torch.nn.Module, criterion: torch.nn.modules.loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, 
                               dataloaders: dict, tensorboard_writer: torch.utils.tensorboard.writer,
                               experiment_folder: str, pretrained_model: str, start_epoch: int, num_epochs: int=25, verbose: bool=False, seed: int=0):
    since = time.time()

    checkpoints_folder, confusion_folder = create_log_folders(experiment_folder)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    validation_loss_list = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in ['train', 'validation']}

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:

            fix_seed(seed)

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
                confusion_matrix = np.zeros((model.task.output_nbr, model.task.output_nbr))

            running_loss = 0.0
            running_corrects = 0

            data_sample_counter = 0

            # Iterate over data.
            for batch in dataloaders[phase]:
                data_sample_counter += dataloaders[phase].batch_size
                input_tensor = batch["input_tensor"]

                if pretrained_model == "r2plus1d_18":
                    input_tensor = preprocess_r2plus1d(input_tensor)
                else:
                    input_tensor = preprocess_EFNet(input_tensor)

                EF = batch["EF"]

                input = input_tensor.to(device, dtype=torch.float)
                labels = EF.to(device, dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if verbose:
                        print("outputs: " + str(outputs))
                        print("labels: " + str(labels))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase == 'validation':
                    # add predictions to confusion matrix
                    pred_np = preds.detach().cpu().numpy().astype(int)
                    labels_np = EF.numpy().astype(int)
                    for pred_idx in range(len(pred_np)):
                        confusion_matrix[labels_np[pred_idx], pred_np[pred_idx]] += 1

                        # statistics
                running_loss += loss.item() * dataloaders[phase].batch_size
                running_corrects += torch.sum(preds == labels.data)

                # Save and update learning rate
            tensorboard_writer.add_scalar('learning rate',
                                          optimizer.param_groups[0]['lr'],
                                          epoch + 1)

            if phase == 'train':
                scheduler.step()

            if phase == 'validation':
                save_confusion_matrix(confusion_matrix, os.path.join(confusion_folder,
                                                                     "conf_matrix" + str(epoch + 1) + '.txt'))
                if model.task.output_nbr == 2:
                    sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
                    specifisity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])                

                    recall = sensitivity
                    precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])

                    f1_score = (2*precision*recall)/(precision+recall)

                    tensorboard_writer.add_scalar('val_specifisity',
                                            specifisity,
                                            epoch+1)

                    tensorboard_writer.add_scalar('val_sensitivity',
                                            sensitivity,
                                            epoch+1)

                    tensorboard_writer.add_scalar('val_recall',
                                            recall,
                                            epoch+1)

                    tensorboard_writer.add_scalar('val_precision',
                                            precision,
                                            epoch+1)
                    
                    tensorboard_writer.add_scalar('val_f1_score',
                                            f1_score,
                                            epoch+1)


            epoch_loss = running_loss / data_sample_counter
            epoch_acc = running_corrects.double() / data_sample_counter

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            tensorboard_writer.add_scalar('{} loss'.format(phase),
                            epoch_loss,
                            epoch + 1)

            tensorboard_writer.add_scalar('{} accuracy'.format(phase),
                            epoch_acc,
                            epoch+1)

            tensorboard_writer.flush()

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                validation_loss_list.append(epoch_loss)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': epoch_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, os.path.join(checkpoints_folder, 'checkpoint_' + str(epoch + 1) + '.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    visualize_validation_loss(validation_loss_list)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss

def train_regression_model(model: torch.nn.Module, criterion: torch.nn.modules.loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, 
                               dataloaders: dict, tensorboard_writer: torch.utils.tensorboard.writer, experiment_folder: str, pretrained_model: str, 
                               start_epoch: int, confusion_matrix_bins: int=5, num_epochs: int=25, verbose: bool=False, seed: int=0):
    
    criterion_l1 = nn.L1Loss()

    since = time.time()

    checkpoints_folder, confusion_folder = create_log_folders(experiment_folder)

    best_model_wts = copy.deepcopy(model.state_dict())
    least_error = float("inf")

    validation_loss_list = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in ['train', 'validation']}

    EF_min = dataloaders['train'].dataset.EF_min
    EF_max = dataloaders['train'].dataset.EF_max

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:

            fix_seed(seed)

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
                confusion_matrix = np.zeros((confusion_matrix_bins, confusion_matrix_bins))
                predicted_EFs = []
                groundtruth_EFs = []

            running_loss = 0.0
            running_loss_l1 = 0.0

            data_sample_counter = 0

            
            # Iterate over data.
            for batch in dataloaders[phase]:
                data_sample_counter += dataloaders[phase].batch_size
                input_tensor = batch["input_tensor"]

                if pretrained_model == "r2plus1d_18":
                    input_tensor = preprocess_r2plus1d(input_tensor)
                else:
                    input_tensor = preprocess_EFNet(input_tensor)

                EF = batch["EF"]

                input = input_tensor.to(device, dtype=torch.float)
                labels = EF.to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input)
                    outputs = outputs.squeeze()

                    if verbose:
                        print("outputs: " + str(outputs))
                        print("labels: " + str(labels))

                    loss = criterion(outputs, labels)
                    loss_l1 = criterion_l1(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                if phase == 'validation':
                    # add predictions to confusion matrix
                    out_np = outputs.detach().cpu().numpy().astype(float)
                    labels_np = EF.numpy().astype(int)


                    for pred_idx in range(len(out_np)):
                        predicted_EFs.append(out_np[pred_idx])
                        groundtruth_EFs.append(labels_np[pred_idx])

                        quantized_GT_EF = quantize_EF(labels_np[pred_idx],EF_max=EF_max,EF_min=EF_min,nbr_of_bins=confusion_matrix_bins)
                        quantized_predicted_EF = quantize_EF(out_np[pred_idx],EF_max=EF_max,EF_min=EF_min,nbr_of_bins=confusion_matrix_bins )
                        confusion_matrix[quantized_GT_EF, quantized_predicted_EF] += 1


                # statistics
                running_loss += loss.item() * dataloaders[phase].batch_size
                running_loss_l1 += loss_l1.item() * dataloaders[phase].batch_size

            # Save and update learning rate
            tensorboard_writer.add_scalar('learning rate',
                                        optimizer.param_groups[0]['lr'],
                                        epoch + 1)
            if phase == 'train':
                scheduler.step()

            if phase == 'validation':
                save_confusion_matrix(confusion_matrix, os.path.join(confusion_folder,
                                                                     "conf_matrix" + str(epoch + 1) + '.txt'))

                r2 = r2_score(groundtruth_EFs, predicted_EFs)
                tensorboard_writer.add_scalar('{} R2'.format(phase),
                                          r2,
                                          epoch + 1)

            epoch_loss = running_loss / data_sample_counter
            epoch_loss_l1 = running_loss_l1 / data_sample_counter

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            print('{} L1 Loss: {:.4f}'.format(
                phase, epoch_loss_l1))

            tensorboard_writer.add_scalar('{} loss'.format(phase),
                                          epoch_loss,
                                          epoch + 1)

            tensorboard_writer.add_scalar('{} loss_l1'.format(phase),
                                          epoch_loss_l1,
                                          epoch + 1)

            tensorboard_writer.flush()

            # deep copy the model
            if phase == 'validation' and epoch_loss < least_error:
                least_error = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                validation_loss_list.append(epoch_loss)

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': epoch_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, os.path.join(checkpoints_folder, 'checkpoint_' + str(epoch + 1) + '.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Least error: {:4f}'.format(least_error))
    print('validation_loss_list: ', validation_loss_list)

    visualize_validation_loss(validation_loss_list)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss
