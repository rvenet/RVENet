import csv
import os
import sys
import json
import shutil
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from RVEnet.data_loader import TaskTypes, Normalize
from RVEnet.data_loader import Task
from RVEnet.data_loader import EchoDataset
from RVEnet.data_loader import Rescale, RandomCrop, ToTensor, ExpandFrameTensor, RandomRotation, HorizontalFlip
from RVEnet.data_loader import preprocess_EFNet, preprocess_r2plus1d

from RVEnet.models.EFNet import EFNet
from RVEnet.train_eval.training import train_calssification_model, train_regression_model
from RVEnet.utils.calculate_dicom_set_indicators import get_indicators
from RVEnet.utils.checkpoint_handler import load_ckp, load_checkpoint_weights
from RVEnet.utils.get_last_experiment_id import get_exp_id
from RVEnet.utils.fix_seed import fix_seed

############################### Set general parameters ###############################

def run_training(parameter_json_path):

    with open(parameter_json_path) as params_file:
        parameters = json.load(params_file)

    seed = parameters['seed']
    fix_seed(seed)

    DICOM_database_path = parameters['DICOM_database_path']
    path_for_otuput_weigthts = parameters['path_for_otuput_weigthts']

    num_of_classes = parameters['num_of_classes']

    binary_classification_threshold = parameters['binary_classification_threshold']
    base_experiment_folder = parameters['base_experiment_folder']
    experiment_summary = os.path.join(base_experiment_folder, parameters['summary_txt'])
    dataset_indicators_path = parameters['dataset_indicators_path']

    continue_experiment_with_id = parameters['continue_experiment_with_id']

    ############################### Set training parameters ###############################

    batch_size = parameters['batch_size']
    DICOM_frame_nbr = parameters['DICOM_frame_nbr']
    input_image_size = parameters['input_image_size']
    start_epoch = parameters['start_epoch']
    is_balancing = parameters['is_balancing']
    regression_loss_function = parameters['regression_loss_function']
    classification_thresholds = parameters['classification_thresholds']
    task_type = parameters['task_type']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_properties(device))

    if continue_experiment_with_id:
        experiment_id = continue_experiment_with_id
    else:
        experiment_id = get_exp_id(base_experiment_folder)

    if not os.path.exists(os.path.join(base_experiment_folder, experiment_id)):
        os.makedirs(os.path.join(base_experiment_folder, experiment_id))

    # Save training params
    head, tail = os.path.split(parameter_json_path)
    shutil.copyfile(parameter_json_path, os.path.join(base_experiment_folder, experiment_id, tail))

    if task_type == "classification":
        task = Task(TaskTypes.CLASSIFICATION, num_of_classes, classification_thresholds)
    elif task_type == "binary_classification":
        task = Task(TaskTypes.BINARY_CLASSIFICATION, 2, classification_thresholds)
    elif task_type == "regression":
        task = Task(TaskTypes.REGRESSION)
    else:
        print('Task type not recognized!')

    if task.task_type == TaskTypes.CLASSIFICATION:
        criterion = nn.CrossEntropyLoss()
    elif task.task_type == TaskTypes.BINARY_CLASSIFICATION:
        criterion = nn.BCEWithLogitsLoss()
    elif task.task_type == TaskTypes.REGRESSION:
        
        if regression_loss_function == "MSE":
            criterion = nn.MSELoss()
        elif regression_loss_function == "MAE":
            criterion = nn.L1Loss()
        elif regression_loss_function == "Huber":
            criterion = nn.HuberLoss(delta=1.35)
        else:
            print("unknown loss function.")
            exit()
    else:
        print("No task")
        exit()

    ############################### Set up augmentation transforms ###############################

    is_horizontal_flip = parameters['is_horizontal_flip']
    horizontal_flip_probability = parameters['horizontal_flip_probability']
    is_random_rotation = parameters['is_random_rotation']
    random_rotation_degree_from = parameters['random_rotation_degree_from']
    random_rotation_degree_to = parameters['random_rotation_degree_to']
    is_random_crop = parameters['is_random_crop']
    random_crop_size = parameters['random_crop_size']
    is_dataset_normalization = parameters['is_dataset_normalization']
    include_binary_mask = parameters['include_binary_mask']

    train_transforms = []
    validation_transforms = []

    train_transforms.append(Rescale(input_image_size))
    validation_transforms.append(Rescale(input_image_size))

    if is_random_rotation:
        train_transforms.append(RandomRotation(random_rotation_degree_from, random_rotation_degree_to))
        train_transforms.append(Rescale(input_image_size))

    if is_horizontal_flip:
        train_transforms.append(HorizontalFlip(horizontal_flip_probability))

    if is_random_crop:
        train_transforms.append(RandomCrop(random_crop_size))
        train_transforms.append(Rescale(input_image_size))

    if is_dataset_normalization:

        average, std = get_indicators(DICOM_database_path, dataset_indicators_path)
        print('Average: {}, Std: {}'.format(average, std))

        train_transforms.append(Normalize(average, std))
        validation_transforms.append(Normalize(average, std))

    train_transforms.append(ToTensor())
    validation_transforms.append(ToTensor())

    train_transforms.append(ExpandFrameTensor(include_binary_mask))
    validation_transforms.append(ExpandFrameTensor(include_binary_mask))

    data_transforms = {
        'train': transforms.Compose(train_transforms),
        'validation': transforms.Compose(validation_transforms)
    }

    ############################### Set up cross validation or simple training ###############################

    is_cross_validation = parameters['is_cross_validation']
    cross_validation_json_folder = parameters["cross_validation_json_folder"]

    pretrained_model = parameters['pretrained_model']
    is_pretrained = parameters['is_pretrained']
    optimizer_type = parameters['optimizer_type']
    learning_rate = parameters['learning_rate']
    scheduler_type = parameters['scheduler_type']
    step_size_for_StepLR_scheduler = parameters['step_size_for_StepLR_scheduler']
    gamma = parameters['gamma']

    momentum_for_SGD = parameters['momentum_for_SGD']
    lambda_for_lambda_scheduler = parameters['lambda_for_lambda_scheduler']

    base_lr = parameters['base_lr']
    max_lr = parameters['max_lr']
    step_size_up = parameters['step_size_up']
    mode = parameters['mode']

    dropout = parameters['dropout']
    is_attention = parameters['is_attention_layers']
    attention_feature_reduction = parameters["attention_feature_reduction"]
    activation_function = parameters['activation_function']

    if is_cross_validation==True:
        cv_json_folder = os.path.join(DICOM_database_path,cross_validation_json_folder)
        folds = int(len(os.listdir(cv_json_folder))/2)
        print("Number of folds: {}".format(folds))
    else:
        folds = 1

    for fold in range(folds):

        ############################### Set up model and optimizer ###############################

        # r2plus1d_18 architecture
        if pretrained_model == "r2plus1d_18":
            model = models.video.r2plus1d_18(num_classes=num_of_classes)
            setattr(model, "task", task)  
        else:
        # EFNet architecture variants
            if pretrained_model == "VGG":
                model = models.vgg16(pretrained=is_pretrained)
                features = model.features  # nn.Sequential(*list(model.features.children())[:-1])
            elif pretrained_model == "ResNext":
                model = models.resnext50_32x4d(pretrained=is_pretrained)
                features = nn.Sequential(*list(model.children())[:-2])
            elif pretrained_model == "ShuffleNet":
                model = models.shufflenet_v2_x1_0(pretrained=is_pretrained)
                features = nn.Sequential(*list(model.children())[:-2])
            elif pretrained_model == "ConvNeXt_tiny":
                model = models.convnext_tiny(pretrained=is_pretrained)
                features = model.features
            elif pretrained_model == "ConvNeXt_base":
                model = models.convnext_base(pretrained=is_pretrained)
                features = model.features
            elif pretrained_model == "ConvNeXt_large":
                model = models.convnext_large(pretrained=is_pretrained)
                features = model.features
        
            model = EFNet(features, batch_size, DICOM_frame_nbr, input_image_size, activation_function=activation_function,
                                task=task, dropout_prob=dropout,
                                self_attention=is_attention, self_attention_feature_reduction=attention_feature_reduction)

        model = model.to(device)

        if optimizer_type == 'Adam':
            optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'SGD':
            optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_for_SGD)
        else:
            print('Optimizer type not recognized!')


        # Decay LR by a factor of 0.1 every 7 epochs
        if scheduler_type == "StepLR":
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size_for_StepLR_scheduler, gamma=gamma)
        elif scheduler_type == "LambdaLR":
            lambda1 = lambda epoch: lambda_for_lambda_scheduler ** epoch
            exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer_ft, lr_lambda=lambda1)
        elif scheduler_type == "CyclicLR":
            exp_lr_scheduler = lr_scheduler.CyclicLR(optimizer_ft, base_lr=base_lr,
                                                    max_lr=max_lr,
                                                    step_size_up=step_size_up,
                                                    mode=mode, cycle_momentum=False)


        ############################### Set datasets and dataloaders ###############################

        train_json_path = parameters['train_json_path']
        validation_json_path = parameters['validation_json_path']
        dataloader_shuffle = parameters['dataloader_shuffle']
        dataloader_drop_last = parameters['dataloader_drop_last']

        if is_cross_validation==True:

            print("Cross validation - fold {}".format(fold) )

            experiment_folder = os.path.join(base_experiment_folder, experiment_id, "fold_{}".format(fold))

            if not os.path.exists(experiment_folder):
                os.makedirs(experiment_folder)

            image_datasets = {'train': EchoDataset(os.path.join(cv_json_folder, "label_file_train_20frames_fold{}.json".format(fold)),
                                                DICOM_database_path, DICOM_frame_nbr, task, is_balancing, data_transforms['train']),
                            'validation': EchoDataset(os.path.join(cv_json_folder, "label_file_validation_20frames_fold{}.json".format(fold)),
                                                DICOM_database_path, DICOM_frame_nbr, task, False,
                                                data_transforms['validation'])}

        else:
        # no cross validation

            experiment_folder = os.path.join(base_experiment_folder, experiment_id)

            if not os.path.exists(experiment_folder):
                os.makedirs(experiment_folder)

            image_datasets = {'train': EchoDataset(os.path.join(DICOM_database_path, train_json_path),
                                                DICOM_database_path, DICOM_frame_nbr, task, is_balancing, data_transforms['train']),
                            'validation': EchoDataset(os.path.join(DICOM_database_path, validation_json_path),
                                                DICOM_database_path, DICOM_frame_nbr, task, False,
                                                data_transforms['validation'])}

        gen = torch.Generator()

        dataloaders = {phase: torch.utils.data.DataLoader(image_datasets[phase], batch_size=batch_size,
                                                        shuffle=dataloader_shuffle, num_workers=4,
                                                        generator=gen, drop_last=dataloader_drop_last)
                    for phase in ['train', 'validation']}

        dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in ['train', 'validation']}
        print(dataset_sizes)

        ############################### Train the model ###############################

        load_checkpoint_weight_path = parameters['load_checkpoint_weight_path']
        num_of_epochs = parameters['num_of_epochs']
        checkpoints_folder_name = parameters['checkpoints_folder_name']
        regression_model_weights = parameters['regression_model_weights']
        regression_confusion_matrix_bins = parameters['regression_confusion_matrix_bins']

        # Create TensorBoard writer object
        tensorboard_writer = SummaryWriter(os.path.join(experiment_folder))

        # Save model architecture to the tensorboard log file
        dataiter = iter(dataloaders['train'])
        batch = next(dataiter)
       
        input_tensor = batch["input_tensor"]
        if pretrained_model == "r2plus1d_18":
            input_tensor = preprocess_r2plus1d(input_tensor)
        else:
            input_tensor = preprocess_EFNet(input_tensor)
        input = input_tensor.to(device, dtype=torch.float)

        tensorboard_writer.add_graph(model, input)
        tensorboard_writer.flush()

        # Check if weight file should be loaded
        if load_checkpoint_weight_path is not None:
            model = load_checkpoint_weights(load_checkpoint_weight_path,model)

        # Check if experiment should be continued
        if continue_experiment_with_id is not None:
            checkpoints_folder_to_load_data = os.path.join(experiment_folder, checkpoints_folder_name)
            if len(os.listdir(checkpoints_folder_to_load_data)) > 0:
                model, optimizer, start_epoch, valid_loss_min = load_ckp(checkpoints_folder_to_load_data, model, optimizer_ft, start_epoch)
        else:
            start_epoch = 0

        if task.task_type == TaskTypes.CLASSIFICATION:
            model_ft, last_validation_loss = train_calssification_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                                                                        dataloaders, tensorboard_writer, experiment_folder, 
                                                                        pretrained_model, start_epoch, num_epochs=num_of_epochs, seed=seed)
        else:
            model_ft, last_validation_loss = train_regression_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                                                                    dataloaders, tensorboard_writer, experiment_folder, 
                                                                    pretrained_model, start_epoch, num_epochs=num_of_epochs, 
                                                                    confusion_matrix_bins=regression_confusion_matrix_bins, seed=seed)


        if not os.path.exists(path_for_otuput_weigthts):
            os.makedirs(path_for_otuput_weigthts)

        torch.save(model_ft.state_dict(), os.path.join(path_for_otuput_weigthts, regression_model_weights))
        tensorboard_writer.close()


if __name__ == '__main__':
    parameter_json_path = sys.argv[1]


