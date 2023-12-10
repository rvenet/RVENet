import os
import sys
import json
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time

from RVEnet.utils.heartcycle_averaging import average_heart_cycles
from RVEnet.utils.calculate_dicom_set_indicators import get_indicators
from RVEnet.data_loader import TaskTypes, Normalize
from RVEnet.data_loader import Rescale, ToTensor, ExpandFrameTensor
from RVEnet.data_loader import EchoDataset

from torchvision import models, transforms
from RVEnet.models.EFNet import EFNet

from RVEnet.data_loader import preprocess_EFNet, preprocess_r2plus1d

from RVEnet.data_loader import Task
from sklearn.metrics import r2_score
import scipy

import matplotlib.pyplot as plt


def create_dataset(evaluation_parameters: dict, model_parameters: dict, task:Task) -> EchoDataset:
    DICOM_database_path = evaluation_parameters['dataset_path']
    dataset_json = evaluation_parameters['dataset_json_path']
    training_data_path = model_parameters['DICOM_database_path']

    dataset_indicators_path = model_parameters['dataset_indicators_path']
    is_dataset_normalization = model_parameters['is_dataset_normalization']

    DICOM_frame_nbr = model_parameters['DICOM_frame_nbr']
    input_image_size = model_parameters['input_image_size']
    include_binary_mask = model_parameters['include_binary_mask']

    if is_dataset_normalization:
        average, std = get_indicators(training_data_path, dataset_indicators_path)

    data_transforms = [Rescale(input_image_size)]
    if is_dataset_normalization:
        data_transforms.append(Normalize(average, std))
    data_transforms.append(ToTensor())
    data_transforms.append(ExpandFrameTensor(include_binary_mask))

    data_transforms = transforms.Compose(data_transforms)

    dataset = EchoDataset(dataset_json, DICOM_database_path, DICOM_frame_nbr, task, False, data_transforms,
                          return_heart_cycle_ids=True)

    return dataset


def load_model(parameters: dict, model_path: str, task: Task) -> torch.nn.Module:
    batch_size = parameters['batch_size']
    DICOM_frame_nbr = parameters['DICOM_frame_nbr']
    input_image_size = parameters['input_image_size']
    dropout = parameters['dropout']

    activation_function = parameters['activation_function']
    self_attention = parameters['is_attention_layers']
    self_attention_feature_reduction = parameters['attention_feature_reduction']

    pretrained_model = parameters['pretrained_model']
    is_pretrained = parameters['is_pretrained']

    task_type = parameters["task_type"]

    # r2plus1d_18 architecture
    if pretrained_model == "r2plus1d_18":
        model = models.video.r2plus1d_18(num_classes=task.output_nbr)
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
    
        model = EFNet(features, batch_size, DICOM_frame_nbr, input_image_size, task=task, dropout_prob=dropout,
                activation_function=activation_function, self_attention=self_attention, self_attention_feature_reduction=self_attention_feature_reduction)

    setattr(model, "task", task) 
    checkpoint = torch.load(model_path)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])

    return model


def run_model(parameters: dict, model: torch.nn.Module, dataset: EchoDataset, output_file_name: str):
    gen = torch.Generator()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=parameters['dataloader_shuffle'], num_workers=4,
                                             generator=gen, drop_last=parameters['dataloader_drop_last'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    softmax_activation = nn.Softmax(dim=1)

    predictions = {}

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    model_param_nbr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("nbr of model parameters: {}".format(model_param_nbr))
    measurements = []
    image_idx = 0

    for image_information, heart_cycle_id in tqdm(dataloader):

        if parameters['pretrained_model']=="r2plus1d_18":
            merged_input_tensor = image_information["input_tensor"]
            merged_input_tensor = preprocess_r2plus1d(merged_input_tensor)
        else:
             merged_input_tensor = preprocess_EFNet(image_information["input_tensor"])
       
        input = merged_input_tensor.to(device, dtype=torch.float)

        with torch.set_grad_enabled(False):

            torch.cuda.synchronize()
            before_inference = time.perf_counter()
            outputs = model(input)
            torch.cuda.synchronize()
            after_inference = time.perf_counter()

            image_idx+=1
            if image_idx >100 and image_idx <200:
                measurements.append(after_inference-before_inference)

            if image_idx==200:
                measurements = np.asarray(measurements)
                avg_inference_time = np.sum(measurements)/len(measurements)
                print("Inference time after GPU warm up (100 runs) averaged on {} runs is: {}".format(len(measurements), avg_inference_time))

            if parameters["task_type"] == "regression":
                regression_output = outputs.detach().cpu().numpy().squeeze().tolist()
                predictions[heart_cycle_id[0]] = {"predicted_EF": regression_output}
            else:
                softmax_outputs = softmax_activation(outputs)

                _, preds = torch.max(outputs, 1)

                pred_np = int(preds.detach().cpu().numpy())
                softmax_outputs = softmax_outputs.detach().cpu().numpy().squeeze().tolist()

                predictions[heart_cycle_id[0]] = {"predicted_class": pred_np,
                                                  "predicted_confidence_values": softmax_outputs}

    # write json to file
    with open(output_file_name, "w") as f:
        json.dump(predictions, f)


def run_model_on_dataset(model_parameters: dict, evaluation_parameters: dict, model_path: str, output_filename: str,
                         only_model_weights=False):
    
    classification_thresholds = model_parameters['classification_thresholds']

    if model_parameters['task_type'] == "classification":
        task = Task(TaskTypes.CLASSIFICATION, model_parameters['num_of_classes'], classification_thresholds)
    elif model_parameters['task_type'] == "binary_classification":
        task = Task(TaskTypes.BINARY_CLASSIFICATION, 2, classification_thresholds)
    elif model_parameters['task_type'] == "regression":
        task = Task(TaskTypes.REGRESSION, output_nbr=None)
    else:
        print('Task type not recognized!')

    # create dataset
    dataset = create_dataset(evaluation_parameters, model_parameters, task)
    print("Torch dataset was created successfully.")

    # read/create model
    if only_model_weights == True:
        model = load_model(model_parameters, model_path, task)
    else:
        model = torch.load(model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Model was loaded successfully.")

    print("Run model on dataset.")

    run_model(model_parameters, model, dataset, output_filename)


def EF_to_class(parameters: dict, EF_value: float, EF_min: int=10, EF_max: int=80) -> int:
    classification_thresholds = parameters['classification_thresholds']
    num_of_classes = parameters['num_of_classes']

    EF_class = None
    if str(num_of_classes) in classification_thresholds:
        ranges = classification_thresholds[str(num_of_classes)]
        EF_class = [i for i in range(len(ranges) - 1) if ranges[i] < EF_value <= ranges[i + 1]][0]
    else:
        EF_range = EF_max - EF_min + 1
        EF_step = EF_range / num_of_classes
        EF_class = int((EF_value - EF_min) / EF_step)

    if EF_class >= num_of_classes:
        EF_class = num_of_classes - 1

    return EF_class


def groundtruth_creator(parameters: dict, validation_json_path: str, output_filename:str, EF_max: int=80, EF_min: int=10):
    evaluation_task_type = parameters["evaluation_task_type"]
    
    f = open(validation_json_path)
    base_json = json.load(f)
    groundtruth_json = {}
    for patient in base_json:
        if base_json[patient]['EF'] == '':
            print('Patient {} skipped, since EF was empty'.format(patient))
            continue

        EF_value = float(base_json[patient]['EF'])

        if evaluation_task_type != "regression":
            EF_class = EF_to_class(parameters, EF_value)

            if EF_class is None:
                raise ValueError('Problem with EF value')

        for dicom_dict in base_json[patient]['dicoms']:
            for heartcycle_key in dicom_dict['frame_indexes']:
                new_id = '{}__{}__{}'.format(patient, dicom_dict['dicom_id'], heartcycle_key)
                if evaluation_task_type == "regression":
                    groundtruth_json[new_id] = {'gt_EF': EF_value}
                else:
                    groundtruth_json[new_id] = {'gt_class': EF_class}

    with open(output_filename, 'w') as outfile:
        json.dump(groundtruth_json, outfile)


def evaluation(model_parameters: dict, evaluation_parameters: dict, ground_truth_json_path: str, predictions_json_path: str,
               save_raw_comparison: bool=True):
    num_of_classes = evaluation_parameters['num_of_classes']

    with open(ground_truth_json_path) as gt_path:
        ground_truth_json = json.load(gt_path)

    with open(predictions_json_path) as predictions_path:
        predictions_json = json.load(predictions_path)

    if evaluation_parameters["is_heartcycle_averaging_needed"]==True:
        ground_truth_json = average_heart_cycles(ground_truth_json)
        predictions_json = average_heart_cycles(predictions_json)

    confusion_matrix = np.zeros((num_of_classes, num_of_classes))

    if save_raw_comparison:
        raw_comparison = "DicomID,GroundTruth,Predicted\n"

    if evaluation_parameters["evaluation_task_type"] == "regression":

        sum_diff = 0
        square_diff = 0

        dicom_gts = []
        preds_averages = []

        for patient_item_id in ground_truth_json:
            
            preds_averages.append(predictions_json[patient_item_id]['predicted_EF'])
            dicom_gts.append(ground_truth_json[patient_item_id]['gt_EF'])

            diff = abs(predictions_json[patient_item_id]['predicted_EF'] - ground_truth_json[patient_item_id]['gt_EF'])

            sum_diff += diff

            square_diff += diff*diff

            if save_raw_comparison:
                raw_comparison += "{},{},{}\n".format(patient_item_id,
                ground_truth_json[patient_item_id]['gt_EF'], predictions_json[patient_item_id]['predicted_EF'])

        average_error = sum_diff / len(ground_truth_json)
        square_error = square_diff / len(ground_truth_json)

        if save_raw_comparison:
            with open(os.path.join(evaluation_parameters["evaluation_folder"], "raw_comparison.csv"), "w") as f:
                f.write(raw_comparison)

        dicom_gts = np.asarray(dicom_gts)
        preds_averages = np.asarray(preds_averages)

        plt.plot(list(range(70)))
        plt.scatter(dicom_gts, preds_averages)
        plt.savefig(os.path.join(evaluation_parameters["evaluation_folder"], "scatter_plot.png"))
        plt.show()

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(dicom_gts, preds_averages)
        r2 = r2_score(dicom_gts, preds_averages)

        result_string = "MAE after averaging heart cycles: {}\nMSE after averaging heart cycles: {}\nRMSE after averaging heart cycles: {}\nR-scipy value: {}\nr2 score-sklearn: {}".format(
            average_error, square_error, math.sqrt(square_error), r_value, r2)

        print(result_string)

        with open(os.path.join(evaluation_parameters["evaluation_folder"], "results.txt"), "w") as f:
            f.write(result_string)

    else:

        for patient_item_id in ground_truth_json:

            if model_parameters["task_type"] == "regression":
                predicted_class = EF_to_class(evaluation_parameters,
                                              predictions_json[patient_item_id]['predicted_EF'])
                confusion_matrix[ground_truth_json[patient_item_id]['gt_class'], predicted_class] += 1
            else:
                confusion_matrix[
                    int(ground_truth_json[patient_item_id]['gt_class']), int(predictions_json[patient_item_id][
                        'predicted_class'])] += 1

            if save_raw_comparison:
                raw_comparison += "{},{},{}\n".format(patient_item_id,
                ground_truth_json[patient_item_id]['gt_class'], predictions_json[patient_item_id]['predicted_class'])

        if save_raw_comparison:
            with open(os.path.join(evaluation_parameters["evaluation_folder"], "raw_comparison.csv"), "w") as f:
                f.write(raw_comparison)

        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        # Overall accuracy for each class
        ACC = (TP + TN) / (TP + FP + FN + TN)

        CLASS_ACC = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)

        print('Confusion matrix: \n {}'.format(confusion_matrix))
        if num_of_classes == 2:
            print('Class Accuracy: {}'.format(CLASS_ACC))
            print('Accuracy: {}'.format(ACC[0]))
            print('Specificity: {}'.format(TNR[0]))
            print('Sensitivity: {}'.format(TPR[0]))
            print('F1-score: {}'.format((2 * PPV[0] * TPR[0]) / (PPV[0] + TPR[0])))
        else:
            print('Class Accuracy: {}'.format(CLASS_ACC))
            print('Accuracy: {}'.format(ACC))
            print('Specificity: {}'.format(TNR))
            print('Sensitivity: {}'.format(TPR))
            print('F1-score: {}'.format((2 * PPV * TPR) / (PPV + TPR)))


def main():

    evaluation_parameter_json_path = sys.argv[1]

    with open(evaluation_parameter_json_path) as f:
        evaluation_parameters = json.load(f)

    model_parameter_json_path = evaluation_parameters["model_parameter_json_path"]
    dataset_json_path = evaluation_parameters["dataset_json_path"]
    model_path = evaluation_parameters["model_path"]
    evaluation_folder = evaluation_parameters["evaluation_folder"]
    is_heartcycle_averaging_needed = evaluation_parameters["is_heartcycle_averaging_needed"]

    if not os.path.exists(evaluation_folder):
        os.makedirs(evaluation_folder)

    with open(model_parameter_json_path) as f:
        model_parameters = json.load(f)

    predictions_json_path = os.path.join(evaluation_folder, "predicted.json")
    run_model_on_dataset(model_parameters, evaluation_parameters, dataset_json_path, model_path, predictions_json_path, only_model_weights=True)
    ground_truth_json_path = os.path.join(evaluation_folder, "ground_truth.json")
    groundtruth_creator(evaluation_parameters, dataset_json_path, ground_truth_json_path)

    evaluation(model_parameters, evaluation_parameters, ground_truth_json_path, predictions_json_path)


if __name__ == '__main__':
    main()
