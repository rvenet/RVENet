import json
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

from RVEnet.utils.task_types import TaskTypes

def generate_histogram(json_data: dict, nbr_histogram_bins: int, EF_min: int, EF_max: int):

    histogram = []

    histogram_EFs = []

    for i in range(nbr_histogram_bins):
        histogram.append([])

    EF_range = EF_max - EF_min + 1
    EF_step = EF_range / nbr_histogram_bins

    for patient in list(json_data):
        if not json_data[patient]['EF']:
            del json_data[patient]
            continue

        EF_value = float(json_data[patient]['EF'])
        if EF_value > EF_max or EF_value < EF_min:
            del json_data[patient]
            continue

        histogram_bin = int((EF_value - EF_min) / EF_step)

        for dicom in json_data[patient]['dicoms']:
            dicom_id = "{}__{}".format(patient,dicom['dicom_id'])
            histogram[histogram_bin].append(dicom_id)
            histogram_EFs.append(EF_value)

    return histogram, histogram_EFs


def balance_jsondata(json_path: str, nbr_histogram_bins: int=10, EF_min: int=10, EF_max: int=80, max_bin_difference: int=5, show_histograms: bool=True):

    with open(json_path, "r") as data:
        json_data = json.load(data)

    # 1. GENERATE HISTOGRAM

    histogram, original_EFs = generate_histogram(json_data, nbr_histogram_bins, EF_min, EF_max)

    # 2. FILTER HISTOGRAM

    bin_counts = [len(hist_bin) for hist_bin in histogram]

    min_bin_count = max(min(bin_counts),1)
    max_bin_sample = min_bin_count*max_bin_difference

    for bin_idx in range(len(histogram)):
        random.shuffle(histogram[bin_idx])

    filtered_histogram = [hist_bin[:max_bin_sample] for hist_bin in histogram]

    filtered_list = [dicom_id for hist_bin in filtered_histogram for dicom_id in hist_bin]

    # 3. REMOVE/DUPLICATE SAMPLES

    balanced_json = copy.deepcopy(json_data)

    for bin_idx in range(len(histogram)):

        if len(histogram[bin_idx])>min_bin_count*max_bin_difference:
            # REMOVE SAMPLES
            
            bin_patients = []
            for bin_dicom_name in histogram[bin_idx]:
                patient_id, _ = bin_dicom_name.split("__")
                bin_patients.append(patient_id)

            bin_patients = list(set(bin_patients))

            only_one_dicom_left = False
            patient_idx = 0
            patient_dicom_counter = [len(balanced_json[patient_id]['dicoms']) for patient_id in bin_patients]

            while sum(patient_dicom_counter)>min_bin_count*max_bin_difference:
                
                if patient_idx >= len(bin_patients):
                    patient_idx=0

                patient_id = bin_patients[patient_idx]
                if len(balanced_json[patient_id]['dicoms'])==1:
                    if only_one_dicom_left:
                        del balanced_json[patient_id]
                        del bin_patients[patient_idx]
                    else:
                        patient_idx+=1
                        continue
                else:
                    del balanced_json[patient_id]['dicoms'][0]

                patient_idx+=1
                patient_dicom_counter = [len(balanced_json[patient_id]['dicoms']) for patient_id in bin_patients]
                if all(counter==1 for counter in patient_dicom_counter):
                    only_one_dicom_left = True

        else:
            # DUPLICATE SAMPLES
            dicom_idx = 0
            dicom_counter = len(histogram[bin_idx])
            while dicom_counter<min_bin_count*max_bin_difference:
                bin_dicom_name = histogram[bin_idx][dicom_idx]
                patient_id, dicom_id = bin_dicom_name.split("__")
                
                target_dicom = [d for d in json_data[patient_id]['dicoms'] if d['dicom_id']==dicom_id][0]
                balanced_json[patient_id]['dicoms'].append(target_dicom)
                dicom_counter+=1

                dicom_idx +=1
                if dicom_idx==len(histogram[bin_idx]):
                    dicom_idx=0

    # 4. PLOT HISTOGRAMS
    
    if show_histograms:
        _, remaining_EFs = generate_histogram(balanced_json, nbr_histogram_bins, EF_min, EF_max)

        fig, axs = plt.subplots(2)
        fig.suptitle('Original and filtered EF histogram')
        axs[0].hist(original_EFs,bins=nbr_histogram_bins)
        axs[1].hist(remaining_EFs,bins=nbr_histogram_bins)
        plt.show()

    # 5 SAVE FILTERED JSON

    output_json_path = json_path[:-5] + "_balanced.json"

    with open(output_json_path, 'w') as f:
        json.dump(balanced_json, f)


class DatasetModifier:
    def __init__(self, EF_min: int, EF_max: int, task: TaskTypes, json_data: dict, is_balancing_needed: bool):
        self.EF_min = EF_min
        self.EF_max = EF_max
        self.task = task
        if self.task.task_type == TaskTypes.CLASSIFICATION or self.task.task_type == TaskTypes.BINARY_CLASSIFICATION:
            self.classes = self.task.output_nbr

        self.json_data = json_data
        self.calculator = {}
        self.is_balancing_needed = is_balancing_needed
        self.ranges = None
        if self.task.classification_thresholds and str(self.task.output_nbr) in self.task.classification_thresholds:
            self.ranges = self.task.classification_thresholds[str(self.task.output_nbr)]

    def prepare_dataset_json(self):
        for patient in list(self.json_data):
            if not self.json_data[patient]['EF']:
                # print("EF is missing at patint: {}".format(patient))
                del self.json_data[patient]
                continue

            EF_value = float(self.json_data[patient]['EF'])
            if EF_value > self.EF_max or EF_value < self.EF_min:
                # print("ERROR: EF: {0} is not in the range[{1},{2}]".format(self.json_data[patient]['EF'], self.EF_min, self.EF_max))
                del self.json_data[patient]
                continue

            if self.task.task_type == TaskTypes.CLASSIFICATION or self.task.task_type == TaskTypes.BINARY_CLASSIFICATION:
                if self.ranges:
                    EF_class = [i for i in range(len(self.ranges) - 1) if self.ranges[i] < EF_value <= self.ranges[i + 1]][0]
                else:
                    EF_range = self.EF_max - self.EF_min + 1
                    EF_step = EF_range / self.classes
                    EF_class = int((EF_value - self.EF_min) / EF_step)

                self.json_data[patient]['EF'] = EF_class

                if EF_class in self.calculator:
                    self.calculator[EF_class] += len(self.json_data[patient]['dicoms'])
                else:
                    self.calculator[EF_class] = len(self.json_data[patient]['dicoms'])
            elif self.task.task_type == TaskTypes.REGRESSION:
                self.json_data[patient]['EF'] = float(self.json_data[patient]['EF'])
            else:
                raise ValueError('Task type is not recognized.')

        if self.is_balancing_needed and self.task.task_type == TaskTypes.CLASSIFICATION or self.task.task_type == TaskTypes.BINARY_CLASSIFICATION:
            self.balance_dataset()
        return self.json_data

    def balance_dataset(self):
        if not self.calculator:
            raise ValueError('Run the prepare_dataset_json function first!')

        max_value = max(self.calculator.values())
        for key in self.calculator:
            self.calculator[key] = int(max_value / self.calculator[key])

        for patient in self.json_data:
            EF_class = self.json_data[patient]['EF']
            if self.calculator[EF_class] == 1:
                continue

            self.json_data[patient]['dicoms'] = list(
                np.repeat(self.json_data[patient]['dicoms'], min(self.calculator[EF_class], 10)))
