import os
import cv2
import numpy as np


def calculate_avarage_and_std(dataset_path: str, indicators_path: str):
    path_of_dicom_set = dataset_path
    dicom_folders = [f.path for f in os.scandir(path_of_dicom_set) if f.is_dir()]

    average_of_images = []
    variance_of_images = []
    for folder in dicom_folders:
        current_folder = os.path.join(path_of_dicom_set, folder)
        dicom_subfolders = [f.path for f in os.scandir(current_folder) if f.is_dir()]
        for subfolder in dicom_subfolders:
            image_paths = [f.path for f in os.scandir(os.path.join(current_folder, subfolder, 'frames'))]
            for image_path in image_paths:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                average_of_images.append(np.mean(img))
                variance_of_images.append(np.var(img))

    with open(os.path.join(path_of_dicom_set, indicators_path), 'w') as f:
        f.write('{:.2f} \n'.format(round(np.average(average_of_images), 2)))
        f.write('{:.2f} \n'.format(round(np.sqrt(np.average(variance_of_images)), 2)))


def get_indicators(dataset_path: str, indicators_path: str):
    if not os.path.exists(os.path.join(dataset_path, indicators_path)):
        calculate_avarage_and_std(dataset_path, indicators_path)

    f = open(os.path.join(dataset_path, indicators_path), "r")
    data = f.read().split('\n')
    average = data[0]
    std = data[1]
    f.close()

    return average, std
