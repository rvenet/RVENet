import argparse
import os
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tqdm import tqdm

def calculate_frame_index_arrays(num_of_frames: int, num_of_frames_to_sample: int, sec:int, fps:int) -> dict:
    dicoms = {}
    largest_index = 1
    counter = 1

    len_of_heart_cycle = sec * int(float(fps))
    while True:
        sampled_indexes = list(np.linspace(largest_index, largest_index + len_of_heart_cycle,  num_of_frames_to_sample))
        if int(sampled_indexes[-1]) <= num_of_frames:
            dicoms[counter - 1] = [int(x) for x in sampled_indexes]
            counter += 1
            largest_index = sampled_indexes[-1]
        else:
            break

    return dicoms


def create_data_dict(dicom_labels: pd.DataFrame, path_to_preprocessed_dicoms: Path, frames_to_sample=20, label='label') -> dict:
    min_hr = 30
    max_hr = 150

    my_json = dict()
    n_excluded_dicoms = 0
    n_valid_dicoms = 0
    n_valid_samples = 0
    excluded_dicoms = []
    for index, row in tqdm(dicom_labels.iterrows(), total=len(dicom_labels), mininterval=0.1):
        dicom_id = row['FileName']
        patient_id = '_'.join(dicom_id.split('_')[:2])
        dicom_idx = str(int(dicom_id.split('_')[2]))
        path_to_dicom = path_to_preprocessed_dicoms / patient_id / dicom_idx / 'frames'
        
        if not os.path.exists(path_to_dicom):
            n_excluded_dicoms += 1
            excluded_dicoms.append(path_to_dicom)
            print(f'{path_to_dicom} excluded because the video is not present in the preprocessed data!')
            continue

        frames = [f for f in os.listdir(path_to_dicom)
                         if f.endswith('.png') and os.path.isfile(path_to_dicom / f)]
        num_of_frames = len(frames)

        if num_of_frames < frames_to_sample:
            n_excluded_dicoms += 1
            excluded_dicoms.append(path_to_dicom)
            print(f'{path_to_dicom} excluded because there are not enough frames in video!')
            continue
        
        if pd.isna(row[label]):
            n_excluded_dicoms += 1
            excluded_dicoms.append(path_to_dicom)
            print(f'{path_to_dicom} excluded because it has no label!')
            continue

        if pd.isna(row['HR']) or row['HR'] <= min_hr or row['HR'] >= max_hr:
            n_excluded_dicoms += 1
            excluded_dicoms.append(path_to_dicom)
            print(f'{path_to_dicom} excluded because it has invalid heart rate')
            continue
        if num_of_frames < (60 / int(row['HR'])) * int(float(row['FPS'])):
            n_excluded_dicoms += 1
            excluded_dicoms.append(path_to_dicom)
            print(f'{path_to_dicom} excluded because the video is shorter than 1 cardiac cycle!')
            continue

        hr = row['HR']
        fps = row['FPS']
        frame_indexes = calculate_frame_index_arrays(num_of_frames, frames_to_sample, (60 / int(hr)), fps)


        patient_id = '_'.join(dicom_id.split('_')[:2])
        if patient_id not in my_json.keys():
            my_json[patient_id] = {
                'EF': row[label],
                'dicoms': []
            }
        
        my_json[patient_id]['dicoms'].append({
            'dicom_id': int(dicom_id.split('_')[2]),
            'frame_indexes': frame_indexes
        })
        
        n_valid_dicoms += 1
        n_valid_samples += len(frame_indexes)
    
    print(f'No. of valid DICOM files: {n_valid_dicoms}')
    print(f'No. of valid samples: {n_valid_samples}')

    return my_json

def create_jsons(path_to_csv: Path,
                path_to_preprocessed_dicoms: Path,
                frames_to_sample: int=None,
                label='RVEF'):
    dicom_labels = pd.read_csv(path_to_csv)
    train_labels = dicom_labels[dicom_labels['Split'] == 'train']
    validation_labels = dicom_labels[dicom_labels['Split'] == 'validation']
    train_dict = create_data_dict(dicom_labels=train_labels,
                                path_to_preprocessed_dicoms=path_to_preprocessed_dicoms,
                                frames_to_sample=frames_to_sample,
                                label=label)
    test_dict = create_data_dict(dicom_labels=validation_labels,
                                path_to_preprocessed_dicoms=path_to_preprocessed_dicoms,
                                frames_to_sample=frames_to_sample,
                                label=label)


    with open(path_to_preprocessed_dicoms / 'train_set.json', 'w') as outfile:
        json.dump(train_dict, outfile)
    with open(path_to_preprocessed_dicoms / 'test_set.json', 'w') as outfile:
        json.dump(test_dict, outfile)

    return train_dict, test_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', required=True, help='path to the csv containing the label and uid of the data')
    parser.add_argument('--path_to_preprocessed_data', required=True, help='path to the folder containing the preprocessed videos')
    parser.add_argument('--label', default='label', help="the name of the label column in the csv, by default 'RVEF' is used")
    parser.add_argument('--frames_to_sample', type=int, default=None, help="number of frames sampled from each heart cycle")
    
    args = parser.parse_args()
    path_to_csv = Path(args.path_to_csv)
    path_to_preprocessed_dicoms = Path(args.path_to_preprocessed_data)
    label = args.label
    frames_to_sample = args.frames_to_sample
    
    create_jsons(path_to_csv=path_to_csv,
                 path_to_preprocessed_dicoms=path_to_preprocessed_dicoms,
                 frames_to_sample=frames_to_sample,
                 label=label)