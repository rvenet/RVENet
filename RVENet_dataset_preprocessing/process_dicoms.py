import argparse
import os
from os.path import exists
import cv2
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def convert_rgb2ybr(pixel_array: np.ndarray) -> np.ndarray:
    new_array = np.empty_like(pixel_array)
    for i in range(pixel_array.shape[0]):
        new_array[i] = cv2.cvtColor(pixel_array[i], cv2.COLOR_RGB2YCrCb)
    return new_array

def normalize(value: float, maximum: int, minimum: int) -> int:
    return int(255 * ((value - minimum) / (maximum - minimum)))

def bounding_box(points: list) -> dict:
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)

    return {'min_x':bot_left_x, 'min_y':bot_left_y, 'max_x': top_right_x, 'max_y':top_right_y}

def create_mask(gray_frames: np.ndarray) -> tuple(np.ndarray, np.ndarray, dict):
    shape_of_frames = gray_frames.shape
    changes = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    changes_frequency = np.zeros((shape_of_frames[1], shape_of_frames[2]))
    binary_mask = np.zeros((shape_of_frames[1], shape_of_frames[2]))

    for i in range(len(gray_frames) - 1):
        diff = abs(gray_frames[i] - gray_frames[i + 1])
        changes += diff
        nonzero = np.nonzero(diff)
        changes_frequency[nonzero[0], nonzero[1]] += 1

    max_of_changes = np.amax(changes)
    min_of_changes = np.min(changes)

    for r in range(len(changes)):
        for p in range(len(changes[r])):
            if int(changes_frequency[r][p]) < 10:
                changes[r][p] = 0
            else:
                changes[r][p] = normalize(changes[r][p], max_of_changes, min_of_changes)

    nonzero_values_for_binary_mask = np.nonzero(changes)

    binary_mask[nonzero_values_for_binary_mask[0], nonzero_values_for_binary_mask[1]] += 1
    kernel = np.ones((5, 5), np.int32)
    erosion_on_binary_msk = cv2.erode(binary_mask, kernel, iterations=1)
    binary_mask_after_erosion = np.where(erosion_on_binary_msk, binary_mask, 0)

    nonzero_values_after_erosion = np.nonzero(binary_mask_after_erosion)
    binary_mask_coordinates = np.array([nonzero_values_after_erosion[0], nonzero_values_after_erosion[1]]).T
    binary_mask_coordinates = list(map(tuple, binary_mask_coordinates))
    bbox = bounding_box(binary_mask_coordinates)
    cropped_mask = binary_mask_after_erosion[int(bbox['min_x']):int(bbox['max_x']),
                    int(bbox['min_y']):int(bbox['max_y'])]

    for row in cropped_mask:
        ids = [i for i, x in enumerate(row) if x == 1]
        if len(ids) < 2:
            continue
        row[ids[0]:ids[-1]] = 1

    return cropped_mask, erosion_on_binary_msk, bbox

def save_frames(dcm, path_to_save):
    pixel_array = dcm.pixel_array
    if dcm.PhotometricInterpretation == 'RGB':
        pixel_array = convert_rgb2ybr(pixel_array)

    if pixel_array.shape[0] < 10:
        raise ValueError('Video is too short!')

    gray_frames = np.zeros(pixel_array.shape)
    gray_frames = pixel_array[:, :, :, 0]

    cropped_mask, erosion_on_binary_mask, bbox = create_mask(gray_frames)

    os.makedirs(os.path.join(path_to_save, 'frames'), exist_ok=True)
    #np.savez(path_to_save / 'mask.npz', mask=cropped_mask)
    cv2.imwrite(str(path_to_save / 'mask.png'), cropped_mask*255)

    for i, gray_frame in enumerate(gray_frames):
        masked_image = np.where(erosion_on_binary_mask, gray_frame, 0)
        cropped_image = masked_image[int(bbox['min_x']):int(bbox['max_x']),
                    int(bbox['min_y']):int(bbox['max_y'])]
        im = Image.fromarray(cropped_image)
        im = im.convert('L')
        im.save(path_to_save / 'frames' / f'frame{str(i + 1)}.png')

def preprocess_dicom(dicom_data: pd.Series, dicom_path: Path, output_path: Path, skip_saving: bool = False):

    dcm = pydicom.dcmread(dicom_path)
    
    dicom_id = dicom_data['FileName']
    patient_id = '_'.join(dicom_id.split('_')[:2])
    dicom_idx = str(int(dicom_id.split('_')[2]))

    path_to_save = output_path / patient_id / dicom_idx

    if not skip_saving:
        save_frames(dcm, path_to_save)
                
    if not exists(path_to_save):
        raise ValueError(f'{path_to_save} does not exists.')
    
def preprocess_data(path_to_data: Path, path_to_csv: Path, output_folder: Path, output_csv_path: Path, skip_saving: bool):
    df = pd.read_csv(path_to_csv)
    os.makedirs(output_folder, exist_ok=True)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        dicom_path = path_to_data / f"{row['FileName']}.dcm"

        try:
            preprocess_dicom(row, dicom_path, output_folder, skip_saving)
        except ValueError as e:
            print(f"{row['FileName']} dropped because: {str(e)}")
            df.drop(i, inplace=True)
            continue
        except Exception as e:
            print(f"Can not preprocess file: {str(e)}, {row['FileName']} dropped")
            df.drop(i, inplace=True)
            continue
        #print(f"{row['FileName']} done!")
    df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', required=True, help='path to the folder containing the dicoms')
    parser.add_argument('--path_to_csv', required=True, help='path to the csv containing the path to the dicoms')
    parser.add_argument('--output_folder', required=True, help='path to the folder where the preprocessed dicoms will be saved')
    parser.add_argument('--out_csv', default='codebook.csv', help='name of the output csv')
    parser.add_argument('--skip_saving', action='store_true', help="with this flag the preprocessing and saving step will be skipped. \
                                                                    Only the already preprocessed and saved dicoms in the ouput_folder will be kept in the output csv")

    args = parser.parse_args()
    path_to_csv = Path(args.path_to_csv)
    path_to_data = Path(args.path_to_data)
    output_folder = Path(args.output_folder)
    out_csv = Path(args.out_csv)
    skip_saving = args.skip_saving
    preprocess_data(path_to_data=path_to_data,path_to_csv=path_to_csv, output_folder=output_folder, output_csv_path=out_csv, skip_saving=skip_saving)