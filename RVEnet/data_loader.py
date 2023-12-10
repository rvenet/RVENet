import os
import random

import skimage
from skimage import transform
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import json

from RVEnet.utils.dataset_json_modifier import DatasetModifier
from RVEnet.utils.task_types import TaskTypes

FRAME_FOLDER_NAME = "frames"
MASK_FILE_NAME = "mask.png"

binary_classification_threshold = 45


################## Set parameters ends ##################


class Task():
    def __init__(self, task_type, output_nbr=None, classification_thresholds=None):
        self.task_type = task_type
        self.output_nbr = output_nbr
        self.classification_thresholds = classification_thresholds


class EchoDataset(Dataset):
    """Echocariography dataset."""

    def __init__(self, json_file: str, root_dir: str, DICOM_frame_nbr: int, task: TaskTypes, is_balancing_needed: bool, 
                 transform: torchvision.transforms=None, EF_min: int=10, EF_max: int=80, return_heart_cycle_ids: bool=False):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the image sequences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(json_file, "r") as data:
            self.pure_data = json.load(data)

        self.dataset_modifier = DatasetModifier(EF_min, EF_max, task, self.pure_data, is_balancing_needed)
        self.echo_data = self.dataset_modifier.prepare_dataset_json()

        self.video_key_list = self.create_key_indexing(self.echo_data)
        self.root_dir = root_dir
        self.DICOM_frame_nbr = DICOM_frame_nbr
        self.transform = transform

        self.EF_min = EF_min
        self.EF_max = EF_max

        self.return_heart_cycle_ids = return_heart_cycle_ids

        if task.task_type not in [t for t in TaskTypes]:
            raise ValueError('Unknown task was given: ' + str(task))

        self.task = task

    def create_key_indexing(self, dictionary: dict):
        key_list = []

        for patient_id in dictionary:
            counter = 0
            for dicom_dict in dictionary[patient_id]['dicoms']:
                for heartcycle_key in dicom_dict['frame_indexes']:
                    key_with_dicomid = '{}__{}__{}__{}'.format(patient_id, dicom_dict['dicom_id'],heartcycle_key, counter)
                    key_list.append(key_with_dicomid)
                    counter += 1

        return key_list

    def __len__(self):
        return len(self.video_key_list)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_key = self.video_key_list[idx]
        video_idx = video_key.split('__')[0]
        dicom_idx = video_key.split('__')[1]
        heart_cycle_idx = video_key.split('__')[2]
        echo_video_data = self.echo_data[video_idx]

        video_path = os.path.join(self.root_dir,
                                  video_idx, dicom_idx)

        # dicom_idx can not be simply used to index the dicom, because in certain patients, the index of the dicom is not continous (e.g. 0,1,3,5)
        frame_indexes = next(
            (dicom["frame_indexes"] for dicom in echo_video_data["dicoms"] if str(dicom["dicom_id"]) == dicom_idx), None)
        
        heart_cycle_frame_indexes = frame_indexes[heart_cycle_idx]

        frames = self.read_frames(video_path, heart_cycle_frame_indexes)

        frames = [np.transpose(frame, (1, 2, 0)) for frame in frames]

        binary_mask = (cv2.imread(str(os.path.join(video_path, MASK_FILE_NAME)))/255.0)[:,:,0]

        if len(frames) != self.DICOM_frame_nbr:
            print("Number of frames in DICOM: {0} is {1}, and the expected frame nbr for DICOMS is {2}.".format(
                str(video_key), str(len(frames)), str(self.DICOM_frame_nbr)))
            raise ValueError

        if len(frames) != self.DICOM_frame_nbr:
            print("Number of frames in DICOM: {0} is {1}, and the expected frame nbr for DICOMS is {2}.".format(
                str(video_key), str(len(frames)), str(self.DICOM_frame_nbr)))
            raise ValueError

        sample = {'frames': frames, 'binary_mask': binary_mask, 'EF': echo_video_data["EF"]}

        if self.transform:
            sample = self.transform(sample)

        if self.return_heart_cycle_ids == True:
            dicom_id = "{}__{}__{}".format(video_idx, dicom_idx, heart_cycle_idx)
            return sample, dicom_id

        return sample

    def read_frames(self, video_path: str, frame_indexes: list):
        frame_list = os.listdir(os.path.join(video_path, 'frames'))
        sampled_frames = []

        for i in frame_indexes:

            try:
                frame = [cv2.imread(os.path.join(video_path, FRAME_FOLDER_NAME, 'frame{}.png'.format(str(i))))[:, :, 0]]
            except:
                x = 1
                # print(os.path.join(video_path, FRAME_FOLDER_NAME, 'frame{}.png'.format(str(i))))

            sampled_frames.append(frame)

        return np.asarray(sampled_frames)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size: int):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample: dict):
        frames, binary_mask, EF = sample['frames'], sample['binary_mask'], sample['EF']
        resized_frames = []

        for frame in frames:
            # frame = np.transpose(frame, (1, 2, 0))

            resized_frame = transform.resize(frame, (self.output_size, self.output_size))

            resized_frames.append(resized_frame)

        resized_frames = np.asarray(resized_frames)
        resized_binary_mask = transform.resize(binary_mask, (self.output_size, self.output_size))

        return {'frames': resized_frames, 'binary_mask': resized_binary_mask, 'EF': EF}


class Normalize(object):
    def __init__(self, average: float, std: float):
        self.average = average
        self.std = std

    def __call__(self, sample: dict):
        frames, binary_mask, EF = sample['frames'], sample['binary_mask'], sample['EF']

        normalized_frames = []
        for frame in frames:
            normalized_frames.append((np.array(frame) - float(self.average)) / float(self.std))

        return {'frames': np.array(normalized_frames), 'binary_mask': np.array(binary_mask), 'EF': EF}

    def normalize_image_data(self, tensor_frames: torch.Tensor):

        normalized_frames = []

        for frame in tensor_frames:
            frame = frame.cpu().numpy()
            binnary_mask = frame[0]
            frame_copy_one =  (frame[1] - float(self.average)) / float(self.std)
            frame_copy_two =  (frame[2] - float(self.average)) / float(self.std)
            merged_frame_data = [binnary_mask, frame_copy_one,frame_copy_two]
            normalized_frames.append(merged_frame_data)

        return torch.tensor(np.array(normalized_frames))



class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size: int):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample: dict):
        frames, binary_mask, EF = sample['frames'], sample['binary_mask'], sample['EF']

        frame_nbr, h, w = frames.shape[:3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_frames = frames[:, top: top + new_h,
                         left: left + new_w]

        cropped_binary_mask = binary_mask[top: top + new_h,
                              left: left + new_w]

        return {'frames': cropped_frames, 'binary_mask': cropped_binary_mask, 'EF': EF}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: dict):
        frames, binary_mask, EF = sample['frames'], sample['binary_mask'], sample['EF']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        # print(frames.shape)
        frames = np.array(frames)
        frames = frames.transpose((0, 3, 1, 2))

        binary_mask = np.array(binary_mask)
        EF = np.array(EF, dtype=np.float64)

        # print('Shape: {}'.format(frames[0].shape))
        return {'frames': torch.from_numpy(frames),
                'binary_mask': torch.from_numpy(binary_mask),
                'EF': torch.from_numpy(EF)}


class ExpandFrameTensor(object):
    def __init__(self, is_binarymask_included: bool=False):
        self.is_binarymask_included = is_binarymask_included

    def __call__(self, sample: dict):
        frames, binary_mask, EF = sample['frames'], sample['binary_mask'], sample['EF']

        f, c, h, w = frames.size()
        new_shape = (f, 3, h, w)

        expanded_frames = frames.expand(new_shape)
        expanded_frames_clone = expanded_frames.clone()

        if self.is_binarymask_included:
            expanded_frames_clone[:, 0, :, :] = binary_mask

        return {'input_tensor': expanded_frames_clone,
                'EF': EF}

    """Read a frame tensor of [frame_nbr,1,H,W]
    and return a [frame_number,3,H,W] dimensional tensor, where
    the three channels are the duplications of the grayscale image is_binarymask_included=False. 
    If the parameter is true, the binary mask is included on one of the channels. """


class RandomRotation(object):

    def __init__(self, angle_from: int, angle_to: int):
        self.angle_from = angle_from
        self.angle_to = angle_to

    def __call__(self, sample: dict):
        frames, binary_mask, EF = sample['frames'], sample['binary_mask'], sample['EF']
        rotation = np.random.randint(self.angle_from, self.angle_to)
        rotated_frames = []
        for frame in frames:
            rotated_frame = skimage.transform.rotate(frame, rotation, resize=True)
            rotated_frames.append(rotated_frame)

        rotated_mask = skimage.transform.rotate(binary_mask, rotation, resize=True)
        return {'frames': rotated_frames,
                'binary_mask': rotated_mask,
                'EF': np.array(EF, dtype=np.float64)}


class HorizontalFlip(object):

    def __init__(self, p: float):
        self.p = p

    def __call__(self, sample: dict):
        frames, binary_mask, EF = sample['frames'], sample['binary_mask'], sample['EF']

        if random.random() < self.p:
            flipped_frames = []
            for frame in frames:
                flipped_frame = frame[:, ::-1]
                flipped_frames.append(flipped_frame)
            flipped_binary_mask = binary_mask[:, ::-1]
            return {'frames': np.array(flipped_frames),
                    'binary_mask': np.array(flipped_binary_mask),
                    'EF': np.array(EF, dtype=np.float64)}

        return sample


class ConcatBinaryMask(object):
    """read a frame tensor of [frame_nbr,3,h,w] and a binary mask of [1,h,w]
      and return a [frame_number,3,h,w] dimensional tensor, where
      the first to channel is the duplicated gayscale image and
      the third is the binary mask."""

    def __call__(self, sample: dict):
        frames, binary_mask, EF = sample['frames'], sample['binary_mask'], sample['ef']

        # todo: merge tensors!!
        input_tensor = frames
        input_tensor[:, 0, :, :] = binary_mask

        return {'frames': input_tensor,
                'binary_mask': binary_mask,
                'EF': EF}


def visualize_data_samples(dataset: list):
    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample['frames'].shape, sample['EF'])

        fig = plt.figure()

        for frame_idx in range(len(sample['frames'])):
            ax = plt.subplot(1, len(sample['frames']), frame_idx + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(frame_idx))
            ax.axis('off')
            plt.imshow(sample['frames'][frame_idx])


def preprocess_EFNet(input_tensor: torch.Tensor):
    s = input_tensor.shape
    new_s = list(s[2:])
    new_s.insert(0, -1)
    reshaped_input_tensor = input_tensor.contiguous().view(new_s)

    return reshaped_input_tensor

def preprocess_r2plus1d(input_tensor: torch.Tensor):
    input_tensor = torch.swapaxes(input_tensor, 1, 2) # channel, fram nbr swap
    return input_tensor
