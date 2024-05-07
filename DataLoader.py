
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import json
import random


class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num

        json_file = open(os.path.join(data_path, f'label2image_{mode}.json'), "r")
        dataset = json.load(json_file)

        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
      
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)
        
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)

        if self.prompt_path is None:
            boxes = get_boxes_from_mask(mask)
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)

class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5, subset_fraction=0.01):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.data_dir = data_dir

        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        # all_image_paths = list(dataset.keys())
        # all_label_paths = list(dataset.values())
        all_image_paths = list(dataset.keys())[30000:]  # Skip the first 30000 entries
        all_label_paths = list(dataset.values())[30000:]
        
        # Sampling a subset of the dataset
        if subset_fraction < 1.0:
            total_samples = int(len(all_image_paths) * subset_fraction)
            sampled_indices = np.random.choice(len(all_image_paths), total_samples, replace=False)
            self.image_paths = [all_image_paths[i] for i in sampled_indices]
            self.label_paths = [all_label_paths[i] for i in sampled_indices]
        else:
            self.image_paths = all_image_paths
            self.label_paths = all_label_paths

    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_file_path = os.path.join('/mnt/dataset2', self.image_paths[index])

        image_input = {}
        try:
            image = cv2.imread(image_file_path)
            image = (image - self.pixel_mean) / self.pixel_std
        

            h, w, _ = image.shape
            transforms = train_transforms(self.image_size, h, w)
        
            masks_list = []
            boxes_list = []
            point_coords_list, point_labels_list = [], []
            mask_file_paths = [os.path.join(self.data_dir, m_path) for m_path in random.choices(self.label_paths[index], k=self.mask_num)]

            for m in mask_file_paths:
                pre_mask = cv2.imread(m, 0)
                if pre_mask.max() == 255:
                    pre_mask = pre_mask / 255

                augments = transforms(image=image, mask=pre_mask)
                image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

                boxes = get_boxes_from_mask(mask_tensor)
                point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

                masks_list.append(mask_tensor)
                boxes_list.append(boxes)
                point_coords_list.append(point_coords)
                point_labels_list.append(point_label)

            mask = torch.stack(masks_list, dim=0)
            boxes = torch.stack(boxes_list, dim=0)
            point_coords = torch.stack(point_coords_list, dim=0)
            point_labels = torch.stack(point_labels_list, dim=0)

            image_input["image"] = image_tensor.unsqueeze(0)
            image_input["label"] = mask.unsqueeze(1)
            image_input["boxes"] = boxes
            image_input["point_coords"] = point_coords
            image_input["point_labels"] = point_labels

            image_name = self.image_paths[index].split('/')[-1]
            if self.requires_name:
                image_input["name"] = image_name
                return image_input
            else:
                return image_input
        except Exception as e:
            print(f"Error processing data at index {index}: {e}")
            return {"image": None, "label": None, "boxes": None, "point_coords": None, "point_labels": None, "name": None}
            
    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TrainingDataset("/mnt/dataset/SAMeD", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=60, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)