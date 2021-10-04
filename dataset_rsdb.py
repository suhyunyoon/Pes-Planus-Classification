import os
import glob
import pandas as pd
from PIL import Image

import torch

from torch.utils.data import Dataset

from torchvision import transforms as tf 
import torch.nn.functional as F
import torch.nn as nn

import rsdb_reader

from os.path import join

from tqdm import tqdm

import argparse

# Mixed Dynamic Maximum Image


# transformation
mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
def get_rsdb_transform(split):
    transform = None
    if split == 'train':
        transform = tf.Compose([#tf.ToTensor(),
                            #tf.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
                            tf.RandomHorizontalFlip(p=0.5),
                            tf.RandomVerticalFlip(p=0.5),
                            #tf.RandomRotation()
                            #tf.ColorJitter(hue=.05, saturation=.05),
                            tf.Normalize(mean, std)])
    else:
        transform = tf.Compose([#tf.ToTensor(),
                            tf.Normalize(mean, std)])
    return transform


def read_annotations(data_root, data_split, val_ratio):
    temp_split = 'train' if data_split == 'val' else data_split
    file_name = temp_split + '_annotation.csv'
    csv_path = os.path.join(data_root, file_name)

    # read csv
    ann = pd.read_csv(csv_path, index_col=0)
    data = ann.iloc[:,-1]
    # split
    pivot = int(len(data) * (1. - val_ratio))
    if data_split == 'train':
        data = data[:pivot]
    elif data_split == 'val':
        data = data[pivot:]
    
    # get data id, target
    ids = list(data.keys())
    if data_split in ['train', 'val']:
        #target = torch.LongTensor(data.values)
        target = list(data.values)
    else:
        #target = torch.zeros(len(ids), dtype=torch.uint8)
        target = [0 for i in range(len(ids))]

    return ids, target


class CombinationDataset(Dataset):
    def __init__(self, data_root='./', data_split='train', transform=None, val_ratio=0.15):
        super(CombinationDataset, self).__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.transform = transform
        self.val_ratio = val_ratio

        temp_split = 'train' if data_split == 'val' else self.data_split

        dataset_dir = join(data_root, data_split)



        self.x_combinations = []
        self.x_img_list = []
        self.y_data = []

        self.foot_list_map = {}

        self.ids, self.targets = read_annotations(self.data_root, self.data_split, self.val_ratio)


        for uuid, target in tqdm(zip(self.ids, self.targets), total=len(self.ids)):
            rsdb_dir = join(dataset_dir, uuid, "rsdb")
            # print('rsdb_dir = {}'.format(rsdb_dir))
            dmi_list = rsdb_reader.convert_Dynamic_Maximum_Image(rsdb_dir)

            resized_dmi_list = {
                'left': [],
                'right': []
            }
            
            for side in ['left', 'right']:
                for foot in dmi_list[side]:
                    f = torch.Tensor(foot)
                    f.unsqueeze_(0)
                    f = f.repeat(3, 1, 1)
                    f.unsqueeze_(0)
                    # print("foot.shape = {}".format(f.shape))
                    
                    f = F.interpolate(f, size=(64,32), mode='bilinear')
                    f.squeeze_(0)
                    resized_dmi_list[side].append(f)


            # for side in ['left', 'right']:
            #     resized_dmi_list[side] = cv2.resize()

            self.foot_list_map[uuid] = resized_dmi_list

            for li in range(len(dmi_list['left'])):
                for ri in range(len(dmi_list['right'])):
                    combination = uuid, li, ri
                    img = torch.cat((resized_dmi_list['left'][li], resized_dmi_list['right'][ri]), 2)
                    
                    self.x_img_list.append(img)
                    self.x_combinations.append(combination)
                    self.y_data.append(target)


    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        
        # uuid, li, ri = self.x_combinations[idx]

        # l_img = self.foot_list_map[uuid]['left'][li]
        # r_img = self.foot_list_map[uuid]['right'][ri]

        img = self.x_img_list[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, self.y_data[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str, help="Must contains train_annotations.csv")
    args = parser.parse_args()
    
    dataset_train = CombinationDataset(data_root=args.data_root, data_split='train', transform=get_rsdb_transform('train'), val_ratio=0.)
    dataset_val = CombinationDataset(data_root=args.data_root, data_split='test', transform=get_rsdb_transform('test'), val_ratio=0.)

    print(len(dataset_train), len(dataset_val))
    print(dataset_train[0][0].shape, dataset_train[0][1])
    print(dataset_val[0][0].shape, dataset_val[0][1])

    print(dataset_train.ids[:5])
    print(dataset_val.ids[:5])

    # dataset_train = PressureDataset(data_root=args.data_roopt, data_split='train', val_ratio=0.2, transform=get_pressure_transform('train'))
    # dataset_val = PressureDataset(data_root=args.data_root, data_split='val', val_ratio=0.2, transform=get_pressure_transform('val'))

    # print(len(dataset_train), len(dataset_val))
    # print(dataset_train[0][0].shape, dataset_train[0][1])
    # print(dataset_val[0][0].shape, dataset_val[0][1])
