import os
import glob
import json
import pandas as pd
from PIL import Image

import torch

from torch.utils.data import Dataset

from torchvision import transforms as tf 

from dataset import read_annotations

# transformation
mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]

def read_aug_annotations(data_root):
    data_dir = os.path.join(data_root, 'xray_data')
    ann_path = os.path.join(data_dir, 'annotations.json')

    with open(ann_path, "r", encoding='us-ascii') as f:
        anns = json.load(f)
    
    keys = list(anns.keys())[:-1]

    # image path (L / R)
    images = [os.path.join(data_dir, i + '.jpg') for i in keys]

    # target
    is_flat = [anns[i]['is_flat'] for i in keys]

    # points
    points = [anns[i]['points'] for i in keys]

    # test data keys
    data_keys = keys[::2]
    ids = [anns[i]['patient_uuid'] for i in data_keys]

    data = {id_:[(is_flat[i*2], points[i*2]), 
                (is_flat[i*2] or is_flat[i*2+1], [[0,0]]*4), 
                (is_flat[i*2+1], points[i*2+1])] for i, id_ in enumerate(ids)}

    # image path, target, points, 
    return data

class FootDatasetAug(Dataset):
    def __init__(self, data_root='./', data_split='train', transform=None, val_ratio=0.):
        # Init
        super(FootDatasetAug, self).__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.transform = transform
        self.val_ratio = val_ratio

        temp_split = 'test'

        # read csv (img path)
        self.ids, targets = read_annotations(self.data_root, temp_split, 0)
        
        # read aug csv
        self.anns = read_aug_annotations(self.data_root)

        # split
        if self.val_ratio > 0.:
            pivot = int(len(self.ids) * (1. - val_ratio))
            if data_split == 'train':
                self.ids = self.ids[:pivot]
            elif data_split == 'val':
                self.ids = self.ids[pivot:]

        # get specific image path (get 4 data in a id)
        self.images, self.labels, self.points, self.types = [], [], [], []
        # type: L: 0, M: 1, R: 2
        type_dict = {'L': 0, 'M': 1, 'R': 2}
        for key in self.ids:
            # get images, txt file
            xray_path = os.path.join(self.data_root, temp_split+'_LMR', key, 'xray', '*.jpg')
            xray_images = glob.glob(xray_path)
            for img in xray_images:
                idx = type_dict[os.path.basename(img)[0]]
                self.images.append(img)
                self.labels.append(self.anns[key][idx][0])
                self.points.append(self.anns[key][idx][1])
                self.types.append(idx)
        # To tensor
        self.labels = torch.LongTensor(self.labels)
        self.points = torch.FloatTensor(self.points) 
        self.types = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_raw = Image.open(self.images[idx])

        if self.transform is not None:
            img = self.transform(img_raw)
        else:
            img = img_raw

        return img, self.labels[idx]#, self.points[idx], self.types[idx]


if __name__ == "__main__":
    dataset_train = FootDatasetAug(data_root='/home/suhyun/dataset/계명대 동산병원_데이터', data_split='train', val_ratio=0.2)
    dataset_val = FootDatasetAug(data_root='/home/suhyun/dataset/계명대 동산병원_데이터', data_split='val', val_ratio=0.2)

    print(len(dataset_train), len(dataset_val))
    print(dataset_train[0])
    print(dataset_val[0])
