import os
import glob
import pandas as pd
from PIL import Image

import torch

from torch.utils.data import Dataset

from torchvision import transforms as tf 

# transformation
mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
def get_transform(split, hw, crop_size):
    transform = None
    if split == 'train':
        transform = tf.Compose([tf.Resize((hw,hw)),
                            tf.ToTensor(),
                            tf.RandomCrop(crop_size, padding=4, padding_mode='reflect')
                            tf.RandomHorizontalFlip(p=0.5),
                            tf.ColorJitter(hue=.05, saturation=.05)
                            tf.Normalize(mean, std)])
    else:
        transform = tf.Compose([tf.Resize((crop_size, crop_size)),
                            tf.ToTensor(),
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
        target = [0 for i in len(ids)]

    return ids, target

class FootDataset(Dataset):
    def __init__(self, data_root='./', data_split='train', transform=None, val_ratio=0.15):
        # Init
        super(FootDataset, self).__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.transform = transform
        self.val_ratio = val_ratio

        temp_split = 'train' if data_split == 'val' else self.data_split

        # read csv (img path)
        self.ids, self.targets = read_annotations(self.data_root, self.data_split, self.val_ratio)
        
        # get specific image path (get 4 data in a id)
        self.images, self.labels = [], []
        for key, target in zip(self.ids, self.targets):
            # get images, txt file
            xray_path = os.path.join(self.data_root, temp_split, key, 'xray', '*')
            xray_images = glob.glob(xray_path)
            for img in xray_images:
                self.images.append(img)
                self.labels.append(target)
        # To tensor
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_raw = Image.open(self.images[idx])

        if self.transform is not None:
            img = self.transform(img_raw)
        else:
            img = img_raw

        return img, self.labels[idx]


if __name__ == "__main__":
    dataset_train = FootDataset(data_root='/home/suhyun/dataset/계명대 동산병원_데이터', data_split='train', val_ratio=0.2)
    dataset_val = FootDataset(data_root='/home/suhyun/dataset/계명대 동산병원_데이터', data_split='val', val_ratio=0.2)

    print(len(dataset_train), len(dataset_val))
    print(dataset_train[0])
    print(dataset_val[0])
