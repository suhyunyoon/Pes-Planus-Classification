import os
import glob
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm

import torch

from torch.utils.data import Dataset

from torchvision import transforms as tf 

# transformation
mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
def get_transform(split, hw=256, crop_size=224, is_tensor=False):
    transform = None
    if split == 'train':
        transform = [tf.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
                    tf.RandomHorizontalFlip(p=0.5),
                    #tf.ColorJitter(hue=.05, saturation=.05),
                    tf.Normalize(mean, std)]
        resize = tf.Resize((hw,hw))
    else:
        transform = [tf.Normalize(mean, std)]
        resize = tf.Resize((crop_size, crop_size))
    
    if not is_tensor:
        transform = [resize, tf.ToTensor()] + transform

    transform = tf.Compose(transform)
    return transform

def get_totensor(hw=256):
    return tf.Compose([tf.Resize((hw,hw)),
                    tf.ToTensor()])

def get_pressure_transform(split):
    transform = None
    if split == 'train':
        transform = tf.Compose([#tf.ToTensor(),
                            tf.CenterCrop((64,128)),
                            #tf.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
                            tf.RandomVerticalFlip(p=0.5),
                            #tf.ColorJitter(hue=.05, saturation=.05),
                            tf.Normalize(mean[0], std[0])])
    else:
        transform = tf.Compose([#tf.ToTensor(),
                            tf.CenterCrop((64, 128)),
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

class FootDatasetOnMem(Dataset):
    def __init__(self, data_root='./', data_split='train', transform=None, val_ratio=0.15):
        # Init
        super(FootDatasetOnMem, self).__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.transform = transform
        self.val_ratio = val_ratio

        temp_split = 'train' if data_split == 'val' else self.data_split

        # read csv (img path)
        self.ids, self.targets = read_annotations(self.data_root, self.data_split, self.val_ratio)
        
        # get specific image path (get 4 data in a id)
        self.image_path, self.images, self.labels = [], [], []
        transform = get_totensor(256)
        print("Read Datasets...")
        for key, target in tqdm(zip(self.ids, self.targets)):
            # get images, txt file
            xray_path = os.path.join(self.data_root, temp_split, key, 'xray', '*')
            xray_images = glob.glob(xray_path)
            for img in xray_images:
                self.image_path.append(img)
                self.labels.append(target)
                # read .jpg
                img_raw = Image.open(img)
                self.images.append(transform(img_raw))
        # To tensor
        self.images = torch.stack(self.images, dim=0)
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):    
        if self.transform is not None:
            img = self.transform(self.images[idx])
        else:
            img = self.images[idx]

        return img, self.labels[idx]

class PressureDataset(Dataset):
    def __init__(self, data_root='./', data_split='train', transform=None, val_ratio=0.0):
        # Init
        super(PressureDataset, self).__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.transform = transform
        self.val_ratio = val_ratio

        temp_split = 'train' if data_split == 'val' else self.data_split

        # read csv (img path)
        self.ids, self.labels = read_annotations(self.data_root, self.data_split, self.val_ratio)
        
        # get specific image path (get 4 data in a id)
        self.images  = []
        for key in self.ids:
            # get txt file
            pressure_path = os.path.join(self.data_root, temp_split, key, 'rsdb', '1_Static_Image.txt')
            
            with open(pressure_path, 'r') as f:
                img = f.read()
            img = [[list(map(float, x.split())) for x in img.split('\n')[:-3]]] * 3
            self.images.append(img)

        # To tensor
        self.images = torch.FloatTensor(self.images)
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.transform(self.images[idx])
        else:
            img = self.images[idx]

        return img, self.labels[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./", type=str, help="Must contains train_annotations.csv")
    args = parser.parse_args()
    
    dataset_val = FootDataset(data_root=args.data_root, data_split='test', transform=get_transform('val'), val_ratio=0.)
    print(len(dataset_val))
    '''
    dataset_train = FootDataset(data_root=args.data_root, data_split='train', transform=get_transform('train'), val_ratio=0.2)
    dataset_val = FootDataset(data_root=args.data_root, data_split='val', transform=get_transform('val'), val_ratio=0.2)

    print(len(dataset_train), len(dataset_val))
    print(dataset_train[0][0].shape, dataset_train[0][1])
    print(dataset_val[0][0].shape, dataset_val[0][1])

    dataset_train = PressureDataset(data_root=args.data_root, data_split='train', val_ratio=0.2, transform=get_pressure_transform('train'))
    dataset_val = PressureDataset(data_root=args.data_root, data_split='val', val_ratio=0.2, transform=get_pressure_transform('val'))

    print(len(dataset_train), len(dataset_val))
    print(dataset_train[0][0].shape, dataset_train[0][1])
    print(dataset_val[0][0].shape, dataset_val[0][1])
    
    # Optional with sufficient RAM
    dataset_train = FootDatasetOnMem(data_root=args.data_root, data_split='train', transform=get_transform('train', is_tensor=True), val_ratio=0.2)
    dataset_val = FootDatasetOnMem(data_root=args.data_root, data_split='val', transform=get_transform('val', is_tensor=True), val_ratio=0.2)

    print(len(dataset_train), len(dataset_val))
    print(dataset_train[0][0].shape, dataset_train[0][1])
    print(dataset_val[0][0].shape, dataset_val[0][1])
    '''
