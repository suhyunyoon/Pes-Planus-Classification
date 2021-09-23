import os
import glob
import pandas as pd
from PIL import Image

import torch


from torch.utils.data import Dataset

def read_annotations(data_root, data_split):
    file_name = data_split + '_annotation.csv'
    csv_path = os.path.join(data_root, file_name)

    # read csv
    ann = pd.read_csv(csv_path, index_col=0)
    data = ann.iloc[:,-1]
    
    # get data id, target
    ids = list(data.keys())
    if data_split == 'train':
        #target = torch.LongTensor(data.values)
        target = list(data.values)
    else:
        #target = torch.zeros(len(ids), dtype=torch.uint8)
        target = [0 for i in len(ids)]

    return ids, target

class FootDataset(Dataset):
    def __init__(self, data_root='./', data_split='train', transform=None, val_ratio=0.0):
        # Init
        super(FootDataset, self).__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.transform = transform

        # read csv (img path)
        self.ids, self.targets = read_annotations(self.data_root, self.data_split)
        
        # get specific image path (get 4 data in a id)
        self.images, self.labels = [], []
        for key, target in zip(self.ids, self.targets):
            # get images, txt file
            #########################
            for i in range(3):
                self.images.append(os.path.join(self.data_root, self.data_split, key, 'xray', 'xray_%d.jpg'%i))
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

        return img, self.targets[idx]


if __name__ == "__main__":
    dataset = FootDataset(data_root='/home/suhyun/dataset/계명대 동산병원_데이터', data_split='train')

    print(len(dataset))
    print(dataset[0])
