import argparse
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd

import torch
from torch import nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from torch.utils.data import DataLoader

from dataset import FootDataset, PressureDataset, get_transform, get_pressure_transform
from dataset_aug import FootDatasetAug
from dataset_rsdb import CombinationDataset, get_rsdb_transform

NUM_CLASSES = 2

# ensemble metric
def predict(logit, t, mode='logit'):
    idx = torch.argsort(t)
    t = t[idx]
    logit = logit[idx]
    if mode=='logit':
        if torch.sum(logit[:,1]) > 1.3:
            return 1
        else:
            return 0
    elif mode == 'vote':
        pred = torch.argmax(logit, dim=1)
        if torch.sum(pred) >= 2:
            return 1
        else:
            return 0
    elif mode == 'l':
        return torch.argmax(logit[0])
    elif mode == 'm':
        return torch.argmax(logit[1])
    elif mode == 'r':
        return torch.argmax(logit[2])
    return -1

def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Foot Dataset(classification)
    if args.dataset == 'foot':
        dataset = FootDataset(data_root=args.data_root, data_split='test', transform=get_transform('val', args.hw, crop_size=args.crop_size), val_ratio=0.)
    elif args.dataset == 'foot_aug':
        dataset = FootDatasetAug(data_root=args.data_root, data_split='val', transform=get_transform('val', args.hw, crop_size=args.crop_size), val_ratio=0.)
        
    # Pressure Dataset(Classification)
    elif args.dataset == 'pressure':
        dataset = PressureDataset(data_root=args.data_root, data_split='test', transform=get_pressure_transform('val'), val_ratio=0.)
    # 4 Point regression
    elif args.dataset == 'point':
        pass
    # rsdb Dynamic set(Classification)
    elif args.dataset == 'rsdb':
        dataset = CombinationDataset(data_root=args.data_root, data_split='test', transform=get_rsdb_transform('val'), val_ratio=0.)
    print(len(dataset))
    # Dataloader
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=None)
    
    # Model
    if args.network == 'resnet18':
        model = resnet18(pretrained=True)
        f_num = 512
    elif args.network == 'resnet34':
        model = resnet34(pretrained=True)
        f_num = 512
    elif args.network == 'resnet50':
        model = resnet50(pretrained=True)
        f_num = 2048
    elif args.network == 'resnet101':
        model = resnet101(pretrained=True)
        f_num = 2048
    elif args.network == 'resnet152':
        model = resnet152(pretrained=True)
        f_num = 2048
    model.fc = nn.Linear(f_num, NUM_CLASSES)

    # Load model
    model.load_state_dict(torch.load(args.weight_path), strict=True)
    
    # model dataparallel
    model = nn.DataParallel(model).cuda()

    # inference
    model.eval()
    logits = []
    for pack in tqdm(dl):        
        img = pack[0]
        img = img.cuda()
        
        # calc loss
        logit = model(img)
        
        # pred
        logit = logit.detach()
        logits.append(logit)
    # make prediction    
    logits = torch.cat(logits, dim=0).cpu()
    # reshape to calc per ids
    pred = logits.reshape(logits.size(0)//3, 3, logits.size(1))
    
    # read submission csv
    df = pd.read_csv(os.path.join(args.data_root, 'submission.csv'), index_col=0)

    # fill the csv
    mode = 'logit'
    for i, id_ in enumerate(dataset.ids):
        df.loc[id_, 'Target'] = predict(pred[i], dataset.types[i*3:(i+1)*3], mode)
    # save submission
    df.to_csv('./submission.csv', sep=',')

    torch.cuda.empty_cache()
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Environment, Dataset
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--data_root", default="./", type=str, help="Must contains train_annotations.csv")
    parser.add_argument("--dataset", default="foot", type=str, choices=['foot', 'foot_aug', 'pressure', 'point', 'rsdb'])

    # Output Path
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument("--weight_path", default="result/resnet.pth", type=str, help="ex. result/resnet50.pth")

    # Inference
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--network", default="resnet50", type=str,
                         choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument("--hw", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--infer_mode", default='logit', type=str, choices=['logit', 'vote', 'l','m','r'])
    
    args = parser.parse_args()
    
    # run
    run(args)
