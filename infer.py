import argparse
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd

import torch
from torch import nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, densenet121, densenet169, densenet201, vgg19_bn, \
                                wide_resnet50_2, wide_resnet101_2, resnext50_32x4d, resnext101_32x8d

from torch.utils.data import DataLoader

from dataset import FootDataset, PressureDataset, get_transform, get_pressure_transform
from dataset_aug import FootDatasetAug
from dataset_rsdb import CombinationDataset, get_rsdb_transform

NUM_CLASSES = 2

# ensemble metric
def predict(logit, t=None, mode='logit'):
    if t is None:
        t = torch.LongTensor([0,1,2])
    idx = torch.argsort(t)
    t = t[idx]
    logit = logit[idx]
    if mode=='logit':
        if torch.mean(logit[:,1]) >= 0.5:
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

# save predict into csv
def save_predict(args, dataset, logits, preds, labels):
    # LMR annotations exists
    if hasattr(dataset, 'types'):
        types = dataset.types
    else:
        types = [0] * len(dataset)
    data = {
        'type': types,
        'logit_0': logits[:,0],
        'logit_1': logits[:,1],
        'pred': preds,
        'label': labels
        }
    # save csv
    if args.dataset in ['foot', 'foot_aug']:
        index = [dataset.ids[i//3] for i in range(len(dataset.ids)*3)]
    elif args.dataset == 'pressure':
        index = dataset.ids
    elif args.dataset == 'rsdb':
        index = [x[0] for x in dataset.x_combinations]
    df = pd.DataFrame(data=data, index=index)
    df.to_csv(os.path.join(args.log_dir, '{}_test.csv'.format(args.network)), sep=',')

    return data

def inference(args, model, dl):
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
    
    return logits

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
        model = resnet18(num_classes=NUM_CLASSES)
    elif args.network == 'resnet34':
        model = resnet34(num_classes=NUM_CLASSES)
    elif args.network == 'resnet50':
        model = resnet50(num_classes=NUM_CLASSES)
    elif args.network == 'resnet101':
        model = resnet101(num_classes=NUM_CLASSES)
    elif args.network == 'resnet152':
        model = resnet152(num_classes=NUM_CLASSES)
    # fixmatch
    elif args.network == 'fixmatch_resnet50':
        model = resnet50(num_classes=NUM_CLASSES)
    # densenet
    elif args.network == 'densenet121':
        model = densenet121(num_classes=NUM_CLASSES)
    elif args.network == 'densenet169':
        model = densenet169(num_classes=NUM_CLASSES)
    elif args.network == 'densenet201':
        model = densenet201(num_classes=NUM_CLASSES)
    # wideresnet
    elif args.network == 'wide_resnet50_2':
        model = wide_resnet50_2(num_classes=NUM_CLASSES)
    elif args.network == 'wide_resnet101_2':
        model = wide_resnet101_2(num_classes=NUM_CLASSES)
    # resnext
    elif args.network == 'resnext50_32x4d':
        model = resnext50_32x4d(num_classes=NUM_CLASSES)
    elif args.network == 'resnext101_32x8d':
        model = resnext101_32x8d(num_classes=NUM_CLASSES)

    # Load model
    if args.network == 'fixmatch_resnet50':
        checkpoint = torch.load(args.weight_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(torch.load(args.weight_path), strict=True)
    
    # model dataparallel
    model = nn.DataParallel(model).cuda()

    # inference
    model.eval()

    # save logit
    '''
    from train import validate
    criterion=nn.CrossEntropyLoss()
    args.log_dir = './'
    validate(args, model, dl, dataset, criterion, verbose=False, save=True)
    '''
    # Forward
    logits = inference(args, model, dl) 
    preds = torch.argmax(logits, dim=1)

    # save predictions
    args.log_dir ='./'
    save_predict(args, dataset, logits, preds, torch.zeros(len(logits)))

    # reshape to calc per ids
    if args.dataset in ['foot', 'foot_aug']:
        pred = logits.reshape(logits.size(0)//3, 3, logits.size(1))
    elif args.dataset == 'rsdb':
        # get uuid, logit pair
        pred = {id_:[] for id_ in dataset.ids}
        for p, l in zip(dataset.x_combinations, logits):
            pred[p[0]].append(l)
        pred = [torch.stack(pred[uuid], dim=0) for uuid in dataset.ids]
    
    # read submission csv
    df = pd.read_csv(os.path.join(args.data_root, 'submission.csv'), index_col=0)

    # fill the csv
    mode = 'logit'
    for i, id_ in enumerate(dataset.ids):
        df.loc[id_, 'Target'] = predict(pred[i], dataset.types[i*3:(i+1)*3] if hasattr(dataset, 'types') else None, mode)

    # save submission
    df.to_csv('./{}_submission.csv'.format(args.weight_path), sep=',')

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
    parser.add_argument("--network", type=str,
                         choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169', 'densenet201',
                                    'vgg19', 'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'resnext101_32x8d'])
    parser.add_argument("--hw", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--infer_mode", default='logit', type=str, choices=['logit', 'vote', 'l','m','r'])
    
    args = parser.parse_args()
    
    # run
    run(args)
