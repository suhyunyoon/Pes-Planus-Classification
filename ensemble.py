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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from infer import inference

NUM_CLASSES = 2

def run_each_model(args, model_name, arch, dataset, dl):
    # Model
    if arch == 'resnet18':
        model = resnet18(num_classes=NUM_CLASSES)
        f_num = 512
    elif arch == 'resnet34':
        model = resnet34(num_classes=NUM_CLASSES)
        f_num = 512
    elif arch == 'resnet50':
        model = resnet50(num_classes=NUM_CLASSES)
        f_num = 2048
    elif arch == 'resnet101':
        model = resnet101(num_classes=NUM_CLASSES)
        f_num = 2048
    elif arch == 'resnet152':
        model = resnet152(num_classes=NUM_CLASSES)
        f_num = 2048
    elif arch == 'fixmatch_resnet50':
        model = resnet50(num_classes=NUM_CLASSES)
    #model.fc = nn.Linear(f_num, NUM_CLASSES)

    # Load model
    if arch == 'fixmatch_resnet50':
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(torch.load(model_name), strict=True)
    
    # model dataparallel
    model = nn.DataParallel(model).cuda()

    # inference
    model.eval()
    logits = []
    # save logit
    logits = inference(args, model, dl) 
    return logits

def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 
    # Foot Dataset(classification)
    if args.dataset in ['foot', 'foot_aug']:
        if args.dataset == 'foot':
            dataset_val = FootDataset(data_root=args.data_root, data_split='val', transform=get_transform('val', args.hw, crop_size=args.crop_size), val_ratio=args.val_ratio)
            dataset_test = FootDataset(data_root=args.data_root, data_split='test', transform=get_transform('val', args.hw, crop_size=args.crop_size), val_ratio=0.)
        else:
            dataset_val = FootDatasetAug(data_root=args.data_root, data_split='val', transform=get_transform('val', args.hw, crop_size=args.crop_size), val_ratio=0.)
            dataset_test = FootDatasetAug(data_root=args.data_root, data_split='val', transform=get_transform('val', args.hw, crop_size=args.crop_size), val_ratio=0.)

    # Pressure Dataset(Classification)
    elif args.dataset == 'pressure':
        dataset_val = PressureDataset(data_root=args.data_root, data_split='val', transform=get_pressure_transform('val'), val_ratio=args.val_ratio)
        dataset_test = PressureDataset(data_root=args.data_root, data_split='test', transform=get_pressure_transform('val'), val_ratio=0.)
    # rsdb Dynamic set(Classification)
    elif args.dataset == 'rsdb':
        dataset_val = CombinationDataset(data_root=args.data_root, data_split='val', transform=get_rsdb_transform('val'), val_ratio=args.val_ratio)
        dataset_test = CombinationDataset(data_root=args.data_root, data_split='test', transform=get_rsdb_transform('val'), val_ratio=0.)
    
    print(len(dataset_val), len(dataset_test))
    
    # Dataloader
    val_dl = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=None)
    test_dl = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=None)
    
    # inference each model
    val_preds, test_preds = [], []
    for model_name, arch in zip(args.weights, args.networks): 
        val_logits = run_each_model(args, model_name, arch, dataset_val, val_dl)
        val_preds.append(val_logits)
        test_logits = run_each_model(args, model_name, arch, dataset_test, test_dl)
        test_preds.append(test_logits)

    import pickle
    d = {'val_logits':val_logits, 'val_preds':val_preds, 'test_logits':test_logits, 'test_preds':test_preds}
    with open("temp.pickle", 'wb') as f:
        pickle.dump(d, f)
    
    import pickle
    with open('temp.pickle', 'rb') as r:
        d = pickle.load(r)
        val_logits = d['val_logits']
        val_preds = d['val_preds']
        test_logits = d['test_logits']
        test_preds = d['test_preds']

    val_preds = torch.cat(val_preds, dim=0)
    test_preds = torch.cat(test_preds, dim=0)
    # reshape into regressor
    if args.dataset in ['foot', 'foot_aug']:
        #val_preds = val_preds.reshape(val_preds.size(0)//3, 3*val_preds.size(1))
        #test_preds = test_preds.reshape(test_preds.size(0)//3, 3*test_preds.size(1))
        val_logits = val_logits.reshape(val_logits.size(0)//3, 3*val_logits.size(1))
        test_logits = test_logits.reshape(test_logits.size(0)//3, 3*test_logits.size(1))
    elif args.dataset == 'rsdb':
        pass
    print(val_preds.shape, val_logits.shape, len(dataset_val.labels))
    print(test_preds.shape, test_logits.shape, len(dataset_test.labels))
    # Ensemble
    if args.regressor == 'logistic':
        model = LogisticRegression()
    elif args.regressor == 'randomforest':
        model = RandomForestClassifier(n_estimators=3, random_state=0)
    model.fit(val_logits, dataset_val.labels[::3])
    print(model.score(val_logits, dataset_val.labels[::3]))
    
    # predict
    sub_preds = model.predict(test_logits)
    
    # read submission csv
    df = pd.read_csv(os.path.join(args.data_root, 'submission.csv'), index_col=0)

    # fill the csv
    for i, id_ in enumerate(dataset_test.ids):
        df.loc[id_, 'Target'] = sub_preds[i]

    # save submission
    df.to_csv('./submission.csv', sep=',')

    torch.cuda.empty_cache()
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Environment, Dataset
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--data_root", default="./", type=str, help="Must contains test_annotations.csv")
    parser.add_argument("--dataset", default="foot", type=str, choices=['foot', 'foot_aug', 'pressure', 'point', 'rsdb'])

    # Output Path
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument('--weights', nargs='+', help="ex.--weights result/resnet50.pth result/fixmatch_r50.pth result/r101.pth") 

    # Inference
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--networks", type=str, nargs='+', help="ex.--networks resnet50 fixmatch_resnet50 resnet101",
                         choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'fixmatch_resnet50'])
    parser.add_argument("--regressor", default="logistic", type=str, choices=['logistic', 'randomforest'])
 
    parser.add_argument("--hw", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--val_ratio", default=0.15, type=float)
    
    args = parser.parse_args()
    
    # run
    run(args)
