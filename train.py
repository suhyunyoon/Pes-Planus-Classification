import argparse
import os
from tqdm import tqdm
from datetime import datetime
import pandas as pd

import torch
from torch import nn, optim

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, densenet121, densenet169, densenet201, vgg19_bn, \
                                wide_resnet50_2, wide_resnet101_2, resnext50_32x4d, resnext101_32x8d

from torch.utils.data import DataLoader

from dataset import FootDataset, PressureDataset, get_transform, get_pressure_transform
from dataset_aug import FootDatasetAug
from dataset_rsdb import CombinationDataset, get_rsdb_transform

from infer import save_predict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score

NUM_CLASSES = 2


# EVALUATION
def eval_score(label, logit):
    pred = torch.argmax(logit, dim=1)
    # evaluation metrics
    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, zero_division=0)
    recall = recall_score(label, pred)
    f1 = f1_score(label, pred)
    fbeta = fbeta_score(label, pred, beta=2)
    
    return acc, precision, recall, f1, fbeta

# Validation in training
def validate(args, model, dl, dataset, criterion, verbose=False, save=False): 
    model.eval()
    with torch.no_grad():
        val_loss = 0.
        logits, labels = [], []
        for pack in tqdm(dl):
            img, label = pack[0], pack[1]
            labels.append(label)
            img, label = img.cuda(), label.cuda()

            # forward
            logit = model(img)
            loss = criterion(logit, label)
            
            # loss, acc
            val_loss += loss.detach().cpu()
            logit = torch.sigmoid(logit).detach()
            logits.append(logit)
        # Eval
        val_loss /= len(dataset)
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0)
        preds = torch.argmax(logits, dim=1)
        if verbose: 
            acc, precision, recall, f1, fbeta = eval_score(labels, logits)
            print('Validation Loss: %.6f, Accuracy: %.6f, Precision: %.6f, Recall: %.6f, F1: %.6f, F2: %.6f' % (val_loss, acc, precision, recall, f1, fbeta))
        if save:
            data = save_predict(args, dataset, logits, preds, labels)
        else:
            data = None
    model.train()

    return val_loss, data
    

def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Foot Dataset(classification)
    if args.dataset in ['foot', 'foot_aug']:
        dataset_train = FootDataset(data_root=args.data_root, data_split='train', transform=get_transform('train', args.hw, crop_size=args.crop_size), val_ratio=args.val_ratio if args.dataset=='foot' else 0.)
        if args.dataset == 'foot':
            dataset_val = FootDataset(data_root=args.data_root, data_split='val', transform=get_transform('val', args.hw, crop_size=args.crop_size), val_ratio=args.val_ratio)
        else:
            dataset_val = FootDatasetAug(data_root=args.data_root, data_split='val', transform=get_transform('val', args.hw, crop_size=args.crop_size), val_ratio=0.)
    # Pressure Dataset(Classification)
    elif args.dataset == 'pressure':
        dataset_train = PressureDataset(data_root=args.data_root, data_split='train', transform=get_pressure_transform('train'), val_ratio=args.val_ratio)
        dataset_val = PressureDataset(data_root=args.data_root, data_split='val', transform=get_pressure_transform('val'), val_ratio=args.val_ratio)
    # 4 Point regression
    elif args.dataset == 'point':
        pass
    # rsdb Dynamic set(Classification)
    elif args.dataset == 'rsdb':
        dataset_train = CombinationDataset(data_root=args.data_root, data_split='train', transform=get_rsdb_transform('train'), val_ratio=args.val_ratio)
        dataset_val = CombinationDataset(data_root=args.data_root, data_split='val', transform=get_rsdb_transform('val'), val_ratio=args.val_ratio)
    
    print(len(dataset_train), len(dataset_val))
    # Dataloader
    train_dl = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, sampler=None)
    val_dl = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=None)
    
    # Model
    if args.network == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
        f_num = 512
    elif args.network == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
        f_num = 512
    elif args.network == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
        f_num = 2048
    elif args.network == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
        f_num = 2048
    elif args.network == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
        f_num = 2048
    elif args.network == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
        f_num = 1024
    elif args.network == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
        f_num = 1664
    elif args.network == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
        f_num = 1920
    elif args.network == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=args.pretrained)
        f_num = 2048
    elif args.network == 'wide_resnet101_2':
        model = wide_resnet101_2(pretrained=args.pretrained)
        f_num = 2048
    elif args.network == 'resnext50_32x4d':
        model = resnext50_32x4d(pretrained=args.pretrained)
        f_num = 2048
    elif args.network == 'resnext101_32x8d':
        model = resnext101_32x8d(pretrained=args.pretrained)
        f_num = 2048
    # resnet
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(f_num, NUM_CLASSES)
    # densenet
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Linear(f_num, NUM_CLASSES)
    # model dataparallel
    model = nn.DataParallel(model).cuda()

    # Optimizer
    criterion = nn.CrossEntropyLoss().cuda() 
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    # Training
    weights_name = '{}_{}_{}_lr{}_b{}_e{}_hw{}'.format(args.dataset, args.network, args.optimizer, args.learning_rate, args.batch_size, args.epoches, args.crop_size)
    os.makedirs(os.path.join(args.weights_dir, weights_name), exist_ok=True)
    best_val_loss = 999999999.
    for e in range(1, args.epoches+1):
        model.train()
        train_loss = 0.
        logits, labels = [], []
        for img, label in tqdm(train_dl):
            
            # memorize labels
            labels.append(label)
            img, label = img.cuda(), label.cuda()
            
            # calc loss
            logit = model(img)
            loss = criterion(logit, label)
            # training
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # loss, acc
            train_loss += loss.detach().cpu()
            logit = torch.sigmoid(logit).detach()
            logits.append(logit)
            
        # Training log 
        train_loss /= len(dataset_train)
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0) 
        acc, precision, recall, f1, fbeta = eval_score(labels, logits)
        print('Epoch %d Train Loss: %.6f, Accuracy: %.6f, Precision: %.6f, Recall: %.6f, F1: %.6f, F2: %.6f' % (e, train_loss, acc, precision, recall, f1, fbeta))
        
        # Validation 
        val_loss, _ = validate(args, model, val_dl, dataset_val, criterion, verbose=True)
        # lr scheduling
        scheduler.step(val_loss)
        # save best model
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            weights_path = os.path.join(args.weights_dir, weights_name, '{}_best.pth'.format(e)) 
            # split module from dataparallel
            torch.save(model.module.state_dict(), weights_path)
            print("Best Model Saved.")
        else:
            weights_path = os.path.join(args.weights_dir, weights_name, '{}.pth'.format(e)) 
            # split module from dataparallel
            torch.save(model.module.state_dict(), weights_path)
    print(weights_name, "Saved.")
    
    print('Final Validation: ', end='')
    val_loss, _ = validate(args, model, val_dl, dataset_val, criterion, verbose=True, save=True)
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
    parser.add_argument("--log_dir", default="log/", type=str)
    parser.add_argument("--weights_dir", default="result/", type=str)

    # Training
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--network", default="resnet152", type=str,
                         choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169', 'densenet201',
                                    'vgg19', 'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'resnext101_32x8d'])
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("--val_ratio", default=0.15, type=float)
    parser.add_argument("--hw", default=256, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epoches", default=20, type=int)
    parser.add_argument("--optimizer", default='sgd', type=str, choices=['sgd', 'adam', 'adamw'])
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--nesterov", default=True, type=bool)
    parser.add_argument("--verbose_interval", default=3, type=int)
    
    args = parser.parse_args()
    
    # run
    run(args)
