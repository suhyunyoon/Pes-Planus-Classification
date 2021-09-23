import argparse
import os
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn, optim

from torch.utils.data import DataLoader

from dataset import FootDataset, get_transform

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score

NUM_CLASSES = 2

# EVALUATION
def eval_score(label, logit):
    pred = torch.argmax(logit, dim=1)

    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred, zero_division=0)
    recall = recall_score(label, pred)
    f1 = f1_score(label, pred)
    fbeta = fbeta_score(label, pred, beta=2)
    
    return acc, precision, recall, f1, fbeta

# Validation in training
def validate(model, dl, dataset, criterion):
    model.eval()
    with torch.no_grad():
        val_loss = 0.
        logits, labels = [], []
        for img, label in tqdm(dl):
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
        acc, precision, recall, f1, fbeta = eval_score(labels, logits)
        print('Validation Loss: %.6f, Accuracy: %.6f, Precision: %.6f, Recall: %.6f, F1: %.6f, F2: %.6f' % (val_loss, acc, precision, recall, f1, fbeta))
    model.train()
    

def run(args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Dataset
    dataset_train = FootDataset(data_root=args.data_root, data_split='train', transform=get_transform('train', args.hw), val_ratio=args.val_ratio)
    dataset_val = FootDataset(data_root=args.data_root, data_split='val', transform=get_transform('val', args.hw), val_ratio=args.val_ratio)
    
    # Dataloader
    train_dl = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, sampler=None)
    val_dl = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, sampler=None)
    
    # Model
    from torchvision.models import resnet50
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, NUM_CLASSES)
    # model dataparallel
    model = nn.DataParallel(model).cuda()

    # Optimizer
    criterion = nn.CrossEntropyLoss().cuda() 
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=args.nesterov)

    # Training 
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
        if e % args.verbose_interval != 0:
            train_loss /= len(dataset_train)
            logits = torch.cat(logits, dim=0).cpu()
            labels = torch.cat(labels, dim=0) 
            acc, precision, recall, f1, fbeta = eval_score(labels, logits)
            print('Epoch %d Train Loss: %.6f, Accuracy: %.6f, Precision: %.6f, Recall: %.6f, F1: %.6f, F2: %.6f' % (e, train_loss, acc, precision, recall, f1, fbeta))
        
        # Validation
        if e % args.verbose_interval == 0:
            validate(model, val_dl, dataset_val, criterion)
    
    # Save final model
    weights_path = os.path.join(args.weights_dir, args.network + '.pth')
    # split module from dataparallel
    torch.save(model.module.state_dict(), weights_path)
    torch.cuda.empty_cache()
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Environment, Dataset
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--data_root", default="/home/suhyun/dataset/계명대 동산병원_데이터", type=str, help="Must contains train_annotations.csv")

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--weights_dir", default="result/", type=str)

    # Training
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--network", default="resnet50", type=str,
                         choices=['resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument("--val_ratio", default=0.15, type=float)
    parser.add_argument("--hw", default=224, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epoches", default=50, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--nesterov", default=True, type=bool)
    parser.add_argument("--verbose_interval", default=1, type=int)
    
    args = parser.parse_args()
    
    # run
    run(args)
