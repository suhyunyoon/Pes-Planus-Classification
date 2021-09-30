# Pes-Planus-Classification
평발 진단

## Running example
```
python train.py --data_root /data/path --dataset foot

or

python train.py --data_root /data/path --dataset pressure --learning_rate 0.01 --network resnet34 --epoches 20
```

## Other hyperparameters

--dataset: foot, pressure, point(TODO)

--network: resnet18~152

```
python train.py --help
```
