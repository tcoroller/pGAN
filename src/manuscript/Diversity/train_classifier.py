# pylint: disable=E1101, C0413, E0401

"""
In this script we train a classifier on VU locations in F2305 (T1) and on VU locations in the
synthetic dataset respectively. As the test set we use A2209 in both cases.
The synthetic dataset per default mirrors the class distribution of F2305: (cervical, thoracic, lumbar) <=> (0.25, 0.55, 0.2) and
cosinsts of 10'000 samples.
We use a multi-dimensional balanced accuracy for evaluating the performance of the network s.t. a balanced-accuracy
of 1/3 corresponds to random guessing.

We compare both trainings by reporting in the accompanying Jupyter notebook:
* balanced accuracies
* confusion matrices
* plotting multi-class roc
* statistical significance test between performance of both classifiers
"""

import argparse

# ROOT
# =================================================================================================================================
parser = argparse.ArgumentParser(description="Region classifier training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpu", action="store_true", help="whether running on a gpu node")
parser.add_argument("--toy", action="store_true", help="whether to train only with 1 percent of the data")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--num-workers", type=int, default=0, help="number of workers in torch.DataLoader")
parser.add_argument("--no-transform", action="store_true", help="no transform")
parser.add_argument("--n-epochs", type=int, default=3, help="number of epoches")
parser.add_argument("--tensorboard", action="store_true", help="to use tensorboard")
parser.add_argument("--experiment", type=str, default=None, help="mlflow experiment (default to be \"berlin\" with a time tag)")
parser.add_argument("--experiment-id", type=str, default=None, help="mlflow run name")
parser.add_argument("--no-pbar", action="store_true", help="if no progress bar showing")
parser.add_argument("--synth-train", action="store_true", help="whether to train on sythetic data")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
args = parser.parse_args()
no_log_keys = ["no_pbar", "experiment", "experiment_id"]
# =================================================================================================================================
# define parameters; import summary writer before other package imports to avoid crashes

import os
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch

# change to root of repository
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.insert(0, 'src')

batch_size = args.batch_size
num_workers = args.num_workers
toy_fraction = 0.01 if args.toy else 1
lr = args.lr
best_acc = 0  # best test accuracy


if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# =================================================================================================================================
# define paths for logging and saving weights
if args.experiment is None:
    args.experiment = datetime.now().strftime("%m%d-%H%M%S")

log_path = "manuscript/Diversity/classifier_logs/" + ("synth/" if args.synth_train else "f2305/")
hyp_param = f"bs_{args.batch_size}_lr_{args.lr}_transform_{not args.no_transform}/"

log_path = log_path+hyp_param
if not os.path.exists(log_path):
    os.makedirs(log_path)
ckpt = log_path+"classifier_weights.ckpt"

writer = SummaryWriter(log_dir=log_path) if args.tensorboard else None
# =================================================================================================================================
# rest of package imports
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.metrics import balanced_accuracy_score
import mlflow
from mlflow.pytorch import log_model
import torch.nn as nn
import torch.optim as optim
import torchvision

from manuscript.Train.restricted.train_dataset import F2305Dataset
from manuscript.Train.restricted.test_dataset import A2209Dataset
from manuscript.Train.restricted.synthetic_dataset import SynthDataset
from manuscript.Train.batchers import F2305Batcher, A2209Batcher, SynthBatcher

# =================================================================================================================================
# Training data
image_shape = (9, 64, 64)
transform = None

if not args.synth_train:
    print("Training using F2305...")
    traindata = F2305Dataset(shape=image_shape)
    print(f"Reading from {traindata.save_path}...")
    traindata.prepare(label_by="reader2", types=('T1',), subset_fraction=toy_fraction)

    vu_loader = DataLoader(F2305Batcher(traindata.dataset, traindata.scan_path,
                                        transform=(None if args.no_transform else transform)),
                           batch_size=batch_size, shuffle=True, num_workers=num_workers)

else:
    print("Training using synthetic dataset...")
    traindata = SynthDataset()
    print(f"Reading from {traindata.scan_path}...")

    vu_loader = DataLoader(SynthBatcher(traindata.dataset, traindata.scan_path,
                                        transform=(None if args.no_transform else transform)),
                           batch_size=batch_size, shuffle=True, num_workers=num_workers)

# =================================================================================================================================
# Test data
testdata = A2209Dataset(shape=image_shape)
testdata.prepare(types=('T1',), subset_fraction=toy_fraction)

vu_loader_test = DataLoader(A2209Batcher(testdata.dataset, testdata.scan_path), batch_size=batch_size, shuffle=False, num_workers=num_workers)

# =================================================================================================================================
# Classifier

net = torchvision.models.resnet18(pretrained=False)
# modify input layer from 3 to 9 channels
net.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# modify output classes from 1000 to 3
net.fc = nn.Linear(in_features=512, out_features=3, bias=True)

net.to(device)

criterion_vu = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


# evaluation
def balanced_accuracy(y_label, y_pred):
    # # Move to cpu
    y_pred = np.array(y_pred.cpu(), dtype=float)
    y_label = np.array(y_label.cpu(), dtype=float)
    # Delete ignore_index = -1
    y_pred = np.delete(y_pred, np.where(y_label == -1))
    y_label = np.delete(y_label, np.where(y_label == -1))
    return balanced_accuracy_score(y_label, y_pred)

def log_scalars(keys, values, step):
    for _k, _v in zip(keys, values):
        if args.tensorboard:
            writer.add_scalar(_k, _v, step)
        mlflow.log_metric(_k, _v, step=step)

# =================================================================================================================================
# training routine
def train(train_loader, test_loader, step, current_acc):

    Loss = 0
    preds_vu = torch.Tensor().to(device).long()
    labs_vu = torch.Tensor().to(device).long()

    start_time_epoch = time()
    for sample in tqdm(vu_loader, disable=args.no_pbar):
        sample['im'] = sample['im'].to(device).squeeze(1)
        sample['region'] = sample['region'].to(device).long()

        net_output = net(sample['im'])  # forward + backward + optimize
        loss = criterion_vu(net_output, sample['region'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate loss
        Loss += loss.item()

        # save prediction and labels
        _, pred_vu = net_output.max(1)
        preds_vu = torch.cat((preds_vu, pred_vu), 0)
        labs_vu = torch.cat((labs_vu, sample['region']), 0)

    # evaluation
    acc_vu = balanced_accuracy(labs_vu, preds_vu)
    Loss /= len(vu_loader)

    end_time_epoch = time()
    temp_str = f'Train - loss={Loss:.4f}; acc_vu={acc_vu:.4f}, time: {end_time_epoch-start_time_epoch}'
    print(temp_str)

    log_scalars([f"train_{s}" for s in ['loss', 'acc_vu']], [Loss, acc_vu], step)

    Loss = 0

    preds_vu = torch.Tensor().to(device).long()
    labs_vu = torch.Tensor().to(device).long()
    softs_vu = torch.Tensor().to(device).long()

    start_time_epoch = time()
    for sample_test in tqdm(vu_loader_test, disable=args.no_pbar):
        with torch.no_grad():
            sample_test['im'] = sample_test['im'].to(device)
            sample_test['region'] = sample_test['region'].to(device).long()

            net_output = net(sample_test['im'])  # forward + backward + optimize
            loss_vu_test = criterion_vu(net_output, sample_test['region'])
            loss_test = loss_vu_test

            # accumulate loss
            Loss += loss_test.item()

            # save prediction and labels
            _, pred_vu = net_output.max(1)
            soft_vu = net_output.softmax(dim=1)

            preds_vu = torch.cat((preds_vu, pred_vu), 0)
            softs_vu = torch.cat((softs_vu, soft_vu), 0)
            labs_vu = torch.cat((labs_vu, sample_test['region']), 0)

    end_time_epoch = time()

    # evaluation
    acc_vu = balanced_accuracy(labs_vu, preds_vu)
    Loss /= len(vu_loader_test)

    temp_str = f'Test - loss={Loss:.4f}; acc_vu={acc_vu:.4f}, time: {end_time_epoch-start_time_epoch}'
    print(temp_str)

    log_scalars([f"test_{s}" for s in ['loss', 'acc_vu']], [Loss, acc_vu], step)

    # Save checkpoint based on val
    if acc_vu > current_acc:
        torch.save({'model': net.state_dict(), 'acc_vu': acc_vu, "preds_vu": preds_vu, "softs_vu": softs_vu, "labs_vu": labs_vu, 'epoch': step}, ckpt)
        current_acc = acc_vu

    return current_acc

# =================================================================================================================================
# training and logging

mlflow.set_experiment(args.experiment)


with mlflow.start_run(run_name=args.experiment_id):
    for _k, _v in vars(args).items():
        if _k not in no_log_keys:
            mlflow.log_param(_k, _v)

    start_time = time()
    for epoch in tqdm(range(args.n_epochs), desc="epochs", disable=args.no_pbar):
        best_acc = train(vu_loader, vu_loader_test, epoch, best_acc)
    mlflow.log_metric("duration", time() - start_time)

    if args.tensorboard:
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(log_path)

    log_model(net, "model")
    mlflow.end_run()
