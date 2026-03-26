import argparse
import copy
import os
import random
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from parse import parse_method, parser_add_main_args

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)
import time
from sklearn.neighbors import kneighbors_graph

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(
    args.device)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits
from dataset import load_nc_dataset
from logger import Logger
from torch_geometric.utils import (add_self_loops, remove_self_loops,
                                   to_undirected)
from spikingjelly.clock_driven import functional

warnings.filterwarnings('ignore')


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
    print('using cpu！')
else:
    print('using gpu-{}'.format(args.device))
    device = torch.device("cuda:" + str(0)
                          ) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_nc_dataset(args, args.data_dir, args.no_feat_norm)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

dataset_name = args.dataset

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

if args.dataset in ('mini', '20news'):
    adj_knn = kneighbors_graph(dataset.graph['node_feat'], n_neighbors=args.knn_num, include_self=True)
    edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
    dataset.graph['edge_index']=edge_index

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

_shape = dataset.graph['node_feat'].shape
print(f'features shape={_shape}')

# whether or not to symmetrize
if args.dataset not in {'deezer-europe'}:
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)

print(f"num nodes {n} | num classes {c} | num node feats {d}")
logger = Logger(args.runs, args)

for run in range(args.runs):
    ### Load method ###
    model = parse_method(args.method, args, c, d, device)

    # using rocauc as the eval function
    if args.dataset in ('deezer-europe'):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    eval_func = eval_acc
    model.train()

    ### Training loop ###
    patience = 0
    if args.method in ['spike_graphormer'] and args.graph_weight > 0:
        optimizer = torch.optim.Adam([
            {'params': model.params1, 'weight_decay': args.trans_weight_decay},
            {'params': model.params2, 'weight_decay': args.gnn_weight_decay}
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(
            model.params1, weight_decay=args.trans_weight_decay, lr=args.lr)

    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    # model.reset_parameters()

    best_val = float('-inf')
    patience = 0
    x = dataset.graph['node_feat']
    edge_index = dataset.graph['edge_index']
    edge_weight = dataset.graph['edge_weight'] if 'edge_weight' in dataset.graph else None
    all_train_time = 0
    for epoch in range(args.epochs):
        functional.reset_net(model)
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        if args.dataset in ('deezer-europe'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(
                    dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()
        result = evaluate(model, dataset, split_idx,
                          eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()
print(results)
out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

