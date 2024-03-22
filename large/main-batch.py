import argparse
import copy
import sys
import os, random
import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import wandb
import time
from parse import parse_method, parser_add_main_args

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(
    args.device)
print('using gpu-{}'.format(args.device))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph
from torch_scatter import scatter
from spikingjelly.clock_driven import functional
from logger import Logger
from dataset import load_dataset
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, \
    to_sparse_tensor, load_fixed_splits, adj_mul
from eval import evaluate_large, evaluate_batch
from timm.optim import create_optimizer_v2
import warnings
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings('ignore')


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(args.seed)

if args.cpu:
    print('use cpu!')
    device = torch.device("cpu")
else:
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

assert not (args.dataset == 'ogbn-proteins' and args.metric == 'acc'), 'acc not support on ogbn-proteins!'

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'amazon2m']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('deezer-europe', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

# model.train()
# print('MODEL:', model)

dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']

if args.dataset in ('deezer-europe', 'ogbn-proteins'):
    if dataset.label.shape[1] == 1:
        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
    else:
        true_label = dataset.label
else:
    true_label = dataset.label

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    # 暂时先用spike-deiven-transformer的重置函数
    # model.reset_parameters()
    # functional.reset_net(model)
    ### Load method ###
    model = parse_method(args, c, d, device)

    if args.method in ['spike_graphormer']:
        # for linear prob
        # if args.linear_prob:
        # for n, p in model.module.named_parameters():
        #     if "patch_embed" in n:
        #         p.requires_grad = False
        # 将需要设置为requires_grad = False的参数找出来并设置
        # for name, param in model.named_parameters():
        #     if 'patch_embed' in name:
        #         param.requires_grad = False
        optimizer = torch.optim.Adam([
            # {'params': model.parameters(), 'weight_decay': args.trans_weight_decay},
            {'params': model.params1, 'weight_decay': args.trans_weight_decay},
            {'params': model.params2, 'weight_decay': args.gnn_weight_decay}
        ],
            lr=args.lr)
        # optimizer = create_optimizer_v2(model, opt='adamw', lr=args.lr, weight_decay=args.trans_weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_epoch = 0
    best_test = float('-inf')
    cnt_wait = 0
    num_batch = n // args.batch_size + (n % args.batch_size > 0)
    print('num_batch:{}'.format(num_batch))
    torch.cuda.empty_cache()
    model.to(device)
    model.train()
    # functional.reset_net(model)
    # with torch.autograd.detect_anomaly():
    idx = torch.range(0, n - 1, dtype=torch.long)

    data = TensorDataset(idx, true_label)
    # dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    # for batch in dataloader:
    #     inputs, labels = batch
    # all_train_time=0
    for epoch in tqdm(range(args.epochs)):
        # idx = torch.randperm(n)
        epoch_loss = 0
        # gpu_mem_alloc = 0
        # epoch_start = time.time()

        for batch_id, batch in enumerate(dataloader):
            # for batch in dataloader:
            idx_i, y_i = batch
            functional.reset_net(model)
            # with torch.autograd.detect_anomaly():
            # idx_i = idx[i * args.batch_size:(i + 1) * args.batch_size]
            train_mask_i = train_mask[idx_i]
            x_i = x[idx_i].to(device)
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            # y_i = true_label[idx_i].to(device)
            y_i = y_i.to(device)
            out_i = model(x_i, edge_index_i)
            # torch.autograd.set_detect_anomaly(True)
            # out_i = model(x_i)
            if args.dataset in ('deezer-europe', 'ogbn-proteins'):
                loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i].to(torch.float))

            else:
                out_i = F.log_softmax(out_i, dim=1)
                loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])

            epoch_loss = epoch_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % args.eval_step == 0:
            if args.evaluate_batch:
                print('\n evaluate_batch on gpu!')
                result = evaluate_batch(model, dataset, split_idx, device, n, true_label, dataloader, eval_func)
            else:
                print('\n evaluate_large on cpu!')
                model_dir = '../model/{}/'.format(args.dataset)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                model_state_file = model_dir + '{}_gpu_{}_'.format(args.method, args.device) + 'model_file.pkl'
                torch.save(model, model_state_file)
                result = evaluate_large(model_state_file, dataset, split_idx, eval_func, criterion, args, device="cpu")
            logger.add_result(run, result[:-1])
            if epoch % args.display_step == 0:
                print_str = f'Epoch: {epoch:02d}, ' + \
                            f'Loss: {loss:.4f}, ' + \
                            f'Train: {100 * result[0]:.2f}%, ' + \
                            f'Valid: {100 * result[1]:.2f}%, ' + \
                            f'Test: {100 * result[2]:.2f}%'
                print(print_str)
            # early stop
            if result[1] > best_val:
                print('best valid: {}->{}'.format(best_val, result[1]))
                best_val = result[1]
                best_epoch = epoch
                cnt_wait = 0
                best_test = result[2]
            else:
                cnt_wait = cnt_wait + 1
                print('not improved for {} patience~'.format(cnt_wait))
            if cnt_wait >= args.patience:
                print('Early stopping at {} epoch! best epoch is {}'.format(epoch, best_epoch))
                break

    logger.print_statistics(run)

logger.print_statistics()
