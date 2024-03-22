import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.utils import subgraph
from spikingjelly.clock_driven import functional


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out


@torch.no_grad()
def evaluate_large(model_state_file, dataset, split_idx, eval_func, criterion, args, device="cpu", result=None):
    model = torch.load(model_state_file)
    functional.reset_net(model)
    if result is not None:
        out = result
    else:
        model.eval()

    model.to(torch.device(device))
    dataset.label = dataset.label.to(torch.device(device))
    edge_index, x = dataset.graph['edge_index'].to(torch.device(device)), dataset.graph['node_feat'].to(
        torch.device(device))
    # todo
    out = model(x, edge_index)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out


def evaluate_batch(model, dataset, split_idx, device, n, true_label, data_loader, eval_func):
    # num_batch = n // args.batch_size + 1
    # num_batch = n // args.batch_size + (n % args.batch_size > 0)

    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    valid_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask[split_idx['valid']] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[split_idx['test']] = True

    model.to(device)
    model.eval()

    idx = torch.randperm(n)
    # train_total, train_correct = 0, 0
    # valid_total, valid_correct = 0, 0
    # test_total, test_correct = 0, 0
    train_metric_sum = 0
    valid_metric_sum = 0
    test_metric_sum = 0
    train_true_list = None
    valid_true_list = None
    test_true_list = None
    train_out_list = None
    valid_out_list = None
    test_out_list = None
    with torch.no_grad():
        for i in range(num_batch):
            functional.reset_net(model)
            # idx_i, y_i = batch
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            x_i = x[idx_i].to(device)
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            y_i = true_label[idx_i].to(device)
            train_mask_i = train_mask[idx_i]
            valid_mask_i = valid_mask[idx_i]
            test_mask_i = test_mask[idx_i]

            out_i = model(x_i, edge_index_i)

            # cur_train_total, cur_train_correct = eval_func(y_i[train_mask_i], out_i[train_mask_i])
            # train_total += cur_train_total
            # train_correct += cur_train_correct

            # cur_valid_total, cur_valid_correct = eval_func(y_i[valid_mask_i], out_i[valid_mask_i])
            # valid_total += cur_valid_total
            # valid_correct += cur_valid_correct

            # cur_test_total, cur_test_correct = eval_func(y_i[test_mask_i], out_i[test_mask_i])
            # test_total += cur_test_total
            # test_correct += cur_test_correct

            # train_acc = eval_func(
            #     dataset.label[split_idx['train']], out[split_idx['train']])
            # valid_acc = eval_func(
            #     dataset.label[split_idx['valid']], out[split_idx['valid']])
            # test_acc = eval_func(
            #     dataset.label[split_idx['test']], out[split_idx['test']])
            if train_out_list is None:
                train_out_list = out_i[train_mask_i].cpu()
                valid_out_list = out_i[valid_mask_i].cpu()
                test_out_list = out_i[test_mask_i].cpu()
                train_true_list = y_i[train_mask_i].cpu()
                valid_true_list = y_i[valid_mask_i].cpu()
                test_true_list = y_i[test_mask_i].cpu()
            else:
                train_out_list = torch.cat((train_out_list, out_i[train_mask_i].cpu()), dim=0)
                valid_out_list = torch.cat((valid_out_list, out_i[valid_mask_i].cpu()), dim=0)
                test_out_list = torch.cat((test_out_list, out_i[test_mask_i].cpu()), dim=0)
                train_true_list = torch.cat((train_true_list, y_i[train_mask_i].cpu()), dim=0)
                valid_true_list = torch.cat((valid_true_list, y_i[valid_mask_i].cpu()), dim=0)
                test_true_list = torch.cat((test_true_list, y_i[test_mask_i].cpu()), dim=0)


            # train_metric_sum += train_metric
            # valid_metric = eval_func(y_i[valid_mask_i], out_i[valid_mask_i])
            # valid_metric_sum += valid_metric
            # test_metric = eval_func(y_i[test_mask_i], out_i[test_mask_i])
            # test_metric_sum += test_metric

        # train_acc = train_correct / train_total
        # valid_acc = valid_correct / valid_total
        # test_acc = test_correct / test_total
        # train_acc = train_metric_sum / len(data_loader)
        # valid_acc = valid_metric_sum / len(data_loader)
        # test_acc = test_metric_sum / len(data_loader)
        train_acc=eval_func(train_true_list,train_out_list)
        valid_acc=eval_func(valid_true_list,valid_out_list)
        test_acc=eval_func(test_true_list,test_out_list)

    return train_acc, valid_acc, test_acc, 0, None

def evaluate_batchv2(model, dataset, split_idx, args,device, n, true_label, eval_func):
    num_batch = n // args.batch_size + 1
    # num_batch = n // args.batch_size + (n % args.batch_size > 0)

    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    valid_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask[split_idx['valid']] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[split_idx['test']] = True

    model.to(device)
    model.eval()

    idx = torch.randperm(n)
    # train_total, train_correct = 0, 0
    # valid_total, valid_correct = 0, 0
    # test_total, test_correct = 0, 0
    # 应为dataloader保证每个batch样本数量想同，所以可以用累加平均的方式
    train_metric_sum = 0
    valid_metric_sum = 0
    test_metric_sum = 0
    train_true_list = None
    valid_true_list = None
    test_true_list = None
    train_out_list = None
    valid_out_list = None
    test_out_list = None
    total_time=0
    with torch.no_grad():
        # for i in tqdm(range(num_batch)):
        for i in range(num_batch):
            strat_time=time.time()
            functional.reset_net(model)
            # idx_i, y_i = batch
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            x_i = x[idx_i].to(device)
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            y_i = true_label[idx_i].to(device)
            out_i = model(x_i, edge_index_i)
            end_time=time.time()
            total_time=total_time+(end_time-strat_time)*1000
            train_mask_i = train_mask[idx_i]
            valid_mask_i = valid_mask[idx_i]
            test_mask_i = test_mask[idx_i]

            # cur_train_total, cur_train_correct = eval_func(y_i[train_mask_i], out_i[train_mask_i])
            # train_total += cur_train_total
            # train_correct += cur_train_correct

            # cur_valid_total, cur_valid_correct = eval_func(y_i[valid_mask_i], out_i[valid_mask_i])
            # valid_total += cur_valid_total
            # valid_correct += cur_valid_correct

            # cur_test_total, cur_test_correct = eval_func(y_i[test_mask_i], out_i[test_mask_i])
            # test_total += cur_test_total
            # test_correct += cur_test_correct

            # train_acc = eval_func(
            #     dataset.label[split_idx['train']], out[split_idx['train']])
            # valid_acc = eval_func(
            #     dataset.label[split_idx['valid']], out[split_idx['valid']])
            # test_acc = eval_func(
            #     dataset.label[split_idx['test']], out[split_idx['test']])
            if train_out_list is None:
                train_out_list = out_i[train_mask_i].cpu()
                valid_out_list = out_i[valid_mask_i].cpu()
                test_out_list = out_i[test_mask_i].cpu()
                train_true_list = y_i[train_mask_i].cpu()
                valid_true_list = y_i[valid_mask_i].cpu()
                test_true_list = y_i[test_mask_i].cpu()
            else:
                train_out_list = torch.cat((train_out_list, out_i[train_mask_i].cpu()), dim=0)
                valid_out_list = torch.cat((valid_out_list, out_i[valid_mask_i].cpu()), dim=0)
                test_out_list = torch.cat((test_out_list, out_i[test_mask_i].cpu()), dim=0)
                train_true_list = torch.cat((train_true_list, y_i[train_mask_i].cpu()), dim=0)
                valid_true_list = torch.cat((valid_true_list, y_i[valid_mask_i].cpu()), dim=0)
                test_true_list = torch.cat((test_true_list, y_i[test_mask_i].cpu()), dim=0)


            # train_metric_sum += train_metric
            # valid_metric = eval_func(y_i[valid_mask_i], out_i[valid_mask_i])
            # valid_metric_sum += valid_metric
            # test_metric = eval_func(y_i[test_mask_i], out_i[test_mask_i])
            # test_metric_sum += test_metric

        # train_acc = train_correct / train_total
        # valid_acc = valid_correct / valid_total
        # test_acc = test_correct / test_total
        # train_acc = train_metric_sum / len(data_loader)
        # valid_acc = valid_metric_sum / len(data_loader)
        # test_acc = test_metric_sum / len(data_loader)
        print('inf_time:{}'.format(total_time))
        train_acc=eval_func(train_true_list,train_out_list)
        valid_acc=eval_func(valid_true_list,valid_out_list)
        test_acc=eval_func(test_true_list,test_out_list)

    return train_acc, valid_acc, test_acc, 0, None



def eval_acc(true, pred):
    '''
    true: (n, 1)
    pred: (n, c)
    '''
    pred = torch.max(pred, dim=1, keepdim=True)[1]
    true_cnt = (true == pred).sum()

    return true.shape[0], true_cnt.item()


