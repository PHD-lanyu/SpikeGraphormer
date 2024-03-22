from models import *
from module.spike_graphormer import *


def parse_method(method, args, c, d, device):
    if method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn).to(device)
    elif args.method == 'spike_graphormer':
        model = SpikeGraphTransformer(feature_size=d,
                                      # model = SpikeGraphTransformer(feature_size=d, patch_size=args.patch_size, in_channels=1,
                                      num_classes=c, embed_dims=args.emb_dim, num_heads=args.trans_num_heads,
                                      mlp_ratios=2, qkv_bias=False, qk_scale=None, drop_rate=args.drop_rate,
                                      # attn_drop_rate=args.attn_drop_rate,
                                      # drop_path_rate=args.drop_path,
                                      depths=args.trans_num_layers,
                                      T=args.T, spike_mode=args.spike_mode,
                                      gnn_num_layers=args.gnn_num_layers,
                                      gnn_dropout=args.gnn_dropout, gnn_use_weight=args.gnn_use_weight,
                                      gnn_use_init=args.gnn_use_init, gnn_use_bn=args.gnn_use_bn,
                                      gnn_use_residual=args.gnn_use_residual, gnn_use_act=args.gnn_use_act,
                                      graph_weight=args.graph_weight, aggregate=args.aggregate).to(device)
    else:
        raise ValueError(f'Invalid method {method}')
    return model


def parser_add_main_args(parser):
    # # dataset and evaluation
    # # parser.add_argument('--data_dir', type=str, default='../../../NodeFormer/data/')
    # parser.add_argument('--data_dir', type=str, default='../data')
    # parser.add_argument('--dataset', type=str, default='cora')
    # parser.add_argument('--device', type=int, default=0,
    #                     help='which gpu to use if any (default: 0)')
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--cpu', action='store_true')
    # parser.add_argument('--epochs', type=int, default=1000)
    # parser.add_argument('--runs', type=int, default=1,
    #                     help='number of distinct runs')
    # parser.add_argument('--train_prop', type=float, default=.5,
    #                     help='training label proportion')
    # parser.add_argument('--valid_prop', type=float, default=.25,
    #                     help='validation label proportion')
    # parser.add_argument('--protocol', type=str, default='semi',
    #                     help='protocol for cora datasets, semi or supervised')
    # parser.add_argument('--rand_split', action='store_true',
    #                     help='use random splits')
    # parser.add_argument('--rand_split_class', action='store_true',
    #                     help='use random splits with a fixed number of labeled nodes for each class')
    # parser.add_argument('--label_num_per_class', type=int, default=20,
    #                     help='labeled nodes per class(randomly selected)')
    # parser.add_argument('--valid_num', type=int, default=500,
    #                     help='Total number of validation')
    # parser.add_argument('--test_num', type=int, default=500,
    #                     help='Total number of test')
    #
    # # model
    # parser.add_argument('--method', type=str, default='gcn')
    # parser.add_argument('--emb_dim', type=int, default=32)
    # parser.add_argument('--num_layers', type=int, default=2,
    #                     help='number of layers for deep methods')
    # parser.add_argument('--num_heads', type=int, default=1,
    #                     help='number of heads for attention')
    # parser.add_argument('--alpha', type=float, default=0.5,
    #                     help='weight for residual link')
    # parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    # parser.add_argument('--use_residual', action='store_true',
    #                     help='use residual link for each GNN layer')
    # parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    # parser.add_argument('--use_weight', action='store_true',
    #                     help='use weight for GNN convolution')
    # parser.add_argument('--use_init', action='store_true', help='use initial feat for each GNN layer')
    # parser.add_argument('--use_act', action='store_true', help='use activation for each GNN layer')
    #
    # # training
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--weight_decay', type=float, default=5e-3)
    # parser.add_argument('--dropout', type=float, default=0.5)
    #
    # # display and utility
    # parser.add_argument('--display_step', type=int,
    #                     default=1, help='how often to print')
    #
    parser.add_argument('--no_feat_norm', action='store_true',
                        help='Not use feature normalization.')
    #
    # # ours
    # parser.add_argument('--patience', type=int, default=30,
    #                     help='early stopping patience.')
    # parser.add_argument('--graph_weight', type=float,
    #                     default=0.8, help='graph weight.')
    # parser.add_argument('--ours_weight_decay', type=float,
    #                      help='Ours\' weight decay. Default to weight_decay.')
    # parser.add_argument('--ours_use_weight', action='store_true', help='use weight for trans convolution')
    # parser.add_argument('--ours_use_residual', action='store_true', help='use residual link for each trans layer')
    # parser.add_argument('--ours_use_act', action='store_true', help='use activation for each trans layer')
    # parser.add_argument('--backbone', type=str, default='gcn',
    #                     help='Backbone.')
    # parser.add_argument('--ours_layers', type=int, default=2,
    #                     help='gnn layer.')
    # parser.add_argument('--ours_dropout', type=float,
    #                     help='gnn dropout.')
    # parser.add_argument('--aggregate', type=str, default='add',
    #                     help='aggregate type, add or cat.')
    # dataset and evaluation
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=2,
                        help='number of distinct runs')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')

    parser.add_argument('--evaluate_batch', action='store_true', help='evaluate_batch or not!')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')

    # spike transformer related
    parser.add_argument('--drop_path', type=float, default=.2,
                        help='drop_path of spike transformer')
    parser.add_argument('--drop_rate', type=float, default=.2,
                        help='drop_rate of spike transformer')
    parser.add_argument('--T', type=float, default=4,
                        help='drop_path of spike transformer')
    parser.add_argument('--spike_mode', type=str, default='lif', choices=['lif', 'plif', 'f1'])
    parser.add_argument('--valid_num', type=int, default=0)
    parser.add_argument('--test_num', type=int, default=0)

    # gnn branch
    parser.add_argument('--method', type=str, default='spike_graphormer')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--use_graph', action='store_true', help='use input graph')
    parser.add_argument('--aggregate', type=str, default='add', help='aggregate type, add or cat.')
    parser.add_argument('--graph_weight', type=float, default=0.8, help='graph weight.')

    parser.add_argument('--gnn_use_bn', action='store_true', help='use batchnorm for each GNN layer')
    parser.add_argument('--gnn_use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--gnn_use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--gnn_use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--gnn_use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--gnn_num_layers', type=int, default=2, help='number of layers for GNN')
    parser.add_argument('--gnn_dropout', type=float, default=0.0)
    parser.add_argument('--gnn_weight_decay', type=float, default=1e-3)

    # all-pair attention (Transformer) branch
    parser.add_argument('--trans_num_heads', type=int, default=4, help='number of heads for attention')
    parser.add_argument('--trans_use_weight', action='store_true', help='use weight for trans convolution')
    parser.add_argument('--trans_use_bn', action='store_true', help='use layernorm for trans')
    parser.add_argument('--trans_use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--trans_use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--trans_num_layers', type=int, default=2, help='number of layers for all-pair attention.')
    parser.add_argument('--trans_dropout', type=float, help='gnn dropout.')
    parser.add_argument('--trans_weight_decay', type=float, default=1e-3)

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1000, help='mini batch training for large graphs')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience.')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to evaluate')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--save_result', action='store_true',
                        help='save result')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--save_att', action='store_true', help='whether to save attention (for visualization)')
    parser.add_argument('--model_dir', type=str, default='../model/')

    # other gnn parameters (for baselines)
    parser.add_argument('--hops', type=int, default=2,
                        help='number of hops for SGC')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')



