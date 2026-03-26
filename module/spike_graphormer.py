import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv, HeteroGraphConv
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module.ms_conv import MS_Block_Conv
from module.sps import MS_SPS


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_init=False, use_act=True):
        super(GConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x

class SpikeGraphTransformer(nn.Module):
    def __init__(
            self,
            feature_size=128,
            num_classes=11,
            embed_dims=128,
            num_heads=1,
            mlp_ratios=2,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=[6, 8, 6],
            T=2,
            attn_mode="direct_xor",
            spike_mode="lif",
            get_embed=False,
            dvs_mode=False,
            TET=False,
            cml=False,
            gnn_num_layers=2, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
            gnn_use_residual=True, gnn_use_act=True,
            graph_weight=0.8, aggregate='add'

    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.TET = TET
        self.dvs = dvs_mode
        # self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        self.embed_dims = embed_dims
        self.mlp_ratios = mlp_ratios

        # graph branch
        if self.graph_weight > 0:
            self.graph_conv = GConv(feature_size, num_classes, gnn_num_layers, gnn_dropout, gnn_use_bn,
                                    gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)
            if aggregate == 'add':
                self.fc = nn.Linear(num_classes, num_classes)
            elif aggregate == 'cat':
                self.fc = nn.Linear(2 * num_classes, num_classes)
            else:
                raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(feature_size, embed_dims))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(embed_dims))
        self.activation = F.relu
        self.drop_rate = drop_rate

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                )
                for j in range(depths)
            ]
        )

        # setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)
        self.params1 = list(self.fcs.parameters())
        self.params1.extend(list(self.bns.parameters()))
        self.params1.extend(list(blocks.parameters()))
        self.params1.extend(list(self.head_lif.parameters()))
        self.params1.extend(list(self.head.parameters()))
        if self.graph_weight > 0:
            self.params2 = list(self.graph_conv.parameters())
            self.params2.extend(list(self.fc.parameters()))
        else:
            self.params2 = []

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv1d):
            # trunc_normal_(m.weight, std=0.02)
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, edge_index=None, hook=None):

        T, B, D = x.shape
        x = self.fcs[0](x)
        # if self.use_bn:
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        block = getattr(self, f"block")
        for blk in block:
            x, _, hook = blk(x, hook=hook)
        return x, hook

    def forward(self, x, edge_index=None, hook=None):

        if len(x.shape) < 3:
            x1 = (x.unsqueeze(0)).repeat(int(self.T), 1, 1)
        else:
            x1 = x.transpose(0, 1).contiguous()

        x1, hook = self.forward_features(x1, hook=hook)
        x1 = self.head_lif(x1)
        if hook is not None:
            hook["head_lif"] = x1.detach()
        x1 = self.head(x1)  # T,B,D
        if not self.TET:
            x1 = x1.mean(0)
        if self.graph_weight > 0:
            x2 = self.graph_conv(x, edge_index)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
            x = self.fc(x)
        else:
            x = x1
        # return x, hook
        return x