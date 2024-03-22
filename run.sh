#Cora
python ./medium/main.py --device 0 --dataset cora --method spike_graphormer --metric acc --patience 30 --seed 123 --runs 10 --epochs 1000 --no_feat_norm --rand_split_class --valid_num 500 --test_num 1000 --T 1 --aggregate add --drop_rate 0.7 --emb_dim 16 --gnn_dropout 0 --gnn_num_layers 9 --gnn_weight_decay 0.1 --graph_weight 0.3 --lr 0.1 --spike_mode plif --trans_num_heads 2 --trans_num_layers 3 --trans_weight_decay 1 --gnn_use_act --gnn_use_residual

#Film
python ./medium/main.py --device 0 --dataset film --method spike_graphormer --metric acc --patience 30  --runs 10 --epochs 1000  --T 3 --aggregate add --drop_rate 0 --emb_dim 256 --gnn_dropout 0.5 --gnn_num_layers 6  --gnn_weight_decay 0.0001 --graph_weight 0.1 --lr 0.05 --spike_mode lif --trans_num_heads 2 --trans_num_layers 1 --trans_weight_decay 0.01 --gnn_use_weight --rand_split

#Squirrel
python ./medium/main.py --device 0 --dataset squirrel --method spike_graphormer --metric acc --patience 30 --seed 123 --runs 10 --epochs 1000 --T 3 --aggregate add --drop_rate 0.6 --emb_dim 16 --gnn_dropout 0.5 --gnn_num_layers 2 --gnn_weight_decay 0.0001 --graph_weight 0.1 --lr 0.05 --spike_mode lif --trans_num_heads 2 --trans_num_layers 1 --trans_weight_decay 0.1 --gnn_use_weight

#Chameleon
python ./medium/main.py --device 0 --dataset chameleon --method spike_graphormer --metric acc --patience 30 --seed 123 --runs 10 --epochs 1000 --T 2 --aggregate add --drop_rate 0 --emb_dim 32 --gnn_dropout 0.2 --gnn_num_layers 2 --gnn_weight_decay 0.00001 --graph_weight 0.7 --lr 0.05 --spike_mode lif --trans_num_heads 1 --trans_num_layers 3 --trans_weight_decay 1 --gnn_use_weight

#Deezer
python ./medium/main.py --device 0 --dataset deezer-europe --method spike_graphormer --rand_split --metric acc --patience 30 --seed 123 --runs 10 --epochs 1000 --T 2 --aggregate cat --drop_rate 0.6 --emb_dim 64 --gnn_dropout 0.2 --gnn_num_layers 0 --gnn_weight_decay 0.0005 --graph_weight 0 --lr 0.01 --spike_mode plif --trans_num_heads 1 --trans_num_layers 1 --trans_weight_decay 0.00001 --gnn_use_residual

# amazon2m
python ./large/main-batch.py --device 0 --dataset amazon2m --method spike_graphormer --patience 30 --metric acc --seed 123 --runs 2 --epochs 1000 --eval_step 1 --batch_size 100000 --T 1 --aggregate add --drop_rate 0.1 --emb_dim 256 --gnn_dropout 0.1 --gnn_num_layers 9 --gnn_use_act --gnn_use_bn --gnn_use_init --gnn_weight_decay 0.0001 --graph_weight 0.5 --lr 0.001 --spike_mode lif --trans_num_heads 1 --trans_num_layers 1 --trans_weight_decay 0.0005

# proteins
python ./large/main-batch.py --device 0 --method spike_graphormer --patience 30 --dataset ogbn-proteins --metric rocauc --seed 123 --runs 2 --epochs 1000 --eval_step 1 --batch_size 10000 --T 2 --aggregate cat --drop_rate 0.4 --emb_dim 256 --gnn_dropout 0 --gnn_num_layers 2 --gnn_use_act --gnn_use_bn --gnn_use_init --gnn_weight_decay 0.00001 --graph_weight 0.7 --lr 0.005 --spike_mode lif --trans_num_heads 4 --trans_num_layers 2 --trans_weight_decay 1

