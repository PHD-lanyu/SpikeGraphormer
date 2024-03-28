# SpikeGraphormer
This repo is for source code of paper "SpikeGraphormer: A High-Performance Graph Transformer with Spiking Graph Attention". 

# A Gentle Introduction
<div align="center">
  <img src="https://github.com/PHD-lanyu/SpikeGraphormer/blob/main/framework.png" alt="Framework">
</div>

Recently, Graph Transformers have emerged as a promising solution to alleviate the inherent limitations of Graph Neural Networks (GNNs) and enhance graph representation performance. Unfortunately, Graph Transformers are computationally expensive due to the quadratic complexity inherent in self-attention when applied over large-scale graphs, especially for node tasks. In contrast, spiking neural networks (SNNs), with event-driven and binary spikes properties, can perform energy-efficient computation. In this work, we propose a novel insight into integrating SNNs with Graph Transformers and design a Spiking Graph Attention (SGA) module. The matrix multiplication is replaced by sparse addition and mask operations. The linear complexity enables all-pair node interactions on large-scale graphs with limited GPU memory. To our knowledge, our work is the first attempt to introduce SNNs into Graph Transformers. Furthermore, we design SpikeGraphormer, a Dual-branch architecture, combining a sparse GNN branch with our SGA-driven Graph Transformer branch, which can simultaneously perform all-pair node interactions and capture local neighborhoods. SpikeGraphormer consistently outperforms existing state-of-the-art approaches across various datasets and makes substantial improvements in training time, inference time, and GPU memory cost (10 ~ 20 × lower than vanilla self-attention). It also performs well in cross-domain applications (image and text classification).
## Environment Settings
> conda create --name Spike_Graphormer python=3.8 \
conda activate Spike_Graphormer \
pip install spikingjelly==0.0.0.0.12 \
pip install torch_geometric==2.4.0 \
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp38-cp38-linux_x86_64.whl \
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_sparse-0.6.18%2Bpt21cu121-cp38-cp38-linux_x86_64.whl \
pip install pandas==1.2.4 \
pip install networkx==2.6.1 \
pip install scikit_learn==1.1.3 \
pip install scipy==1.6.2 \
pip install googledrivedownloader==0.4 \
pip install ogb==1.3.1 \
pip install timm==0.6.12 \
pip install torchinfo \
pip install wandb 

GeForce RTX 4090  24GB GPU Memory, 256GB Main Memory.
# Usage
Fisrt, download the datasets from [this URL](https://pan.baidu.com/s/1t-EOvsRiWil3CaZGk82MpA?pwd=26uy), unzip the data.zip file, and place the data folder in the root directory \
Second, go into the root directory, and then you can use the commends in run.sh to run our model. 


# License
This repository is released under the Apache 2.0 license.

# Acknowledgement
This repository is built upon [NodeFormer](https://github.com/qitianwu/NodeFormer), and [Spike-driven Transformer](https://github.com/BICLab/Spike-Driven-Transformer), we thank the authors for their open-sourced code.


# Citation
If you find our codes useful, please cite our work. Thank you.
```bibtex
@article{sun2024spikegraphormer,
  title={SpikeGraphormer: A High-Performance Graph Transformer with Spiking Graph Attention},
  author={Sun, Yundong and Zhu, Dongjie and Wang, Yansong and Tian, Zhaoshuo and Cao, Ning and O'Hared, Gregory},
  journal={arXiv preprint arXiv:2403.15480},
  year={2024}
}
```
Of course, you can also cite the paper of NodeFormer and Spike-driven Transformer.

```bibtex
  @article{wu2022nodeformer,
  title={Nodeformer: A scalable graph structure learning transformer for node classification},
  author={Wu, Qitian and Zhao, Wentao and Li, Zenan and Wipf, David P and Yan, Junchi},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={27387--27401},
  year={2022}
}

```bibtex
  @article{yao2024spike,
  title={Spike-driven transformer},
  author={Yao, Man and Hu, Jiakui and Zhou, Zhaokun and Yuan, Li and Tian, Yonghong and Xu, Bo and Li, Guoqi},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
}
```

# Contact
If you have any questions, please feel free to contact me with hitffmy@163.com
