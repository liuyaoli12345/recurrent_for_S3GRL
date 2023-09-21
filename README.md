# 复现S3GRL

## TimeLine

| 时间 | 内容                                                        | 心得                                                         |
| ---- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| 9.17 | 了解什么是Link Prediction                                   | 十分粗浅的了解了一下经典Link Prediction目标和意义，但是我选的论文是一个新方法，希望这一点知识有帮助 |
| 9.18 | 了解了一下GNN和作者第一个比较的算法GCN                      | 还没有开始正式开始看作者改进的SGRL，搜了一下这个算法关注度貌似并不高，希望不会被作者背刺嘞 |
| 9.20 | 尝试在colab上把项目跑起来，但是卡在一个有五万多个参数的地方 | google colab是好东西，如果再大方一点就好了苦鲁西             |
| 9.21 | 继续尝试运行                                                | 现在可以在大规模阉割数据集的条件下完成至少一轮运行，今天运气比较好，google没有跑到一半掐掉我的代码（其实今天也尝试了一下kaggle和ali-cloud的教育gpu平台，kaggle gpu倒是很大方，但是cpu和ram也太小气了，阿里云的是用不了一点...） |



## 前言：小鼠鼠被背刺啦！

小鼠鼠和一个同学约好了解决同一篇论文，结果那个同学飞快跳票到一个大佬去了，还没有通知小鼠鼠，鼠鼠只好在接近ddl的时候~~捡了一篇没人要的~~选了一篇还没有被别人看中的珍珠（高情商捏）

小鼠鼠对于被背刺已经认命了，被背刺是小鼠鼠的命运...

现在小鼠鼠只能一个人尝试复现这篇论文，希望一切顺利...

## 前置知识

### Link Prediction

根据[Wikipedia](https://en.wikipedia.org/wiki/Link_prediction#Euclidean_distance)的解释，经典的Link Prediction是这样一类问题：有一张有图$G=(V,E)$,V是图中的节点，它们可能是Facebook中一个一个的联系人，可能是淘宝页面上一个一个的商品，E对应着它们之间的真实联系，比如相互认识，或是商品相似，Link Prediction的任务是根据已经给出的一组真链接（又叫做“观察到的链接”），预测一个完全的或者针对某个部分的真链接，这就意味着我们可以根据已知的好友关系分析未知的好友关系，通常，还会给出一个潜在的链接集合，我们在潜在链接中预测出真链接。

综上，最经典的Link Prediction是这样一个任务过程：在一张图上，根据已知的节点间的边，求解一个分类器，这个分类器能在给出的潜在的链接集合中，将所有链接分为两类：真链接和假链接。

### 研究采用的方法：基于图

Link Prediction有很多方法，但是这篇研究（包括助教给出的其他论文），几乎都是采用的图方法

#### 经典GNN

通用 GNN 的架构实现了以下基本[层](https://en.wikipedia.org/wiki/Layer_(deep_learning))：[[6\]](https://en.wikipedia.org/wiki/Graph_neural_network#cite_note-bronstein2021-6)

1. *置换等变*：置换等变层[将](https://en.wikipedia.org/wiki/Map_(mathematics))图的表示映射为同一图的更新表示。在文献中，排列等变层是通过图节点之间的成对消息传递来实现的。[[6\] ](https://en.wikipedia.org/wiki/Graph_neural_network#cite_note-bronstein2021-6)[[10\]](https://en.wikipedia.org/wiki/Graph_neural_network#cite_note-velickovic2022-10)直观地说，在消息传递层中，节点通过*聚合*从其直接邻居接收到的*消息*来*更新*其表示。因此，每个消息传递层都会将 GNN 的感受野增加一跳。
2. *局部池化：局部池化层通过*[下采样](https://en.wikipedia.org/wiki/Downsampling_(signal_processing))使图变粗。局部池化用于增加 GNN 的感受野，其方式与[卷积神经网络](https://en.wikipedia.org/wiki/Convolutional_neural_network)中的池化层类似。例子包括[k-近邻池](https://en.wikipedia.org/wiki/Nearest_neighbor_graph)、top-k 池、[[21\]](https://en.wikipedia.org/wiki/Graph_neural_network#cite_note-gao2019-21)和自注意力池。[[22\]](https://en.wikipedia.org/wiki/Graph_neural_network#cite_note-lee2019-22)
3. *全局池化*：全局池化层，也称为*读出*层，提供整个图的固定大小表示。全局池化层必须是排列不变的，以便图节点和边的排序中的排列不会改变最终输出。[[23\]](https://en.wikipedia.org/wiki/Graph_neural_network#cite_note-lui2022-23)示例包括逐元素求和、平均值或最大值。

上面这部分直接复制的wiki，没有看太懂，主要是**池化**这个词没有弄懂

来自chatgpt的解释

> 在图神经网络中，池化（Pooling）是一种常用的操作，用于减少图形数据的维度或尺寸，并提取出图形数据的关键特征。与传统的卷积神经网络（CNN）中的池化类似，图神经网络中的池化也可以通过降采样来减小图形的规模。
>
> 图神经网络中的池化操作针对的是图形数据，其中图形由节点（或顶点）和边组成。池化的目的是将图形数据的节点或边聚合成更小尺寸的表示，同时保留关键的结构信息和特征。
>
> 1. 局部池化（Local Pooling）：
>    局部池化通过降采样的方式来粗化图形数据。它类似于卷积神经网络中的池化层，通过减小图的规模来提高图神经网络的感知野（receptive field）。局部池化可以增加节点之间的局部连接性，并捕捉节点与邻居节点之间的局部结构信息。
>
> 在局部池化中，常用的方法包括：
>
> - k最近邻池化（k-nearest neighbors pooling）：选择每个节点的k个最近邻节点，并将它们聚合为一个新的节点。这样可以减少图的节点数量，同时保留节点之间的局部连接信息。
> - 前k池化（top-k pooling）：根据某个特定的准则，选择每个节点的前k个相邻节点进行聚合。准则可以是节点之间的相似度、权重等。这种池化方法可以在保留关键信息的同时减小图的规模。
> - 自注意力池化（self-attention pooling）：通过学习节点之间的注意力权重，对节点进行选择性的聚合。自注意力池化能够根据节点的重要性自适应地选择和聚合节点，提取关键的图结构信息。
>
> 局部池化可以帮助减小图形数据的规模，降低计算复杂度，并提取节点之间的局部结构特征。
>
> 1. 全局池化（Global Pooling）：
>    全局池化层，也称为读出层（readout layer），提供了对整个图形的固定大小表示。全局池化层的输出不依赖于图中节点的顺序，应该是置换不变的（permutation invariant）。这意味着对图中节点和边的排列顺序的变化不应该改变最终的输出。
>
> 在全局池化中，常见的方法包括：
>
> - 逐元素求和（element-wise sum）：对整个图中的节点特征进行逐元素相加，得到一个固定大小的向量表示。这种池化方法简单有效，不受节点排列顺序的影响。
> - 平均池化（mean pooling）：计算整个图中节点特征的平均值，得到一个固定大小的向量表示。平均池化可以提取图的全局特征，并保持置换不变性。
> - 最大池化（maximum pooling）：选择整个图中节点特征的最大值作为固定大小的向量表示。最大池化可以捕捉图中的最显著特征。
>
> 全局池化能够提供整个图形的固定大小表示，用于整体图形属性的建模和分析。

GNN的置换等变依赖消息传递层MPNN，一个消息传递层大致是这样的：最外层是一个更新函数，这个更新函数考虑两个值：一个是节点本身的特征，一个是通过一个消息函数计算而来的值，这个值又考虑了节点的特征，邻居节点的特征以及两个节点间边的特征

![](https://files.lsmcloud.top/blog0af3a9ccc3bf7d6a1ef1ed58957b4e0e.png)

一层消息传递就能让某个节点感受它直接的邻居，两层就能感受到隔一个的邻居，所以为了让大家都能感知到，一般会安排至少和图直径一样多的MPNN层，但是这样又会造成过度压缩和过度平滑的问题

#### GCN-图卷积网络

这是论文当中第一开始被比较的同行，GCN（The graph convolutional network ）可以被看作是卷积神经网络在图数据结构上的一种实现，

图卷积网络层引入了一个激活函数，著名激活函数包括ReLU等，激活函数根据节点的特征矩阵、图邻接矩阵、图度矩阵和参数矩阵得到节点矩阵**（这个我没有看懂）**

![](https://files.lsmcloud.top/bloge9c8a1ece53057436be9cc3dd4bf72ba.png)

## 复现代码

见google colab

### 复现中的主要挑战：算力限制

#### Google Colab使用心得

*希望那些在colab上跑stable diffusion的寄生虫可以收敛点，我谢谢你们了...*

- Google colab最长运行时间是12小时，但是实际上会经常因为资源不足大概运行到3小时就断开
- 使用Gdrive真的很方便，我想这是大家不用kaggle用colab的主要原因
- 等我有钱了一定冲会员感谢google，我觉得我大学五年五万的学费一万应该交给google，一万交给openai，一万交给cloudflare，一万交给github，五千交给github，剩下的分给bilibili、kaggle、stackoverflow、csdn，南大就配收个住宿费

### 算力真的不够！

colab为我提供了12G的运行内存和15G的显存，这个算法最搞笑的是它硬要GPU，要了它又不怎么用，显存绰绰有余，内存直接爆炸了，我有机会一定好好检查他到底怎么写的（

下面这张图是我只跑20%训练集的情况，显存怎么搞都是占1G，内存只要我敢跑大于50%的数据集它就敢爆炸（broken pipe）

![](https://files.lsmcloud.top/blogba29231cd495ceb6c0915cf88a59070f.png)

另外一个麻烦的是时间限制，即便运气超级无敌好，我也只能跑12个小时，而理想的情况是：

**利用10个seed，运行10轮，每次运行100%的数据集，并且用大于0的k_sign**

但是细想我的情况，我应该最极限只能跑：

**利用3个seed，运行3轮，每次运行20%的数据集，并且令k_sign为0**

现在是9.21，我已经成功完成了一轮：

结果是

```shell
WARNING:root:The OGB package is out of date. Your version is 1.3.5, while the latest version is 1.3.6.
Run 1 of ogbl-collab with id ogbl_collab_pos_plus_11_uvai using device cuda:0
Current arguments accepted are: {'M': 0,
 'base_gae': '',
 'batch_size': 64,
 'cache_dynamic': False,
 'calc_ratio': False,
 'checkpoint_training': False,
 'continue_from': None,
 'cuda_device': 0,
 'data_appendix': '',
 'dataset': 'ogbl-collab',
 'dataset_split_num': 1,
 'dataset_stats': False,
 'delete_dataset': False,
 'device': device(type='cuda', index=0),
 'dropedge': 0.0,
 'dropout': 0.5,
 'dynamic_test': True,
 'dynamic_train': True,
 'dynamic_val': True,
 'edge_feature': '',
 'epochs': 10,
 'eval_steps': 1,
 'fast_split': False,
 'hidden_channels': 1024,
 'init_features': 'n2v',
 'init_representation': '',
 'k_heuristic': 1,
 'k_node_set_strategy': 'intersection',
 'k_pool_strategy': 'mean',
 'keep_old': True,
 'log_steps': 1,
 'loss_fn': '',
 'lr': 0.0001,
 'm': 0,
 'max_nodes_per_hop': None,
 'model': 'SIGN',
 'n2v_dim': 16,
 'neg_ratio': 1,
 'node_label': 'zo',
 'normalize_feats': False,
 'num_hops': 1,
 'num_layers': -1,
 'num_workers': 70,
 'only_test': False,
 'optimize_sign': True,
 'pairwise': False,
 'pool_operatorwise': True,
 'pretrained_node_embedding': None,
 'profile': False,
 'ratio_per_hop': 1,
 'runs': 1,
 'save_appendix': '',
 'seed': 1,
 'sign_k': 0,
 'sign_type': 'PoS',
 'size_only': False,
 'sortpool_k': -1,
 'split_by_year': True,
 'split_test_ratio': 0.1,
 'split_val_ratio': 0.05,
 'test_multiple_models': False,
 'test_percent': 20,
 'train_gae': False,
 'train_mf': False,
 'train_mlp': False,
 'train_n2v': False,
 'train_node_embedding': False,
 'train_percent': 20,
 'use_edge_weight': False,
 'use_feature': True,
 'use_heuristic': None,
 'use_mlp': False,
 'use_valedges_as_input': True,
 'val_percent': 20}
Results will be saved in results/ogbl-collab_20230921131340_seed1
Command line input: python sgrl_run_manager.py --config configs/ogbl/ogbl_collab.json --results_json ogbl_collab_results.json
 is saved.
Filtering ogbl-collab training set to >= 2010 year
Adding validation edges to training edges
Init features using: n2v
Using cached n2v embeddings. Skipping n2v pretraining. Parameters are not counted.
Setting up Train data
Setting up Val data
Setting up Test data
Total Prep time: 8.076520744999925 sec
S3GRLLight selected
Model architecture is: S3GRLLight(
  (operator_diff): MLP(17, 1024)
  (link_pred_mlp): MLP(2048, 1024, 1)
)
Total number of parameters is 2121729
100%|█████████████████████████████| 7681/7681 [29:20<00:00,  4.36it/s]
100%|███████████████████████████████| 501/501 [02:09<00:00,  3.88it/s]
100%|███████████████████████████████| 458/458 [01:57<00:00,  3.89it/s]
Hits@20
Run: 01, Epoch: 01, Loss: 0.0279, Valid: 98.94%, Test: 68.72%
Hits@50
Run: 01, Epoch: 01, Loss: 0.0279, Valid: 99.43%, Test: 72.32%
Hits@100
Run: 01, Epoch: 01, Loss: 0.0279, Valid: 99.60%, Test: 75.54%
Hits@20
Picked Valid: 98.94, Picked Test: 68.72
Hits@50
Picked Valid: 99.43, Picked Test: 72.32
Hits@100
Picked Valid: 99.60, Picked Test: 75.54
```

有点奇怪好像比作者跑出来还要好？

![](https://files.lsmcloud.top/blog1127042e0541dcdcd41b7a342acd2f04.png)

小鼠鼠决定后面再看看，今天就回去洗澡好咯
