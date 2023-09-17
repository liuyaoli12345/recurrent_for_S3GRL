# 复现S3GRL

## TimeLine

| 时间 | 内容                      | 心得                                                         |
| ---- | ------------------------- | ------------------------------------------------------------ |
| 9.17 | 了解什么是Link Prediction | 十分粗浅的了解了一下经典Link Prediction目标和意义，但是我选的论文是一个新方法，希望这一点知识有帮助 |
|      |                           |                                                              |
|      |                           |                                                              |



## 前言：小鼠鼠被背刺啦！

小鼠鼠和一个同学约好了解决同一篇论文，结果那个同学飞快跳票到一个大佬去了，还没有通知小鼠鼠，鼠鼠只好在接近ddl的时候~~捡了一篇没人要的~~选了一篇还没有被别人看中的珍珠（高情商捏）

小鼠鼠对于被背刺已经认命了，被背刺是小鼠鼠的命运...

现在小鼠鼠只能一个人尝试复现这篇论文，希望一切顺利...

## 前置知识

### Link Prediction

根据[Wikipedia](https://en.wikipedia.org/wiki/Link_prediction#Euclidean_distance)的解释，经典的Link Prediction是这样一类问题：有一张有图$G=(V,E)$,V是图中的节点，它们可能是Facebook中一个一个的联系人，可能是淘宝页面上一个一个的商品，E对应着它们之间的真实联系，比如相互认识，或是商品相似，Link Prediction的任务是根据已经给出的一组真链接（又叫做“观察到的链接”），预测一个完全的或者针对某个部分的真链接，这就意味着我们可以根据已知的好友关系分析未知的好友关系，通常，还会给出一个潜在的链接集合，我们在潜在链接中预测出真链接。

综上，最经典的Link Prediction是这样一个任务过程：在一张图上，根据已知的节点间的边，求解一个分类器，这个分类器能在给出的潜在的链接集合中，将所有链接分为两类：真链接和假链接。