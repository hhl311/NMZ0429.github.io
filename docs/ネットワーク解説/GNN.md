# Graph Attention Network 解説

# どんなモデル？
**ノードの特徴表現を通常の畳み込みではなく，周辺ノードの重み付き和で表現**する

---> 要するに，**エッジ表現をシンプルにattention weight で表現**

異なるサイズの近傍を扱いながら、近傍内の異なるノードに異なる重要度を（暗黙的に）割り当てることができる
グラフ構造全体を事前に知ることに依存しないなど、**従来のスペクトルベース**のアプローチの理論的な問題の多くを解決
エッジを単純に表現できるため計算速度もそこそこ改善

## 目的
GNN で近年頻繁に使われている **GATs(Gragh Attention Networks)** について理解する

## 前提
[論文はこちら ICLR 2018](https://arxiv.org/pdf/1710.10903.pdf)

## 背景
graphはノードに加えてエッジの情報も重要
しかしながら，エッジの潜在表現を作っていては計算速度が遅い
-> エッジを単純にattentionの重みで表現する

## 手法
GATs の概要図を以下に示す

![](http://namazu.tokyo/wp-content/uploads/2021/02/a71d363f2a2cb9a982115d927ecc6bdd-300x192.png)

ノードの更新式は以下の式で表現

![](http://namazu.tokyo/wp-content/uploads/2021/02/52afea8f9df1224d6401c01a09dbdf4c-300x53.png)

実際には Multi-Head Attention (head数=K) を計算する (|| は concat を意味する)

![](http://namazu.tokyo/wp-content/uploads/2021/02/87601e2ca45b477c1263251ad607bb84-300x45.png)

## もう少し噛み砕いて

<img src="https://data.dgl.ai/tutorial/gat/gat.png" width=500>

![](http://namazu.tokyo/wp-content/uploads/2021/02/b67394fa07b0df3ef633a34a6a4fb7b2-300x178.png)

各々の式の解釈

    (1) node i の特徴量を linear層で変換

    (2) 隣接するnode i と j を concatenate (式の || は concatenate) し，linear + LeReLU でエネルギー関数を計算
        dot-product attentionではなく additive attention(加法注意) と呼ばれるもの

    (3) 重みに変換

    (4) node i の Layer (l+1) の特徴量は attentionにより計算した(3) と (1) の重み付き和で表現

## 実験
### Transductive Learning
学習に使うグラフとテストに使うグラフが同じ
(すでに存在しているが，ラベルが未知のノードを予測するなど)
- 3つのcitation  network  benchmark  datasetを使用

### Inductive Learning
学習に使うグラフとテストに使うグラフが異なる場合がある
(新しいエッジやノードの予測，別のグラフでの評価など)
- タンパク質の相互作用を表すdataset (PPI dataset)

**一般的にInductiveな設定の方が汎化性能が求められる**



![](http://namazu.tokyo/wp-content/uploads/2021/02/b67394fa07b0df3ef633a34a6a4fb7b2-1-300x178.png)

- 可視化例

**node間の強い結び付きは大きい重みで表現するというシンプルな方法が表現できている**

![](http://namazu.tokyo/wp-content/uploads/2021/02/7930aee3e0d195686475ceff67a6d077-300x192.png)

- epochs毎の node classification の可視化例(余談)

![](http://namazu.tokyo/wp-content/uploads/2021/02/4083b1cdb8ddbe4870faecc1766cc509-300x200.gif)


## 参考資料
https://www.slideshare.net/takahirokubo7792/graph-attention-network
