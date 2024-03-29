# Faster R-CNN まとめ

* 物体検出のモデルの一つ
* 精度が高く速度が遅い
  * 速度が必要なく、精度が欲しい時に使うモデル

## 他の物体検出のモデルとの比較

![物体検出モデルの精度x速さ](./attachment/モデルの比較.png)

| モデル名 | Yolo v3 | SSD  | Faster R-CNN | CenterNet |
| ---- | ------- | ---- | ------------ | --------- |
| 精度   | 悪い      | 少し悪い | 少し良い         | 良い        |
| 速さ   | 早い      | 少し早い | 少し遅い         | 遅い        |

## 歴史

### R-CNN

1. Selective Searchを用いて物体らしいところを2000個ほど検出
   * ここは人が作ったアルゴリズム
   * テクスチャ、色などで物体を識別する
2. 検出したところをクロップし、一定の大きさにしてCNNで特徴量抽出
3. SVMでクラス分類、回帰でBBOXの位置調整

### Fast R-CNN

#### R-CNNの問題点

* R-CNNはCNN,SVM,BBOX回帰を別々に学習しなければならない
* 実行時間が遅い

#### 改善方法

* RoI poolingを導入することで、Selective Search→CNN時の重複部分を無くす
  * 特徴抽出のためのCNNの後に、幅の大きさが可変的なPooling層を入れる
  * こうすることで、不均一なサイズの入力に対し固定サイズの特徴ベクトルを得る
* CNN,SVM,BBOX回帰を単一のモデルにする
* 流れはR-CNNと同じ

### Faster R-CNN

#### Fast R-CNNの問題点

* Selective Searchが遅い
* Selective Search（候補領域の提案）は学習しない

#### 改善点

* Selective SearchをRegion Proposal Networkに置換
  * 小さなCNNをかけた後、縦横比・サイズが異なるAnchorBOX（BBOX）で特徴抽出
  * その後、FC層で物体か否かとBBOXの位置を回帰
* 高速化・完全なEnd2Endによる高精度化を達成

### 類似モデルなど（詳細は割愛）

#### Yolo

* 画像をグリッド状に分割する
* リアルタイム性が必要なものに使用

#### SGD

* 畳み込みの異なる段階の特徴量を使用する
* 低解像度・小さな画像に強い

#### Cascade R-CNN

* Faster R-CNNの改良
* IoUの閾値を段階的に上げていく
* 精度は良いが、計算資源を要求

## モデルの構造

![論文画像](./attachment/FasterRCNNの構造.png)

1. 画像を畳み込み層に入れ、特徴マップを出す
   * image - conv_layers - feature_maps
2. 特徴マップにnxn（論文ではn=3）のConv層をかけた後、FC層のような畳み込み層（kernel=1,stride=1にすることで総結合っぽくする）で物体らしさと位置（x1,x2,w,h）を出力
   * Region Proposal Network
3. 1の特徴マップと2の提案領域から、BBOXを計算・分類する
   * RoI pooling
   * classifier

## 論文での工夫点

* RPN部分では、そのまま学習すると負のサンプルの影響が大きいため、正負の割合が１対１になるようにサンプリングする
* 上記のモデルの構造だと、厳密な学習ではなく近似解になる
  * こちらの方が学習は早い
  * 論文ではRPNとFast R-CNN部分を交互に学習していた

# CenterNetとは

* CenterNetという名前の2種類モデルがあるが、Object as Pointsの方

## 歴史

* Heatmapベースの手法
  * Faster R-CNNなどより後
  * アンカーの代わりにHeatmapを用いる
* CornerNet（2018/8）が源流
  * アンカー→BBOX回帰ではなく、Heatmapで左上・右下をキーポイントとして学習する

## モデルの構造

* アンカーではなく、ヒートマップでクラスごとに物体の中心を予測
* 高さ・幅・クラス・などの特性は、各位置で回帰する

1. 画像を畳み込み層のみで出来たネットワークに入れ、ヒートマップを作成する
2. ヒートマップのピーク（周囲8箇所と比較して）を物体の中心とする
3. 必要に応じて、物体の中心の特徴ベクトルから、物体のサイズ・奥行き・向き・姿勢などを推定する
