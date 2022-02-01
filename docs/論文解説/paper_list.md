# 読んだ論文まとめ（随時更新）

## 距離学習

1. [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](https://arxiv.org/pdf/1909.05235.pdf)
    * Classifcationとmetric learningを一つのロスで行う手法の提案。Triplet lossをスムージングしていくと cross entropyになることが証明された（本質的には同じだったらしい）
2. [Visual Explanation for Deep Metric Learning](https://arxiv.org/abs/1909.12977)
    * 距離学習モデルの可視化
3. [Embedding Expansion: Augmentation in Embedding Space for Deep Metric Learning](https://arxiv.org/abs/2003.02546)
    * マイニングにいくつかシンプルなルールベースの最適化を行うことでどの距離損失関数に対しても精度向上が確認された
4. [Moving in the Right Direction: A Regularization for Deep Metric Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohan_Moving_in_the_Right_Direction_A_Regularization_for_Deep_Metric_CVPR_2020_paper.pdf)
    * 深層距離学習の正則化手法の比較、triplet lossの危険性について書いてあった
5. [Deep Metric Learning via Adaptive Learnable Assessment](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Deep_Metric_Learning_via_Adaptive_Learnable_Assessment_CVPR_2020_paper.html)
    * マイニングのルールを学習ベースに置き換えエピソードベースの学習スキームを採用した

## 動画タスク

1. [Spatiotemporal Contrastive Video Representation Learning](https://arxiv.org/abs/2008.03800)
    * SimCLRを動画分類タスクに適用した、導入したい。
2. [Predicting Video with VQVAE](https://arxiv.org/pdf/2103.01950.pdf)
    * kinetics600で65%、teacher_forcing likeな方法が取れる
3. [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
    * Transformerによる動画分類器、色々新しい。
4. [VideoMix: Rethinking Data Augmentation for Video Classification](https://arxiv.org/pdf/2012.03457.pdf)
    * VideoMixという動画行動認識のための新しいDAを提案、T-VideoMixという手法が導入できそう。
5. [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf)
    * 3DCNN重すぎ問題をTSMというモジュールを2DCNNに挿入することで代用した、TSMはパラメータ0なので2DCNNのcomplexityのままらしい。

### 未来予想

1. [Improved Conditional VRNNs for Video Prediction](https://arxiv.org/abs/1904.12165)
   * Variational Recurrent Autoencoder で動画の未知のフレームを予測する。典型的なRAEで非常にシンプル、生成するならこれでしょ。
2. [Video Prediction via Example Guidance](https://arxiv.org/abs/2007.01738)
    * 読み終わってない、動画未来予測で初のマルチモーダルモデル
3. [Predictive Learning: Using Future Representation Learning Variantial Autoencoder for Human Action Prediction](https://arxiv.org/pdf/1711.09265.pdf)
    * RGBとOptical Flowの2ストリーム

## 学習手法

1. [Invariant Information Clustering for Unsupervised Image Classification and Segmentation](https://arxiv.org/pdf/1807.06653.pdf#page9)
    * 教師無し+予測値を直出力できるモデルの学習方法、ノイズに強い
2. [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
    * SimCLRベースで教師有学習を行う
3. [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882v5)
    * SwAV
4. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
    * SimCLR
6. [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
    * 学習ベースでDAを行う
7. [What Makes Training Multi-modal Classification Networks Hard?](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.pdf)

## その他

1. [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579v1)
    * ResNetの学習とスケーリング方法
2. [An annotation-free whole-slide training approach to pathological classification of lung cancer types using deep learning](https://www.nature.com/articles/s41467-021-21467-y#Abs1)
    * ユニファイドメモリ（UM）メカニズムといくつかのGPUメモリ最適化手法で画像の縮小を回避する
3. [Prototypical Contrastive Learning of Unsupervised Representations](https://arxiv.org/abs/2005.04966)
    * EMアルゴリズムベースのクラスタリング、クラスターが収束しずらくなる様に距離関数を変更していき過学習を抑制する
