# Vision Transformer
自然言語処理においてSOTAを掻っ攫っていったトランスフォーマーですがCVにおいても非常に強力なアーキテクチャであることが発覚しました。困った時はトランスフォーマーか？

## TL;DR
最近の行動認識のトレンドは３DCNNベースのものが主流であった．
しかし，３DCNNは短い距離の依存関係しかモデリングできないため，
長いクリップの動画に対して，短いクリップで推論を行っていた．
これは直感に反するため，Transformerを使って，長いシーケンスを処理したい．

Transformerで長いシーケンスを処理しようとすると，計算リソースが非常に大きくなってしまうという課題がある．

そこで，自然言語処理の既存手法であるLongformerを利用することで，解決した．

SOTA手法と比較して精度を維持しながら，学習速度を16.1倍、推論速度を5.1倍にできた．

([元論文](https://arxiv.org/pdf/2102.00719.pdf))

## Longformer
自然言語処理のモデルで，Transformerの問題点である計算リソースをO(n^2)からO(n*w)に減少させたことで，長い文章に対しても利用できるようにした．(w：ウィンドウサイズ)

計算リソースを減少させるために2つのAttention機構を提案した．

<img src='/attachment/6082690b2d97530047ce0424', width=45%>
<img src='/attachment/608269122d97530047ce0425', width=45%>

左からFull n2 attention(普通のattention)，Sliding Window Attention，Global+Sliding Attention

1. sliding window attention

    自分自身のすぐ近くだけにattentionを向ける構造をとり，ウィンドウサイズをwとし，自分自身から左右それぞれ1/2w個の単語にattentionを向ける．
    
    これによって計算リソースがO(n^2)からO(n*w)に減少させることができる．
    
    ウィンドウサイズはレイヤーごとに変えることで，精度が向上することが実験で示されている．
    \
    具体的には低位のレイヤーはウィンドウサイズを小さくし，上位のレイヤーになるにつれて大きくする．これによって低位のレイヤーはより局所的な情報を集約し，上位のレイヤーは全体的な情報を集約することで精度が向上する．
    
2. global attention

    Sliding Window Attentionと一緒に使われる．
    
    特定の単語位置の単語については，全ての単語に対してattentionを向け，全ての単語はその特定の単語位置にattentionを向けるようにする．
    
    BERTでは一番初めの単語に[CLS]というトークンを付加して，最終的な分類に使用しているため[CLS]トークンは非常に重要である.そのため，[CLS]に対してglobal attentionを適用する．
    \
    Video Transformer Networkでも同様の理由で，[CLS]に対してglobal attentionを適用する.


## アーキテクチャ
アーキテクチャの全体像を以下に示す．

![image 28.png](/attachment/608286d92d97530047ce05a8)    

各コンポーネント
1. Spatial Backbone

    アーキテクチャの`f(x)`に当たる部分で，各フレームから空間特徴量を抽出する．モジュールは2DCNNでもTransformerでも良い．

    Ablation Studyでは，バックボーンの性能が高いほど行動認識の精度も高くなった．また，重み固定よりファインチューニングした方が精度が高くなった．

2. Temporal attention-based encoder

    上述のようにLongformerを利用する．具体的にはSpatial Backboneで抽出された空間特徴量をPositional Encodingに通して入力する．
    \
    BERTと同様に[CLS]トークンをHeadにつけて，分類タスクに使う．

    Ablation Studyでは，層の深さを変化させたが，深ければ深いほど良いというわけではなかった．これは使用したデータセットのビデオが１０秒前後と短かったかららしい．

3. Classification MLP head
    2層のMLPで，Temporal attention-based encoderの[CLS]トークンを入力する．
    分類の結果が出力される．

## Ablation Experiments
Kinetics-400データセットを使用．
- フレーム数やフレームレートを変えて実験したが，精度は変わらなかった．
- 学習と検証にかかる時間について，SlowFast(SOTA手法)と比べてパラメータ数は大きいが，収束が早いため学習時が16.1倍，推論時が5.1倍高速化した．
- Attentionを以下のように定性評価した．関連領域の重みが高くなっていることからAttentionは役割を果たしていると評価できた．

![image 29.png](/attachment/60828e082d97530047ce0601)

(懸垂下降の動画で，内容的に近いフレームで重みが高くなっている．)
## 総括
自然言語処理で使われた手法をVideoでやってみましたという内容．
\
ViT以前に試されててもおかしくない内容ではある．