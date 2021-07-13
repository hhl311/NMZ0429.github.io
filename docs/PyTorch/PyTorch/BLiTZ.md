# BLiTZ_+_PyTorch_Lightning

ここではライトニングで組み込みから推論までの一連の流れを説明します、理論は別ページにて追記。

## 簡単4ステップで組み込み

1. [ベイズ化するレイヤーを決める](#ベイズ化するレイヤーを決める)
2. [ベイズ化するレイヤーをBlitzモジュールに置き換える、デコレーターも忘れずに](#ベイズ化するレイヤーをBlitzモジュールに置き換える、デコレーターも忘れずに)
3. [ロス計算をELBO推定に変える](#ロス計算をELBO推定に変える)
4. [Optional: 評価用関数を作成](#Optional:-評価用関数を作成)

### 1.ベイズ化するレイヤーを決める

#### 組み込み済みのレイヤー

  * [BayesianModule](https://github.com/piEsposito/blitz-bayesian-deep-learning/blob/master/doc/layers.md)
  * [BayesianLinear](#class-BayesianLinear)
  * [BayesianConv1d](#class-BayesianConv1d)
  * [BayesianConv2d](#class-BayesianConv2d)
  * [BayesianConv3d](#class-BayesianConv3d)
  * [BayesianLSTM](#class-BayesianLSTM)
  * [BayesianGRU](#class-BayesianGRU)
  * [BayesianEmbedding](#class-BayesianEmbedding)

上記以外の処理をベイズ化する場合は`BayesianModule`を継承したモジュールを自分で作る必要があります。


### 2.ベイズ化するレイヤーをBlitzモジュールに置き換える、デコレーターも忘れずに

1. モデルのフォワードを定義する`nn.Module`に`@variational_estimator`をつける
2. モジュール内のレイヤーを置き換える
3. 必要であれば重みの事前分布を明記する。デフォルトは混合ガウス分布。

下記例は動画認識モデル`CNNLSTM`をベイズ化する。`forward`関数は変更する必要は無い。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLSTM, BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class BayesianCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5,5))
        self.conv2 = BayesianConv2d(6, 16, (5,5))
        self.fc1   = BayesianLinear(1024, 512)
        self.fc2   = BayesianLinear(512, 128)
        self.fc3   = BayesianLinear(128, 9)
        self.lstm = BayesianLSTM(
                    in_features=1024,
                    out_features=1024
                    )

    def forward(self, x):
        n, t, c, h, w = out.shape
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view([n, t, c, h, w])
        out = self.lstm(out, batch_first=True)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
```
デコレーターによってモジュールに訓練時に必要になるメソッドを追加する。
この二点を守ればモデル側で重みのサンプリングを行ってくれる。計算量は増えます。

別のベイジアンモデルからファインチューニングする時や重み共有など、重みの事前分布を知っている時は

```python
 self.weight_prior_dist = PriorWeightDistribution(
                          self.prior_pi,
                          self.prior_sigma_1,
                          self.prior_sigma_2,
                          dist = self.prior_dist # PriorWeightDistribution インスタンス
                          )

self.fc1   = BayesianLinear(1024, 512, prior_dist = self.weight_prior_dist)

```
として分布をインスタンス化して渡すことができる。が、使うことは無い。

### 3.ロス計算をELBO推定に変える

ここからは`Lightning`のコードを変える。前提として**自動最適化をオフにする必要がある**。

```python
self.automatic_optimization = False
```

次に、`training_step`内のロス計算コード、例えば

```python
out = self.model(x) # self.model()は上記のCNNLSTM
loss = F.cross_entropy(out, target)
```

の部分を

```python
out = self.model(x) # self.model()は上記のCNNLSTM

opt = self.optimizers()
opt.zero_grad()
loss = self.model.sample_elbo(
          inputs=out,
          labels=target,
          criterion=F.cross_entropy,
          sample_nbr=5,
          complexity_cost_weight=1 / out.shape[0],
      )
self.manual_backward(loss)
opt.step()
```

とする。これは

1. `Lightning`の自動最適化をオフにし、手動最適化する
2. `loss = self.model.sample_elbo(...)`で複数回推論を実行しELBOを推定する。

を行っている。注意点として`sample_elbo()`メソッドの

1. `criterion`: 最適化する損失関数、tensorを返す関数ならなんでも良い。
2. `sample_nbr`: トライする回数、増やしすぎると遅くなるが多い方が`weight complecity loss`の計算が正確になり推論に現実的なばらつきを出すことができる。少なすぎると安定して学習できない。
3. `complexity_cost_weight`: コンスタント値で、`criterion`に合わせて変える必要がある。通常の`CrossEntropy` or `MSE`ならバッチサイズの逆数にします。ここの理論は後述。

下記は`sample_nbr`と`complexity_cost_weight`を正しく設定した場合とそうで無い場合の比較、sample_nbrは多めにしよう。

<img src="/attachment/60b49733ca6fcd0048f33f91" width=400>

<img src="/attachment/60b49723ca6fcd0048f33f90" width=350>



これで`Lightning`のtrainerでの学習が可能になります。

### 4.Optional: 評価用関数を作成

ベイジアンDNNの特徴として予測値の**区間推定**ができるので、推論結果が何%の確率で正しいかを好きな信頼区間で求めることができる。  
通常のNNだとSoftmaxの出力の眺めることになるが[extrapolation問題](https://stats.stackexchange.com/questions/309642/why-is-softmax-output-not-a-good-uncertainty-measure-for-deep-learning-models)がありよろしく無い。  
ベイジアンモデルは重みが確率変数として毎回生成されるので、同じクエリに対して複数回推論し、結果のばらつき（分散、平均）から区間推定を行える。回帰問題の場合、

```python
def evaluate_regression(model,
                        X,
                        y,
                        samples = 100,
                        std_multiplier = 2):
    preds = [model(X) for i in range(samples)] # 複数回同じクエリを推論する
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()
```

と簡単に求められる。上記コードは公式コードそのまま。

![1*pqUSAVHQLislxZNW7M6WxA.png](/attachment/60b4962dca6fcd0048f33f09)

**問題はクラス分類における区間推定**で、Blitzはベイズ回帰を想定したライブラリなのでここら辺は自分で計算する。推定は仮説検定として行う。

```python
n = 100
target = [0] * n # Xに対応する正解ラベルをn個よういする
z = 1.64

logits = [model(X) for i in range(n)] # 同クエリを分類しまくる
logits = torch.stack(logits)
preds = torch.argmax(logits, dim=1)
acc = torch.sum(target == preds) * 1.0 / n

interval = z * sqrt( (acc * (1 - acc)) / n)
```
ここで`z`はガウス分布の信頼区間として設定する。良く使うのは

* 1.64 (90%)
* 1.96 (95%)
* 2.33 (98%)
* 2.58 (99%)

これで信頼区間が計算できる。例えば`acc = 0.8`つまりクエリ`X`に対して100回中80回正解できた場合の信頼係数95%のときの信頼区間は

```python
interval = 1.96 * sqrt( (0.2 * (1 - 0.2)) / 100)
# interval = 0.0784
```
としてこの推論結果を棄却する判断ができる。

逆に信頼係数99%だと`acc = 0.9`でも`interval = 0.0774`と棄却するべきである。信頼係数99%だと`acc = 0.999`くらいでやっと p値が0.0081くらいになって推論結果に一定の信頼を置くことができる様になる。

これが通常の分類モデルだとaccは常に1か0（Xへの推論結果は常に同じ）なのでp値は常に0になってしまうので計算できない。ベイズ最高。

