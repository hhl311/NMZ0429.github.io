# PyTorch Lightning Boltsの使い方

便利な時代になりましたね。

## Boltsとは？

PyTorch Lightning用の便利コードが詰まった公式ライブラリ。

* 訓練済みSOTAモデル
* よく使うモデルコンポネート
* Callback用のフォワード、バックフック
* ロス関数
* 有名なデータセット

これらがPyTorch Lightningで直ぐに使えるようになっていてとっても便利。
以下、使い方の例

## 1.訓練済みモデルをそのまま使う

最新のクラスタリングを使ってみたい時などは
```python
from pl_bolts.models.self_supervised import SwAV

weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar' # weight file of ImageNet
swav = SwAV.load_from_checkpoint(weight_path, strict=True)

swav.freeze()
```

以降`swav`は通常の`nn.Module`として扱うことができる。

## 2.コンポネート単位で使う

訓練済みモデルをバックボーンにしたりエンコーダー部分だけ採用したりもできる。
今回は`ResNet152`の入力チャンネル数を3から4にしてみた。

```python
from pl_bolts.models.self_supervised.resnets import resnet152

model = resnet152(pretrained=True)

temp_weight = model.conv1.weight.data.clone() # 既存の重みを退避
model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) # input_channelだけ増やす
model.conv1.weight.data[:, :3] = temp_weight # 3 channelまでは既存の重み
model.conv1.weight.data[:, 3] = model.conv1.weight.data[:, 0] # Rの重みを4つめの重みとして採用
```

逆に`model.conv1`だけを別のモデルに使ったりもできる。

## 3.コールバックを使う

PyTorch Lightning の`Callback`　API用の便利な処理が色々揃ってる。
**必要なCallbackオブジェクトを宣言してそれらのリストをトレイナーに渡すだけで使える**。
ここでは二つ紹介。

1. エポック毎にロスを表示する

```python
from pl_bolts.callbacks import PrintTableMetricsCallback

print_callback = PrintTableMetricsCallback()
trainer pl.Trainer(callback=[print_callback])
trainer.fit(model)
```

2. GANのforward時に生成した画像をTensorBoardに表示する

```python
model.img_dim = (1, 28, 28)

# model forward must work for sampling
z = torch.rand(batch_size, latent_dim)
img_samples = GAN(z)

from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
trainer.fit(GAN)
```

## 4.ロス関数を使う

タスク毎にいくつかの関数が実装されているがまだ数が少ない。
インポートするとPyTorchのロス関数になるのでモデルクラスを作成する時に`lossfun`メソッドに渡してあげればいい。
下記は物体検知用のGeneralizedIoU。


```python
>>> import torch
>>> from pl_bolts.losses.object_detection import giou_loss
>>> preds = torch.tensor([[100, 100, 200, 200]])
>>> target = torch.tensor([[150, 150, 250, 250]])
>>> giou_loss(preds, target)
tensor([[1.0794]])
```

```python
def lossfun(self, y, t): # method of a network
        return giou_loss(y, t)
```



## 5.データモジュールを使う

`LightningDataModule`化されたデータセットが揃っている。
指定したディレクトリにダウンロードする機能もあるのでモデルのテストを直ぐに始められる。マルチGPU対応。

* DAを自分で変更したりできる

```python
from pl_bolts.datamodules import CIFAR10DataModule

dm = CIFAR10DataModule('PATH_to_download/to_load')
dm.train_transforms = ... # ここにComposeオブジェクトを渡せばおk
dm.test_transforms = ...
dm.val_transforms  = ...
```

* Numpyでx,yを渡すだけで`LightningDataModule`化してくれて凄いと思いました（語彙力）

```python
>>> from sklearn.datasets import load_boston
>>> from pl_bolts.datamodules import SklearnDataset
...
>>> X, y = load_boston(return_X_y=True)
>>> dataset = SklearnDataset(X, y) # シェイプがあってればなんでも渡せる
```
