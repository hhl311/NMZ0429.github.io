# 【PyTorch Lightning】LightningDataModuleについて

ちゃんとした日本語解説が無かったので今後の参考になればと思いメモしておきます。

## 概要

PyTorch Lightningでモデルを動かす時のDataLoader(場合によってはDatasetも)となるクラス、
PyTorchの該当モジュールと互換性がある。これを含めて、

1. `LightningModule`がモデル
2. `LightningDataModule`がデータ
3. その他必要なカスタマイズ（`Callbacks API`,`LR_FINDER`等）

を書けばおk

## LightningDataModuleの書き方

`init`以外に3つのメソッドを実装する必要がある。

1. `prepare_data` （無くても動く）
2. `setup`
3. `~_dataloader`

### 0. `__init__`

必要なparametersを作る。Datasetオブジェクトを作るわけではないので注意してください。  
以下の例ではテストデータと訓練データがディレクトリで別れてると仮定します。

```python
import pytorch-lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

class DataModule(pl.LightningDataModule):
    def __init__(self, train_dir='./train', test_dir='./test', batch_size=64):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_augmentation = transforms.Compose([
            transforms.ToTensor(),
            # ... some data augmentations...
            transforms.Normalize((0.1307,), (0.3081,))
        ])
```

#### 余談：albumentationsというDAライブラリが便利です、torchvisionのtransformと互換性がありますのでここでも使えます。

### 1. `prepare_data`

最初に呼ばれるメソッドでデータのダウンロードなど**GPU数にかかわらず一回行いたい処理を書く**。  
ここに書くことでマルチGPUでもダウンロード処理をよしなにやってくれるみたいです。  
例えば、MNISTをダウンロードする場合

```python
def prepare_data(self):
    # download
    MNIST(self.data_dir, train=True, download=True)
    MNIST(self.data_dir, train=False, download=True)
```

### 2. `setup`

２番目に呼ばれるメソッドです。  
`Trainer.fit()`と`Trainer.test()`が呼ばれた時に異なるDatasetを流す処理をここに書きます。  
DAの有無等もここでスイッチするのがいいでしょう。  
何かしらのDatasetクラスを別に作っておくと読みやすいと思います。

* 注意：Trainerからstage引数にモードが文字列として渡されてくるようですが、Noneになった時の処理を
書いておきましょう。setupを手動で呼ぶことがあります。
* 注意２：マルチGPUの場合各GPUから一回づつ呼ばれます。

```python
def setup(self, stage=None):
    if stage == 'fit' or stage is None:
        self.train_set = MyDataset(
            self.train_dir,
            transform=self.data_augmentation
        )
        size = len(self.train_set)
        t, v = (int(size * 0.9), int(size * 0.1)) # if using holdout method
        t += (t + v != size)
        self.train_set, self.valid_set = random_split(self.train_set, [t, v])

    if stage == 'test' or stage is None:
        self.test_set = MyDataset(
            self.test_dir,
            transform=self.transform
        )
```

### 3. `~_dataloader`

最後に呼ばれるメソッドで、Dataloaderオブジェクトを返します。  
訓練、検証、テストように三つ書きます。

```python
def train_dataloader(self):
    return DataLoader(
        self.train_set,
        batch_size=self.batch_size,
    )

def val_dataloader(self):
    return DataLoader(
        self.test_set,
        batch_size=self.batch_size,
    )

def test_dataloader(self):
    return DataLoader(
        self.valid_set,
        batch_size=self.batch_size,
    )
```

必要なメソッドは以上になります。

## EXTRA: LightningDataModuleを使う

通常の場合,

```python
dm = DataModule()
model = Model()
trainer.fit(model, dm)

trainer.test(datamodule=dm)
```

で上記のメソッドを勝手に呼んで訓練まで行ってくれます。

が、場合によってはモデルを生成する時にデータセットの情報（クラス数や画像サイズ、ちゃんねる数）が
必要になるのでその時は`setup`内に必要な情報を収集する処理を記載してから

```python
dm = DataModule()
dm.prepare_data()
dm.setup('fit')　# アトリビュートに情報を格納しておけるようにしておくこと

model = Model(num_classes=dm.num_classes, width=dm.=img_size)
trainer.fit(model, dm)
```
