# PyTorchVideo 使い方

色々便利そうだったのでメモ

## 訓練済みモデルを使う

### 1. モデルのロード（torch hub経由）

master branchに専用の`hubconf.py`があるのでそれへのパスを記述して,stringでモデルを指定してロードする。

```python
model_name = "slow_r50"
path = 'path/to/directory/of/hubconf.py'
model = torch.hub.load(path, source="local",
                        model=model_name, pretrained=True)
```
 ちなみにモデルは`'pytorchvideo.models.net.Net'`という型になっているのでライトニングに組み込むときはアトリビュートにする。

 ### 2. 入力動画を規定の形にできる様Transformを準備する

動画のモデルごとの規格を合わせる。`slow_50`は256四方かつRGBが標準化されている必要がある。また、１インプットが何フレームかもモデルによって違うのであらかじめ書いておく必要がある。

詳しくは [こちら](https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html)

 ```python
 from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 8
frames_per_second = 30
clip_duration = (num_frames * sampling_rate)/frames_per_second

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)
 ```


### 3. 動画をエンコードする

適当な動画を用意しておけばエンコードも任せられる。`.avi`でもいけました。
手順としては
1. エンコードする
2. 秒を指定して切り取る
3. トランスフォームを通す

```python
from pytorchvideo.data.encoded_video import EncodedVideo

sample_path = 'sample.avi'

# 1
video = EncodedVideo.from_path(sample_path)
# 2
video_cliped = video.get_clip(start_sec=0, end_sec=10)
# 3
video_data = transform(video_data)
```

### 4. モデルに入力する

変換した動画は辞書型になってて

`video`で動画（C, T, H, W）`audio`で音声,`video_name`で元のパスが取得できる。

モデルは`(batch_size, C, T, H, W)`shapeのtensorをとるので

```python
input = video_data['video']
prediction = model(input.unsqueeze(0))
```

## おまけ Lightning組み込む

やることは二点

1. modelを組み込む
2. DataModuleを書く(Transform + label付与)

```python
class VideoClassification(pytorch_lightning.LightningModule):
  def __init__(self, path):
      super().__init__()
      self.model = torch.hub.load(path, source="local",
                        model=model_name, pretrained=True)

  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=1e-1)

  def forward(self, x):
      return self.model(x)

  def training_step(self, batch, batch_idx):
      y = self.model(batch["video"])
      t = batch["label"]
      loss = F.cross_entropy(y, t)

      return loss
```

```python
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip
)

class KineticsDataModule(pytorch_lightning.LightningDataModule):

    def setup(self);
        self.train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "train.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            transform=train_transform
        )

    def train_dataloader(self):
        train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )
```
