# PyTorch Dataset API で動画データを扱う方法

## 今Video Recognitionの分野がアツい？

YouTubeやTikTok等大規模な動画データを収集しやすくなった昨今、`3DCNN`や `ViT`を用いた動画内での姿勢推定やクラス分類が盛んに研究されています。SoTAモデルだと実装が大変ですが実は既存のCNN+LSTMなどでも結構いい精度がだせたりします。  
今回はそんなVideo Recognitionをお手軽に試す上で以外と面倒くさいデータセットの作り方を書いていきます。

なお、動画データは各フレームを画像データとしてそれらの集合として扱うのが一般的ですのでそこまでの変換は別途行っておいてください。`ffmpeg`や`torchvision.io`等で調べると幸せになれます。

## やり方

今回は、**数フレームの動画を学習し、それがなんの動作か予測する**というタスクを仮定します。例えば動画の3フレームを受け取り、それがなんの動作なのかを推論します。つまり任意の一連の4フレームと対応するラベルを出力できればおk。

```python
import torch

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, frames):
        self.idxs = [0,1,2,3,4]
        self.data = "a b c d e".split()
        self.labels = "a b c d e".split()
        self.frames = frames

    def __len__(self):
        return 5 - (self.frames - 1)

    def __getitem__(self, idx):
        res = [self.data[i] for i in range(idx, idx + self.frames)]
        return res, self.labels[idx + (self.frames - 1)]
```

上記の例では動画の各フレーム画像のパスをabcdeとして一連の三枚と、その3枚目に対応するラベルを出力します。

```python
train = VideoDataset(frames=3)

for x, t in train:
    print(x, t)

# out
['a', 'b', 'c'] c
['b', 'c', 'd'] d
['c', 'd', 'e'] e

```

`__getitem__`の最終行を帰れば出力するラベルの位置を変更することができます。  
一連のフレームに対してどこをラベルとするかは問題設定によって変わって来ますが、最初のフレームをラベルにしてしまうとある意味未来のデータを学習させていることになるので大抵の場合は最後のフレームを教師データにします。  
ここら辺は時系列データに関する記事で改めて触れます。

次回はこのデータセットを使ったクラス分類モデルを書きます。
