# PyTorch LightningのCheckpointCallbackの便利機能

トレーニング中になんでもチェックポイントファイルに含めてセーブできる機能があるみたいなのでメモ。

## `on_save_checkpoint`

```python
def on_save_checkpoint(self, checkpoint):
    # 99% of use cases you don't need to implement this method
    checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object
```

コレを使うとparameter以外に推論時に使う情報をcheckpointとして保持できるので推論時に手動でロードしたり別のファイルで保存しておく手間が省ける。

## `def on_load_checkpoint`

これでロードできる

```python
def on_load_checkpoint(self, checkpoint):
    # 99% of the time you don't need to implement this method
    self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']
```

## Use case

訓練データの共分散行列とかをモデルの重みと一緒に保存できる。
