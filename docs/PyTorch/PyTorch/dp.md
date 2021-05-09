# PyTorchの分散計算処理を使う

## 実験環境

CUDNNインストール済みLinuxで動作確認、winはわからん

## 使用するAPI

torch.distributedの中の[Distributed RPC Framework](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)というライブラリを使用する。 <br>
torch.distributedのバックエンドは、実行時にgloo, mpi, ncclの３通りから選ぶことができる。<br>
バックエンドによって、関数ごとに扱えるテンソル(cpu, gpu)の種類が違ってくる. <br>
具体的にはCPU使用時にはGloo、GPU使用時にはncclを使うと良いそうだ。 <br>
しかし、ncclを使う場合にはcpu上のテンソルは扱えず、さらに一部メソッドが使えなくなることに注意。

どのメソッドが使えなくなるのか？については[公式doc](https://pytorch.org/docs/stable/distributed.html)を参照。

[Pytorchのドキュメント](https://pytorch.org/tutorials/beginner/dist_overview.html#data-parallel-training)では、
> Use multi-machine **DistributedDataParallel** and the **launching script**, if the application needs to scale across machine boundaries.

とあり、一見するとRPCでなくこちらが使えそうに見えるが <br>
DistributedDataParallelなどのデータ並列用APIはどちらかと言えば教師あり学習用のものであり、例えば<br>
>In reinforcement learning, it might be relatively expensive to acquire training data from environments while the model itself can be quite small. In this case, it might be useful to spawn multiple observers running in parallel and share a single agent. In this case, the agent takes care of the training locally, but the application would still need libraries to send and receive data between observers and the trainer.

といったSyncOnPolicyのシチュエーションに対応できないそうだ。
こういったものに対応できる、より一般的なAPIとしてRPCが推奨されている。<br>
(https://pytorch.org/tutorials/beginner/dist_overview.html#general-distributed-training)

## 注意

*RPC*という言葉は公式ドキュメントにおいて文脈次第で以下の事柄のいずれかを指す。混同しやすいので、注意が必要。

1. [Distributed RPC Framework](#anchor1)
    - torch.distriutedの中のライブラリ。
2. 1の中のRPCという機能
3. 遠隔呼び出し一般を指す言葉 


## Distributed RPC Framework(https://pytorch.org/docs/stable/rpc.html#basics)<div id='anchor1'></div>

torch.distributedの中の、強化学習でメインで用いられるであろうAPI。
上の
windowsではサポートされていない(次の次のバージョンから部分的にサポートされる)


[RPCを使ってParameter Serverを作るチュートリアル](https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html)

1. Remote Procedure Call(RPC)
      - リモートの関数を引数を指定して返り値を受け取るAPI
          - 呼び出し元のユーザが返り値を受け取るまでつぎの処理に写らない（rpc_sync）
          - 呼び出し元のユーザが必要に応じて、返り値を待つ(rpc_async)
          - 呼び出し元のユーザが必要に応じて返り値への参照を待つ。(remote)
2. Remote Reference(RRef)
      - 他のマシン上のオブジェクトへの参照
3. Distributed Autograd <br>
      -　複数マシンでgradを送り合う機能を提供 <br>
      -　複数マシン上のgradの流れを把握してgradを取ってくれるAPI
4. Distributed Optimizer
      - Distributed Autogradで計算したgradで各マシンのワーカを更新してくれるAPI

## Distributed Autograd

リモートのマシンから送られてきたデータの計算グラフを把握し、各マシンの重みに対して勾配を計算、逆伝播させられる。

- 使用例

```python
>>> import torch.distributed.autograd as dist_autograd
>>> with dist_autograd.context() as context_id:
>>>   t1 = torch.rand((3, 3), requires_grad=True)
>>>   t2 = torch.rand((3, 3), requires_grad=True)
>>>   loss = rpc.rpc_sync("worker1", torch.add, args=(t1, t2)).sum()
>>>   dist_autograd.backward(context_id, [loss])
```

## Distributed Optimizer

https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim
渡されたリモート/ローカル両方のパラメータ（への参照 = rrefs）に対する勾配を計算する。 <br>
例

```python
import torch
import torch.multiprocessing as mp
import torch.distributed.autograd as dist_autograd
from torch.distributed import rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer

def random_tensor():
    return torch.rand((3, 3), requires_grad=True)

def _run_process(rank, dst_rank, world_size):
    name = "worker{}".format(rank)
    dst_name = "worker{}".format(dst_rank)

    # Initialize RPC.
    rpc.init_rpc(
        name=name,
        rank=rank,
        world_size=world_size
    )

    # Use a distributed autograd context.
    with dist_autograd.context() as context_id:
        # Forward pass (create references on remote nodes).
        rref1 = rpc.remote(dst_name, random_tensor)
        rref2 = rpc.remote(dst_name, random_tensor)
        loss = rref1.to_here() + rref2.to_here()

        # Backward pass (run distributed autograd).
        dist_autograd.backward(context_id, [loss.sum()])

        # Build DistributedOptimizer.
        dist_optim = DistributedOptimizer(
        optim.SGD,
        [rref1, rref2],
        lr=0.05,
        )

        # Run the distributed optimizer step.
        dist_optim.step(context_id)

def run_process(rank, world_size):
    dst_rank = (rank + 1) % world_size
    _run_process(rank, dst_rank, world_size)
    rpc.shutdown()

if __name__ == '__main__':
  # Run world_size workers
  world_size = 2
  mp.spawn(run_process, args=(world_size,), nprocs=world_size)
```

## 使用例

### SyncOnPolicy

https://pytorch.org/tutorials/intermediate/rpc_tutorial.html#distributed-reinforcement-learning-using-rpc-and-rref

### AsyncOnPolicy(ParameterServer)

https://pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html
