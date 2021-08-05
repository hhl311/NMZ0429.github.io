# Pythonの並列処理のまとめ

## 目次

1. 環境
2. コアとプロセスとスレッド
3. Numpy の BLAS
4. 計算スピードテスト
5. 結論
6. 懸念点

## 1. 環境

* Python=3.6.10
* numpy=1.17.1

## 2. コアとプロセスとスレッド

### 2.1 CPU 論理コア数

**CPU 論理コア数**とは実際に命令を行う部品のことで、CPU 論理コア数＝同時に実行できる命令の数。

自分の Mac の CPU 論理コアが 4 個しかないですが、`multiprocessing.cpu_count()`でカウントしたら`8`でした。それはハイパースレッディング(Hyper-Threading)が使われているからです。

```python
import multiprocessing
multiprocessing.cpu_count()

>>> 8
```

> **ハイパースレッディング**とは、従来 CPU のコア一つに一つしか搭載していなかったコードを実行する装置を複数搭載してコードの処理能力を向上するものである。これにより、ハイパースレッディングを備えた CPU ではホスト OS から実際搭載しているコア数より多くのコアを搭載しているよう「論理的に」見えることとなり、実コア数より多くのスレッドやプロセスを OS が同時に実行できるようになる。

参考：[ハイパースレッディング・テクノロジー](https://ja.wikipedia.org/wiki/%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%82%B9%E3%83%AC%E3%83%83%E3%83%87%E3%82%A3%E3%83%B3%E3%82%B0%E3%83%BB%E3%83%86%E3%82%AF%E3%83%8E%E3%83%AD%E3%82%B8%E3%83%BC)

ただし、ハイパースレッディングを使って、効率が倍増になるではないです。外部の記事によりますと、およそ 1.15 ~ 1.30 ぐらいになリました。
（参考：[ハイパースレッドによる CPU 性能向上効果検証 （Linux 編）](https://hesonogoma.com/linux/hyper-threading_cpu_performance_test_on_linux_2007.html)）

### 2.2 プロセス

**プロセス**とは、実行中のプログラムのことです。1 つのプロセスには、1 つのメモリ空間（メモリ領域）が割り当てられます。メモリ空間はプロセスから OS に要求すれば（空きがあれば）増やしてくれます。

### 2.3 スレッド

**スレッド**とは、プロセス内で命令を逐次実行する部分であり、CPU 論理コア数を利用する単位のことです。前述の通り、SMT（同時マルチスレッディング）登場以前では 1 スレッドに 1 コアが基本でした。

## 3. Numpy の BLAS

> BLAS とは、Basic Linear Algebra Subprograms の略です。難しく言うと、基本的な線形演算のライブラリ、簡単にいえば、行列やベクトルの基本的な計算をやってくれる関数群です。
>
> **BLAS の種類**
>
> BLAS には様々な種類のものが開発されています。代表的なものを一部取り上げてみます。
>
> 1. OpenBLAS
>
> BLAS のオープンソース実装。pip でインストールした numpy では、OpenBLAS が内部で呼び出されて演算が行われる。
> さまざまな CPU に対応しており、intel の Sandy Bridge CPU に対して Intel MKL と同等の速度を出せる言われている。マルチスレッド機能でよく知られており、コアの数と高速化が良く比例するそう。
>
> 2. Intel MKL ( Intel Math Kernel Library )
>
> Intel 開発の BLAS。Intel 製 CPU のみサポートしており、他社製の CPU では使えない。Xeon や Core i シリーズ、Atom などの Intel 製 CPU に最適化されているため、これらが CPU に使われているパソコンでは intel MKL を演算ライブラリとして利用した方が計算が高速になる。Anaconda や Miniconda で「conda install」でインストールした numpy は intel MKL が利用される。
>
> 3. ATLAS ( Automatically Tuned Linear Algebra Software )
>
> フリーの BLAS の 1 つ。名前の通り、チューニングすることによってインストールするハードウェアに最適なパラメータを設定し、高速な演算を実現する。

参考：[Numpy に使われる BLAS によって計算速度が変わるらしい【Python】](https://insilico-notebook.com/python-blas-performance/)

> 以下は Anaconda 公式によるものです。conda install と pip install、それぞれでインストールした tensorflow で、複数のディープラーニングモデルのトレーニングにかかった時間を比較していますが、最大 8 倍も差が出ています。

![image.png](/attachment/5f069bd13a5ba50047bc79bb)

* OpenBLAS(`pip install numpy`)
* Intel MKL(`conda install numpy`)

※ 自分の numpy が使っている線形代数の数値演算ライブラリの確認方法：

```python
import numpy
numpy.__config__.show()
```

![image.png](/attachment/5f069c8f3a5ba50047bc79bf)

## 4. 計算スピードテスト

（結果の一部スクリーンショット）
![image.png](/attachment/5f154d943a5ba50047bc8bfc)

**上のグラフの説明:**

1. それぞれのテストは全部 3 回ずつ
2. 「thread 数」：1 プロセス中指定したスレッド数。
3. 「core 数」：使ったコアの数。ローカル Macbook は 8 コア（CPU 論理コア数は 4 個）だけなので、8 コアまでテストしていません。GPU サーバの方は 40 コアあるですが、他の方の使用状況を考慮して、最大 20 コアまでしかテストしていません。
4. 「time x cores」：コア数とかかる時間の関係を表すコラムです。理想的なのはコア数にかかわらず、「time x cores」が全部同じです。そうすると、コア数とスピードが線形的な関係が言えます。
5. 使った numpy の数学ライブラリは「OpenBLAS」です。

**上のグラフからわらる内容：**

1. mac の物理コア数「4」に達したら、スピードの上がり率が 20~30%しかありません。（ハイパースレッディングが働いたからです。）
2. GPU サーバの場合、コアが 40 個あるので、上限に達すまでスピードが上がり続けています。ただし、「time x cores」の結果から見ると、線形的な関係ではないです。それは Numpy の全ての関数が BLAS を使用しているわけではなく、一部の関数（`dot()`、`vdot()`、`innerproduct()`および`numpy.linalg`モジュール。）だけなので、マルチスレッドがより良いパフォーマンスを提供できるかどうかは、ハードウェアに大きく依存します。（参考：[python / numpy のマルチスレッド blas](https://www.it-swarm.dev/ja/python/python-numpy%E3%81%AE%E3%83%9E%E3%83%AB%E3%83%81%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89blas/971418056/)）
3. 1 プロセス中に指定するスレッド数が増えれば増えるほど、全体的かかる時間が上がります。（ここの記事も同じことをテストしました：[Optimal number of threads per core](https://stackoverflow.com/questions/1718465/optimal-number-of-threads-per-core/10670440##10670440)）

## 5. 結論

1 プロセス中のスレッド数が多ければ多いほどではない、GPU サーバ(idun)を使う時、スレッド数を指定しないまま実行すると、1 プロセス=24 スレッドがデフォルトになっています。

並列処理（multiprocessing）で実行スピードを重視したい場合、まず 1 プロセス=1 スレッドに指定してから実行した方がいいです。

## 6. 懸念点

1. 今回テストしかのは大林組の案件だけなので、他のタスクにおいても同じ結論が言えるかどうかわからないです。
2. 使っている CPU コア数だけじゃなく、コアごとの利用率「%CPU」にも関係ありそうです。(2 コアで全部 100%CPU 利用率と、4 コアで 60%CPU 利用率の場合、スピードが必ず上がるとは言えないです。)
3. `multiprocessing`以外並列処理を実現できる手法やパッケージもあるので、それらを使った場合同じ結論かどうかはまだわからないです。

## 補足

1. スレッド数の指定方法：

* 1.1 OpenBLAS の場合（テスト済）

```shell
export OMP_NUM_THREADS=1
```

* 1.2 Intel MKL の場合（まだ未テスト）

```shell
export MKL_NUM_THREADS=1
```
