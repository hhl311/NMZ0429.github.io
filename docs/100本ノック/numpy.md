# Numpy100本ノック答え
[numpy-100](https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md)の問題を解いたもの

## Q1. Import the numpy package under the name np
numpyをインストールする。
### 解答
```
import numpy as np
```
## Q2.Print the numpy version and the configuration
numpyのバージョンを確認する。
### 解答
```
import numpy as np

print(np.__version__)
```
1.20.2だった。

## Q3.Create a null vector of size 10
サイズ１０の０の配列作成
### 解答
```
import numpy as np

a = np.zeros(10)
print(a)
```

## Q4.How to find the memory size of any array
配列のメモリサイズを見る
### 解答
```
a = np.zeros((10, 10))
print(a.nbytes)　#800
```
一つの要素のバイト数は
```
a.itemsize
```
で得られる。

## Q5.How to get the documentation of the numpy add function from the command line?
numpyのadd関数のドキュメントをコマンドラインから見る方法
### 解答
```
python -c "import numpy; numpy.info(numpy.add)"
```

## Q6.Create a null vector of size 10 but the fifth value which is 1
サイズ10の0ベクトルで5番目の値だけ1のものを作成する。
### 解答
```
a = np.zeros((10))
a[4] = 1
print(a)
```

## Q7.Create a vector with values ranging from 10 to 49
10~49の範囲の値をもつ配列の作成
### 解答
```
b = np.arange(10, 50)
print(b)
```

## Q8. Reverse a vector (first element becomes last)
ベクトルを反転
### 解答
```
b = np.arange(10, 50)
print(b[::-1])
```

## Q9.Create a 3x3 matrix with values ranging from 0 to 8
3×3の行列に0~9を入れる
### 解答
```
c = np.arange(0, 9)
print(c.reshape(3, 3))
```
reshapeは重要な気がする

## Q10. Find indices of non-zero elements from [1,2,0,0,4,0]
ベクトルから０ではないベクトルを見つけろ
### 解答
```
d = np.array([1, 2, 0, 0, 4, 0])
print(np.where(d != 0))
```

## Q11. Create a 3x3 identity matrix
単位行列を作れ
### 解答
```
e = np.eye(3)
print(e)
```

## Q12. Create a 3x3x3 array with random values
ランダムな行列を作れ
### 解答
```
f = np.random.random((3, 3, 3))
print(f)
```

## Q13. Create a 10x10 array with random values and find the minimum and maximum values
ランダムな行列から最大、最小の値をとれ
### 解答
```
f = np.random.random((10, 10))
print(np.max(f), np.min(f))
```

## Q14.Create a random vector of size 30 and find the mean value
ランダムなベクトルから平均値をとれ
### 解答
```
g = np.random.random(30)
print(np.mean(g))
```

## Q15. Create a 2d array with 1 on the border and 0 inside
境界が0、中が１の２次元の行列を作成せよ
### 解答
```
size = 5
h = np.zeros((size, size))
h[1:size-1, 1:size-1] = 1
print(h)
```

## Q16.How to add a border (filled with 0's) around an existing array?
周りに0のパディングを追加せよ
### 解答
```
i = np.ones((5, 5))
print(np.pad(i, 1))
```
padも慣れていく必要がある

## Q17.What is the result of the following expression?
### 解答
```
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)
```

実行結果
```
nan
False
False
nan
True
False
```
np.nanの特徴をしっかり把握する必要がある

## Q18.Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
対角の下に1,2,3,4の値を入れる
### 解答
```
j = np.diag(np.arange(4) + 1, k=-1)
print(j)
```

## Q19.Create a 8x8 matrix and fill it with a checkerboard pattern
行列をチェッカーボードパターンにしましょう
### 解答
```
j = np.zeros((8, 8))
j[1::2, ::2] = 1
j[::2, 1::2] = 1
print(j)
```
配列操作の::の意味を知らなかった。

## Q20.Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element
100番目の要素を取り出せ
### 解答
```
print(np.unravel_index(99, (6, 7, 8)))
```

## Q21. Create a checkerboard 8x8 matrix using the tile function
tile関数を使用してチェッカーボードパターンを作成せよ
### 解答
```
j = np.tile(((0, 1), (1, 0)), (4, 4))
print(j)
```

## Q22. Normalize a 5x5 random matrix
行列を正規化せよ
### 解答
```
k = np.random.random((5, 5))
k = (k - np.mean(k))/np.std(k)
print(k)
```

## Q23. Create a custom dtype that describes a color as four unsigned bytes (RGBA)
カスタムdtypeを作成せよ
### 解答
```
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
```
自分でdtypeを作成している。この機能を使う機会はあるのだろうか？

## Q24.Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
行列積を行う
### 解答
```
p = np.ones((5, 3))
m = np.ones((3, 2))
n = np.dot(p, m)
print(n)
```

## Q25 Given a 1D array, negate all elements which are between 3 and 8, in place.
3-8までの要素を負の値
### 解答
```
q = np.arange(10)
q[(3 < q) & (q <= 8)] *= -1
print(q)
```
画像処理100本ノックでもよく見た。

## Q26.What is the output of the following script?
以下の実行結果を確認せよ
```
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
### 結果
```
9
10
```
上のsumは-1を計算に入れていて、下のsumは軸を-1に指定しているとなっている

## Q27.Consider an integer vector Z, which of these expressions are legal?
以下の式は有効か?(Zは整数)
```
Z = np.arange(5)
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```
### 結果
```
[  1   1   4  27 256]
[0 1 2 4 8]
[False False False False False]
[0.+0.j 0.+1.j 0.+2.j 0.+3.j 0.+4.j]
[0. 1. 2. 3. 4.]
error
```
・各要素値ごとの累乗計算の結果を示している
```
[0^0,1^1,2^2,3^3,4^4]
```
・ビット演算をしている

・各要素の大小関係

・虚数にしている

・2回１で割っている

・2つの計算はできな
