# 画像処理100本ノック答え

## Q1.[チャンネル入れ替え](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10#q1-%E3%83%81%E3%83%A3%E3%83%B3%E3%83%8D%E3%83%AB%E5%85%A5%E3%82%8C%E6%9B%BF%E3%81%88)

画像を読み込み、BGRからRGBへと変換。
### 解答
```
import cv2
img = cv2.imread("./img/imori.jpeg")
rgb_img = img[:, :, [2,1,0]].copy()
```
numpy配列の操作で変換できる。
```
cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
```
で変換可能

## Q2.[画像のグレースケール化](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10#q2-%E3%82%B0%E3%83%AC%E3%83%BC%E3%82%B9%E3%82%B1%E3%83%BC%E3%83%AB%E5%8C%96)
Y= 0.2126 R + 0.7152 G + 0.0722 B、で表される。
### 解答
```
img = cv2.imread("./img/imori.jpeg")
gray_img = img[:,:,0] * 0.0722 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.2126
```
opencvの機能で
```
cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
```
で変換できる。
行列計算でも出来そう。

## Q3.[２値化](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10#q3-%E4%BA%8C%E5%80%A4%E5%8C%96)
グレースケール画像の２値化。閾値は128
### 解答
```
img = cv2.imread("./img/imori.jpeg")
gray_img = img[:,:,0] * 0.0722 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.2126
ret, gray_img = cv2.threshold(gray_img,128,255,cv2.THRESH_BINARY)
```
一個一個条件分岐させると計算量が多そう。

## Q4.[大津の二値化](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10#q4-%E5%A4%A7%E6%B4%A5%E3%81%AE%E4%BA%8C%E5%80%A4%E5%8C%96)
二値化における分離の閾値を自動決定する手法である。
クラス間分散が最大となれば良い。
```
ret, gray_img = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
```
でうまくいかなかった。
事前の
```
gray_img = img[:,:,0] * 0.0722 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.2126
```
が原因。gray_imgがfloatだとうまくいかない。
### 解答
```
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
```

## Q5.[HSV変換](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10#q5-hsv%E5%A4%89%E6%8F%9B)
HSV変換とは、Hue(色相)、Saturation(彩度)、Value(明度) で色を表現する手法である。
```
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_img[:,:,0] = (hsv_img[:,:,0] + 180) % 360
hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
```
若干色が違う気がする。
Hueの範囲が[0:179]らしい。
```
hsv[:,:,0] = (hsv[:,:,0] + 90) % 180
```
まだ違う。
opencvを使わずに実装した。

### 解答
```
import cv2
import numpy as np
img = cv2.imread("./img/imori.jpeg")
h,w,c = img.shape
hsv = img / 255
for i in range(w):
    for j in range(h):
        b,g,r = img[i,j,:] / 255
        max_val = max(b,g,r)
        min_val = min(b,g,r)
        val = max_val
        sat = max_val - min_val
        if max_val == min_val:
            hue = 0
        elif min_val == b:
            hue = 60 * (g-r) / sat + 60
        elif min_val == r:
            hue = 60 * (b-g) / sat + 180
        else:
            hue = 60 * (r-b) / sat + 300
        # print(hsv[i,j,:])
        hsv[i,j,:] = [hue,sat,val]
        # print(hsv[i,j,:])
cv2.imwrite("./img/hsv_moto.jpeg",hsv)
hsv[:,:,0] = (hsv[:,:,0] + 180) % 360
revers_img = img/255
for i in range(w):
    for j in range(h):
        hue,sat,val = hsv[i,j,:]
        c = sat
        h_dot = hue / 60
        x = c * (1 - abs(h_dot % 2 - 1))
        if (0 <= h_dot) & (h_dot < 1):
            add_h = [c,x,0]
        elif (1<= h_dot) & (h_dot < 2):
            add_h = [x,c,0]
        elif (2<= h_dot) & (h_dot < 3):
            add_h = [0,c,x]
        elif (3<= h_dot) & (h_dot < 4):
            add_h = [0,x,c]
        elif (4<= h_dot) & (h_dot < 5):
            add_h = [x,0,c]
        elif (5<= h_dot) & (h_dot < 6):
            add_h = [c,0,x]
        else:
            add_h = [0,0,0]
        revers_img[i,j,:] = np.multiply([1,1,1], (val - c)) + add_h

revers_img = revers_img * 255
revers_img = revers_img[:, :, [2,1,0]]
cv2.imwrite("./img/hsv.jpeg",revers_img)
```
height,width,channelの順序をもっと意識する必要がある。
もっとコードを短くできると良いと思う。ifの分岐が多すぎる。

## Q6.[減色処理](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10#q6-%E6%B8%9B%E8%89%B2%E5%87%A6%E7%90%86)
すなわちR,G,B in {32, 96, 160, 224}の各4値に減色せよ。
### 解答
```
img = cv2.imread("./img/imori.jpeg")
img = (img // 64 + 1) * 64 - 32
cv2.imwrite("./img/result_img.jpeg",img)
```
if文を使わずに実装してみた。

## Q7.[平均プーリング](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10#q7-%E5%B9%B3%E5%9D%87%E3%83%97%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0)

画像をグリッド分割(ある固定長の領域に分ける)し、かく領域内(セル)の平均値でその領域内の値を埋める。
imori.jpgは128x128なので、8x8にグリッド分割し、平均プーリングせよ。
### 解答
```
import cv2
import numpy as np
img = cv2.imread("./img/imori.jpeg")

def average_pooling(img,karnel):
    pool_img = img.copy()
    height,width,channel = img.shape
    for i in range(0,height,karnel[0]):
        for j in range(0,width,karnel[1]):
            ave = np.mean(img[i:i+karnel[0],j:j+karnel[1],:],axis = 0)
            ave = np.mean(ave,axis = 0)
            pool_img[i:i+karnel[0],j:j+karnel[1],:] = ave
    return pool_img


kar = (8,8)
img = average_pooling(img,kar)
cv2.imwrite("./img/pool_img.jpeg",img)
```

for文を2回使ってしまっている。これ以上減らす方法が思いつかなかった。
また行と列の平均を2回に分けて求めてるのでこれを一度にできる方法があればよかった。
```
def average_pooling(img,karnel)
```
imgは画像、karnelはグリッドの分割の範囲

## Q8.[Maxプーリング](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10#q8-max%E3%83%97%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0)
ここでは平均値でなく最大値でプーリングせよ。
### 解答
```
import cv2
import numpy as np
img = cv2.imread("./img/imori.jpeg")

def average_pooling(img,karnel):
    pool_img = img.copy()
    height,width,channel = img.shape
    for i in range(0,height,karnel[0]):
        for j in range(0,width,karnel[1]):
            ave = np.max(img[i:i+karnel[0],j:j+karnel[1],:],axis = 0)
            ave = np.max(ave,axis = 0)
            pool_img[i:i+karnel[0],j:j+karnel[1],:] = ave
    return pool_img


kar = (8,8)
img = average_pooling(img,kar)
cv2.imwrite("./img/poolmax_img.jpeg",img)
```
平均を求める部分を、最大値に変更した。

## Q9.[ガウシアンフィルタ](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10#q9-%E3%82%AC%E3%82%A6%E3%82%B7%E3%82%A2%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF)
ガウシアンフィルタ(3x3、標準偏差1.3)を実装し、imori_noise.jpgのノイズを除去せよ。

ガウシアンフィルタは注目画素の周辺画素を、ガウス分布による重み付けで平滑化し、次式で定義される。 このような重みはカーネルやフィルタと呼ばれる。
### 解答
```
import cv2
import numpy as np
img = cv2.imread("./img/imori_noise.jpeg")

def gausu_filter(img,karnel,sigma):
    height,width,channel = img.shape
    pad = karnel // 2
    pad_img = np.zeros((height + pad * 2,width + pad * 2, channel))
    pad_img[pad:pad+height,pad:pad+width] = img
    weight = gausu(sigma,karnel,pad)
    gausu_img = img.copy()
    for i in range(height):
        for j in range(width):
            gausu_img[i,j,0] = np.sum(pad_img[i:i+pad*2+1,j:j+pad*2+1,0]*weight)
            gausu_img[i,j,1] = np.sum(pad_img[i:i+pad*2+1,j:j+pad*2+1,1]*weight)
            gausu_img[i,j,2] = np.sum(pad_img[i:i+pad*2+1,j:j+pad*2+1,2]*weight)
            print(gausu_img)
    return gausu_img

def gausu(sigma,karnel,pading):
    filt = np.zeros((karnel,karnel))
    for x in range(pading * -1, pading + 1):
        for y in range(pading * -1, pading + 1):
            print(x,y)
            filt[x+pading,y+pading] = 1 / (2*np.pi*sigma*sigma) * np.exp((-1 * (x*x + y*y))/(2 * (sigma**2)))
    filt /= filt.sum()
    return filt



kar = 3
sig = 1.3
img = gausu_filter(img,kar,sig)
cv2.imwrite("./img/gausu_img.jpeg",img)
```

フィルターの作成にも関数を用いた。

## Q10.[メディアンフィルタ](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10#q10-%E3%83%A1%E3%83%87%E3%82%A3%E3%82%A2%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF)
メディアンフィルタ(3x3)を実装し、imori_noise.jpgのノイズを除去せよ。
これは注目画素の3x3の領域内の、メディアン値(中央値)を出力するフィルタである。 これもゼロパディングせよ。

[サイト](https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy/54966908)を参考にして、for文を少なく書いてみる。
### 解答
```
import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, ((padding,padding),(padding,padding),(0,0)), mode='constant')


    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1,
                    A.shape[2])
    kernel_size = (kernel_size, kernel_size)

    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1],
                                   stride*A.strides[2]
                                   ) + A.strides[0:2])
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == "med":
        return np.median(A_w,axis=(1,2)).reshape(output_shape)



img = cv2.imread("./img/imori_noise.jpeg")
height,width,channel = img.shape
karn = 3
padding = karn // 2
medhian = pool2d(img, kernel_size=karn, stride=1, padding=padding, pool_mode='med')

cv2.imwrite("./img/medhian_img.jpeg",medhian)
```
参考にした結果for文を使わないで実装することができた。
しかしas_stridedの引数stridesの役割がよく分かっていない。調べたところメモリの移動距離のようだった。
```
strides = (stride*A.strides[0],stride*A.strides[1],stride*A.strides[2]) + A.strides[0:2])
```
のshapeが
```
(390, 3, 1, 390, 3)
```
となっている。前半の(390,3,1)は(height,width,channel)を表していて、動かす(height,width)をもう一度追加しているのか？

## Q11.[平滑化フィルタ](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_11_20#q11-%E5%B9%B3%E6%BB%91%E5%8C%96%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF)
平滑化フィルタ(3x3)を実装せよ。

平滑化フィルタはフィルタ内の画素の平均値を出力するフィルタである。
### 解答
```
import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, ((padding,padding),(padding,padding),(0,0)), mode='constant')


    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1,
                    A.shape[2])
    kernel_size = (kernel_size, kernel_size)
    print((stride*A.strides[0],stride*A.strides[1],stride*A.strides[2]) + A.strides[0:2])

    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1],
                                   stride*A.strides[2]
                                   ) + A.strides[0:2])
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == "med":
        return np.median(A_w,axis=(1,2)).reshape(output_shape)



img = cv2.imread("./img/imori.jpeg")
height,width,channel = img.shape
karn = 3
padding = karn // 2
mean = pool2d(img, kernel_size=karn, stride=1, padding=padding, pool_mode='avg')

cv2.imwrite("./img/mean_img.jpeg",mean)
```
メディアンフィルタの最後の部分を平均に変えただけである。

## Q12.[モーションフィルタ](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_11_20#q12-%E3%83%A2%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF)
モーションフィルタ(3x3)を実装せよ。

モーションフィルタとは対角方向の平均値を取るフィルタであり、次式で定義される.
```
[[1/3,0,0]
 [0,1/3,0]
 [0,0,1/3]]
```
### 解答
```
import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, ((padding,padding),(padding,padding),(0,0)), mode='constant')


    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1,
                    A.shape[2])
    kernel_size = (kernel_size, kernel_size)
    print((stride*A.strides[0],stride*A.strides[1],stride*A.strides[2]) + A.strides[0:2])

    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1],
                                   stride*A.strides[2]
                                   ) + A.strides[0:2])
    A_w = A_w.reshape(-1, *kernel_size)
    weight = [[1/3,0,0],[0,1/3,0],[0,0,1/3]]
    weight = np.array(weight).reshape(-1,3,3)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == "med":
        return np.median(A_w,axis=(1,2)).reshape(output_shape)
    elif pool_mode == "motion":
        return np.sum(A_w*weight,axis = (1,2)).reshape(output_shape)



img = cv2.imread("./img/imori.jpeg")
height,width,channel = img.shape
karn = 3
padding = karn // 2
motion = pool2d(img, kernel_size=karn, stride=1, padding=padding, pool_mode='motion')
print(motion.shape)

cv2.imwrite("./img/motion_img.jpeg",motion)
```
重みを作成してその値をそれぞれにかけた。

## Q13.[MAX-MINフィルタ](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_11_20#q13-max-min%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF)
MAX-MINフィルタとはフィルタ内の画素の最大値と最小値の差を出力するフィルタであり、エッジ検出のフィルタの一つである。
### 解答
```
import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, ((padding,padding),(padding,padding)), mode='constant')


    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    # print((stride*A.strides[0],stride*A.strides[1],stride*A.strides[2]) + A.strides[0:2])

    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1],
                                   ) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    weight = [[1/3,0,0],[0,1/3,0],[0,0,1/3]]
    weight = np.array(weight).reshape(-1,3,3)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'min':
        return A_w.min(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == "med":
        return np.median(A_w,axis=(1,2)).reshape(output_shape)
    elif pool_mode == "motion":
        return np.sum(A_w*weight,axis = (1,2)).reshape(output_shape)
    elif pool_mode == "max_min":
        max_pool = A_w.max(axis=(1,2)).reshape(output_shape)
        min_pool = A_w.min(axis=(1,2)).reshape(output_shape)
        return max_pool - min_pool



img = cv2.imread("./img/imori.jpeg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
height,width= img.shape
karn = 3
padding = karn // 2
max_min = pool2d(img, kernel_size=karn, stride=1, padding=padding, pool_mode='max_min')
print(max_min.shape)

cv2.imwrite("./img/min_max_img.jpeg",max_min)
```
事前にグレースケールにしたが、BGR画像で作成した後にグレースケールにしても同じなのか？


## Q14.[微分フィルタ](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_11_20#q14-%E5%BE%AE%E5%88%86%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF)
微分フィルタ(3x3)を実装せよ。

微分フィルタは輝度の急激な変化が起こっている部分のエッジを取り出すフィルタであり、隣り合う画素同士の差を取る。

縦のフィルタ
```
[[0,0,0]
 [-1,1,0]
 [0,0,0]]
```
横のフィルタ
```
[[0,-1,0]
 [0,1,0]
 [0,0,0]]
```

 ### 解答
```
import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, ((padding,padding),(padding,padding),(0,0)), mode='constant')


    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1,
                    A.shape[2])
    kernel_size = (kernel_size, kernel_size)
    print((stride*A.strides[0],stride*A.strides[1],stride*A.strides[2]) + A.strides[0:2])

    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1],
                                   stride*A.strides[2]
                                   ) + A.strides[0:2])
    A_w = A_w.reshape(-1, *kernel_size)
    weight = [[1/3,0,0],[0,1/3,0],[0,0,1/3]]
    weight = np.array(weight).reshape(-1,3,3)
    weight_w = [[0,0,0],[-1,1,0],[0,0,0]]
    weight_h = [[0,-1,0],[0,1,0],[0,0,0]]
    weight_w = np.array(weight_w).reshape(-1,3,3)
    weight_h = np.array(weight_h).reshape(-1,3,3)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'min':
        return A_w.min(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == "med":
        return np.median(A_w,axis=(1,2)).reshape(output_shape)
    elif pool_mode == "motion":
        return np.sum(A_w*weight,axis = (1,2)).reshape(output_shape)
    elif pool_mode == "max_min":
        max_pool = A_w.max(axis=(1,2)).reshape(output_shape)
        min_pool = A_w.min(axis=(1,2)).reshape(output_shape)
        return max_pool - min_pool
    elif pool_mode == "diff_w":
        return np.sum(A_w*weight_w,axis = (1,2)).reshape(output_shape)
    elif pool_mode == "diff_h":
        return np.sum(A_w*weight_h,axis = (1,2)).reshape(output_shape)



img = cv2.imread("./img/imori.jpeg")
height,width,channel = img.shape
karn = 3
padding = karn // 2
diff_w_img = pool2d(img, kernel_size=karn, stride=1, padding=padding, pool_mode='diff_w')
diff_h_img = pool2d(img, kernel_size=karn, stride=1, padding=padding, pool_mode='diff_h')

cv2.imwrite("./img/diff_h_img.jpeg",diff_h_img)
cv2.imwrite("./img/diff_w_img.jpeg",diff_w_img)
```

フィルタごとに分離しておけば処理が簡単にできることが分かった。

## Q20.[ヒストグラム表示](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_11_20#q20-%E3%83%92%E3%82%B9%E3%83%88%E3%82%B0%E3%83%A9%E3%83%A0%E8%A1%A8%E7%A4%BA)
matplotlibを用いてimori_dark.jpgのヒストグラムを表示せよ。

### 解答
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./img/imori_dark.jpeg")
gaso = np.array(img).flatten()
plt.hist(gaso,bins=255,range=(0,255),rwidth=0.8)
plt.show()
```
ヒストグラムにするときは１次元にしないと処理できない。

## Q21.[ヒストグラム正規化](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_21_30#q21-%E3%83%92%E3%82%B9%E3%83%88%E3%82%B0%E3%83%A9%E3%83%A0%E6%AD%A3%E8%A6%8F%E5%8C%96)
ヒストグラム正規化を実装せよ。
[c,d]の画素値を持つ画像を[a,b]のレンジに変換する。

### 解答
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

def gray_scale_trans(img,a=0,b=255):
    out = img.copy()
    c = img.min()
    d = img.max()

    out = (b-a)/(d-c)*(out-c)+a

    np.where(out < a, a, out)
    np.where(b < out, b, out)

    return out

img = cv2.imread("./img/imori_dark.jpeg")

trans_img = gray_scale_trans(img)

gaso = np.array(trans_img).flatten()
plt.hist(gaso,bins=255,range=(0,255),rwidth=0.8)
cv2.imwrite("./img/trans_img.jpeg",trans_img)
plt.show()
```
np.whereを用いて実装することができた。
ヒストグラム正規化を関数として表した。

## Q.22[ヒストグラム操作](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_21_30#q22-%E3%83%92%E3%82%B9%E3%83%88%E3%82%B0%E3%83%A9%E3%83%A0%E6%93%8D%E4%BD%9C)

ヒストグラムの平均値をm0=128、標準偏差をs0=52になるように操作せよ。
ヒストグラムを平坦に変更する操作である.

### 解答
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

def hist_heitan(img,m0=128,s0=52):
    out = img.copy()
    s = np.std(img)
    m = np.average(img)
    out = s0 / s * (out - m) + m0
    return out

img = cv2.imread("./img/imori_dark.jpeg")

trans_img = hist_heitan(img)
gaso = np.array(trans_img).flatten()
plt.hist(gaso,bins=255,range=(0,255),rwidth=0.8)
cv2.imwrite("./img/trans_img_1.jpeg",trans_img)
plt.show()
```
ヒストグラムを平坦にする関数を実装した。


## Q.23[ヒストグラム平坦化](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_21_30#q23-%E3%83%92%E3%82%B9%E3%83%88%E3%82%B0%E3%83%A9%E3%83%A0%E5%B9%B3%E5%9D%A6%E5%8C%96)
ヒストグラム平坦化を実装せよ。
ヒストグラム平坦化とはヒストグラムを平坦に変更する操作であり、上記の平均値や標準偏差などを必要とせず、ヒストグラム値を均衡にする操作である。
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

def hist_heitan_function(img, z_max=255):
    out = img.copy()
    height, width, channel = img.shape
    S = height * width * channel

    sum_h = 0

    for i in range(1, 255):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime

    return out


img = cv2.imread("./img/imori.jpeg")

trans_img = hist_heitan_function(img)
gaso = np.array(trans_img).flatten()
plt.hist(gaso, bins=255, range=(0, 255), rwidth=0.8)
cv2.imwrite("./img/trans_img_3.jpeg", trans_img)
plt.show()
```
np.whereを用いてfor文を無くそうとしたが上手く行かなかった。
