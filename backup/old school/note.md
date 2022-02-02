<div style="text-align: center;">
# CSC413 MATOME
</div>

---

## 1. Non-deep Learning AI models

1. Linear reg
2. Logistic reg
3. Feature mapping


    * Lack of complexity.
    * Can not fit non-convex set. (Non-linearity)
    * Dataset needs to be manifolding.

### 1-1. Optimization in High Dimensional space

1. **Convexity**: Linear regression and logistic regressions are **convex** i.e. has exactly one minima
2. **Saddle Points**: Minima with respect to some direction but not global minima
3. **Plateaux**: Flat surface. Usually occurs due to **saturated unit** a.k.a **dead unit**
4. **Ravines**: Mixture of high and small gradients. Must be dealt with **Momentum**

```python
def gradient_decent(x, t):
    b_ = 0
    w_ = 0
    y = forward(x)
    for i in range(iterations):
        b_ += -2/N * (t - y) * a
        w_ += -2/N * (t - y) * x * a

    return
```


## 2. MLP

* Basic neural network.
* Composition of linear transformations
* Can be used as both embedder and classifier
* The more layers, the higher risk of overfitting
* Not convext hence has multiple minima
* Unlimited complexity under the universal approximation theorems which asserts
that MLP with infinite number of neurons can regress any linear function.

1. Dropout: Enabled only on training

### 2-1. Language Modeling

1. **N-gram**: Constract a table of all possible inputs and probabilities of all possible outputs
2. **GloVe**: Embedding space that relies on **Distributional Hypothesis** and built **co-occurrence matrix** that indicates wether two words appears in the similar context or not.

### 2-2. Training Neural Network

1. **Genelarization**: Increase test accuracy
2. **Data Augmentation**:

## 3. CNN

* Convert 2d image into 1d vector along channel dimension
* High robustness with various size of images
* Classification is done by MLP Affine layers
* Still overfits without skip-connection and residual blocks

1. BatchNorm : Channel wise normalization increasing robustness
2. Loss function : Cross Entropy in classification, MSE in regression


### AlexNet, VGG, ResNet, U-net

Major models used for fine-tuning and transfer learning.
Transfer learning usually freezes Conv layers and only train FC layers while fine-tuning trains all parameters from scratch.

## 4. RNN

<img src=https://tips-memo.com/wp-content/uploads/2020/05/RNN_1.gif width=700>

Some major architectures are

1. LSTM MLP
2. GRU MLP
3. VAE
4. RCNN
5. ConvLSTM

* LSTM layer has 4 logic gates while GRU has 3

**LSTM GATES**

<img src="https://camo.qiitausercontent.com/9ae5e3cc75fd1c30041a91493e5e9e0a57bf4db3/
687474703a2f2f636f6c61682e6769746875622e696f2f706f7374732f323031352d30382d556e64
65727374616e64696e672d4c53544d732f696d672f4c53544d322d6e6f746174696f6e2e706e67"
width=500>

### 4-1 Encoder-Decoder Model (Seq2Seq)

<img src=https://tips-memo.com/wp-content/uploads/2020/05/seq2seq_4.gif width=600>

* **Context Vector**: Dimensional vector representing context of the sentece. The dimension is equivalent to the number of hidden state.

### 4-2. VAE

**VAE is the first generative image model**

<img src="https://i0.wp.com/gagbot.net/gagbot.net/wp-content/uploads/2016/08/深層学習入門（HP用）6.jpg?resize=768%2C576" width=300>

1. Encoder learns a mapping from input image to gaussian parameters **mean and variance**
2. Generator takes a sampling as an input from normal distribution with given gaussian parameters by the encoder.
3. If regression, generator outputs new gaussian parameters otherwise output an image depending on the task

#### Loss function

1. The training of VAE is equivalent to MAP with respect to decoder's log likelihood

$$
\begin{eqnarray}
 \log p_\theta(x) &=& \log \int p_\theta(x, z) dz \\
 &=& \log \int q_\varphi(z|x)\frac{p_\theta(x, z)}{q_\varphi(z|x)} dz \\
 &\geq& \int q_\varphi(z|x) \log \frac{p_\theta(x, z)}{q_\varphi(z|x)} dz \\
 &=& L(x; \varphi, \theta)
 \end{eqnarray}
$$

2. The difference between LHS and RHS is **KL-divergence**

$$
\begin{eqnarray}
 \log p_\theta(x) – L(x; \varphi, \theta) &=&  \log p_\theta(x) – \int q_\varphi(z|x) \log \frac{p_\theta(x, z)}{q_\varphi(z|x)} dz \\
 &=& \log p_{\theta}(x) \int q_{\varphi} (z|x) dz – \int q_{\varphi} (z|x) \log \frac{p_{\theta} (z|x)p(x)}{q_{\varphi}(z|x)} dz \\
 &=& \int q_\varphi (z|x) \{ \log p_{\theta}(x) – \log p_\theta(z|x) – \log p_{\theta}(x) + \log q_\varphi (z|x) \} dz\\
 &=& \int q_\varphi (z|x) \{ \log q_\varphi (z|x) – \log p_\theta(z|x) \} dz\\
 &=& KL[q_\varphi (z|x) \| p_\theta (z|x)]
 \end{eqnarray}
$$

3. This is equivalent to maximizing the evidence lower bound.

$$
\begin{eqnarray}
 E_{q_\varphi (z|x)}[\log p_\theta (x|z)]
 &=& E_{q_\varphi (z|x)}[\log \prod_l^{L} f(z_l)^x (1 – f(z_l))^{(1 – x)}] \\
 &=& \frac{1}{L} \sum_{l=1}^L \{ x \log f(z_l) + (1 – x) \log (1 – f(z_l)) \}
 \end{eqnarray}
$$
Translation from first line to second line is called **monte carlo estimation**.  
Minimizing the difference stricts the encoder's divergence.

## 5. GAN

Composition of generator and discriminator MLPs. It is represented as a two-player minimax game.

<img src=http://www.plantuml.com/plantuml/svg/TP512i8m44NtEKMawrv1N0ZUmLrDnXfCfp2DkBPTg1iF80KttGZI0uZYOKBr6aQ4rchTpl_pClzda9Y0p2BIZ42O04CDV0G859YOkATLE3CX0U27NgskZ-tHhagRhMrsiUdjZt6e4a4gKlY64SWFyPwVkYaJlG3Map1L21nZbG0NYeHRyIOw47FsdjbiPzF2fkTnRPPW75x5-BNGp1_vmLrVROggt3FM8BXFhwVzVXEUDJNKuczorX73b_4IdRiLdtMqLdFRpdPjjQbr-m1Gkbzx0W00 width=400>

**Modified loss**

$$minEz [−logDθ(Gφ(z))]$$

### 5-1. Cycle GAN
GAN with extra MLP after
