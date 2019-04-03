# Concise Implementation of Softmax Regression

# Softmax 회귀의 간결한 구현

We already saw that it is much more convenient to use Gluon in the context of [linear regression](linear-regression-gluon.md). Now we will see how this applies to classification, too. We being with our import ritual.

우리는 이미 [선형 회귀 구현](linear-regression-gluon.md)에서 Gluon을 이용하는 것이 아주 편리하다는 것을 보았습니다. 이제 Gluon이 분류에 어떻게 적용되는지 보도록 하겠습니다. 역시 몇가지 패키지와 모듈을 import하는 것으로 시작합니다.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

We still use the Fashion-MNIST data set and the batch size set from the last section.

앞 절과 동일하게 Fashion-MNIST 데이터셋과 같은 배치 크기를 사용합니다.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initialize Model Parameters

## 모델 파라메터 초기화하기

As [mentioned previously](softmax-regression.md), the output layer of softmax regression is a fully connected layer. Therefore, we are adding a fully connected layer with 10 outputs. We initialize the weights at random with zero mean and standard deviation 0.01.

[앞 절](softmax-regression.md)에서 언급했듯이 softmax regression의 output 레이어는 fully connected 레이어입니다. 따라서, 10개의 output을 갖는 fully connected 레이어를 추가하고, weight을 평균이 0이고 표준 편차가 0.01인 분포에서 난수를 뽑아서 초기화를 합니다.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## The Softmax

In the previous example, we calculated our model's output and then ran this output through the cross-entropy loss. At its heart it uses `-nd.pick(y_hat, y).log()`. Mathematically, that's a perfectly reasonable thing to do. However, computationally, things can get hairy, as we've already alluded to a few times (e.g. in the context of [Naive Bayes](../chapter_crashcourse/naive-bayes.md) and in the problem set of the previous chapter). Recall that the softmax function calculates $\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$, where $\hat y_j$ is the j-th element of ``yhat`` and $z_j$ is the j-th element of the input ``y_linear`` variable, as computed by the softmax.

이전 예제에서는 모델의 결과를 계산하고, 이 결과에 cross-entropy loss를 적용 했었습니다. 이는 위해서  `-nd.pick(y_hat, y).log()` 를 이용했습니다. 수학적으로는 이렇게 하는 것이 매우 논리적입니다. 하지만, 이미 수차례 언급했듯이 연산의 관점에서 보면 어려운 문제가 될 수 있습니다.  (예를 들면,  [Naive Bayes](../chapter_crashcourse/naive-bayes.md) 의 예나 이전 장의 문제들 처럼).  ``yhat`` 의 j 번째 원소를  $\hat y_j$ 라고 하고, 입력인 `y_linear` 변수의 j 번째 원소를  $z_j$ 라고 할때, softmax 함수는  $\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$ 를 계산합니다.

If some of the $z_i​$ are very large (i.e. very positive), $e^{z_i}​$ might be larger than the largest number we can have for certain types of ``float`` (i.e. overflow). This would make the denominator (and/or numerator) ``inf`` and we get zero, or ``inf``, or ``nan`` for $\hat y_j​$. In any case, we won't get a well-defined return value for ``cross_entropy``. This is the reason we subtract $\text{max}(z_i)​$ from all $z_i​$ first in ``softmax`` function. You can verify that this shifting in $z_i​$ will not change the return value of ``softmax``.

만약 몇개의  $z_i$ 가 매우 큰 값을 갖는 다면,  $e^{z_i}$ 값이 `float` 변수가 표한할 수 있는 값보다 훨씬 커질 수 있습니다 (overflow). 따라서, 분모 (또는 분자)가 `inf` 가 돼서 결과  $\hat y_j$가 0, `inf` 또는 `nan` 가 될 수 있습니다. 어떤 경우에든지 `cross_entropy` 는 잘 정의된 값을 리턴하지 못할 것입니다. 이런 문제 때문에, `softmax` 함수에서는 모든 $z_i$ 에서  $\text{max}(z_i)$ 뺍니다. 이렇게  $z_i$ 를 이동시키는 것이 `softmax` 의 리턴 값을 변화시키지 않는다는 사실을 확인해볼 수도 있습니다.

After the above subtraction/ normalization step, it is possible that $z_j$ is very negative. Thus, $e^{z_j}$ will be very close to zero and might be rounded to zero due to finite precision (i.e underflow), which makes $\hat y_j$ zero and we get ``-inf`` for $\text{log}(\hat y_j)$. A few steps down the road in backpropagation, we start to get horrific not-a-number (``nan``) results printed to screen.

위에서 설명한 빼기와 normalization 단계를 거친 후에도 $z_j​$ 가 매우 작은 음수값이 될 가능성이 여전히 있습니다. 따라서,  $e^{z_j}​$ 가 0과 매우 근접해지거나 finite precision (underflow) 때문에 0으로 반올림될 수도 있습니다. 이렇게 되면,  $\hat y_j​$ 는 0이 되고, $\text{log}(\hat y_j)​$ 는 `-inf` 가 됩니다. Backpropagation을 몇 번 거치면, 화면에 not-a-number (`nan`) 결과가 출력되는 것을 보게 될 것입니다.

Our salvation is that even though we're computing these exponential functions, we ultimately plan to take their log in the cross-entropy functions. It turns out that by combining these two operators ``softmax`` and ``cross_entropy`` together, we can elude the numerical stability issues that might otherwise plague us during backpropagation. As shown in the equation below, we avoided calculating $e^{z_j}$ but directly used $z_j$ due to $\log(\exp(\cdot))$.

이 문제에 대한 해결책은 지수 함수를 계산함에도 불구하고, cross-entropy 함수에서 이 값에 대한 log 값을 취하도록 합니다. 이 두 연산 `softmax`  와 `cross_entropy` 를 함께 사용해서 수치 안정성 문제를 피하고, backpropagation 과정에서 위 문제를 만나지 않게 할 수 있습니다. 아래 공식에서 보이는 것처럼, $\log(\exp(\cdot))$ 를 사용해서  $e^{z_j}$ 를 계산하는 것을 피하고,  $z_j$ 를 직접 사용할 수 있습니다.
$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}\right) \\
& = \log{(e^{z_j})}-\text{log}{\left( \sum_{i=1}^{n} e^{z_i} \right)} \\
& = z_j -\log{\left( \sum_{i=1}^{n} e^{z_i} \right)}
\end{aligned}
$$

We'll want to keep the conventional softmax function handy in case we ever want to evaluate the probabilities output by our model. But instead of passing softmax probabilities into our new loss function, we'll just pass $\hat{y}$ and compute the softmax and its log all at once inside the softmax_cross_entropy loss function, which does smart things like the log-sum-exp trick ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)).

모델의 확률 결과에 대한 평가를 해야하는 경우에 사용되는 이런 전형적인 softmax 함수를 간편하게 만들고 싶습니다. 하지만, softmax 확률들을 새로운 loss 함수에 대입하는 것 보다는,  $\hat{y}$ 만 전달하면, softmax와 log 값들이 모두 softmax_cross_entropy loss 함수에서 계산되도록 하겠습니다. 이는 log-sum-exp 트릭  ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)) 과 같이 스마트한 것을 하는 것과 같다고 볼 수 있습니다.

```{.python .input  n=4}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## Optimization Algorithm

## 최적화 알고리즘

We use the mini-batch random gradient descent with a learning rate of 0.1 as the optimization algorithm. Note that this is the same choice as for linear regression and it illustrates the portability of the optimizers.

최적화 알고리즘으로 learning rate를 0.1로 하는 미니 배치 random gradient descent를 사용하겠습니다. 이는 선형 회귀에서도 동일하게 사용했는데, optimizer들의 이식성 보여줍니다.

```{.python .input  n=5}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## Training

## 학습

Next, we use the training functions defined in the last section to train a model.

다음으로, 앞 절에서 정의된 학습 함수를 이용해서 모델을 학습 시킵니다.

```{.python .input  n=6}
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```

Just as before, this algorithm converges to a fairly decent accuracy of 83.7%, albeit this time with a lot fewer lines of code than before. Note that in many cases Gluon takes specific precautions beyond what one would naively do to ensure numerical stability. This takes care of many common pitfalls when coding a model from scratch.

이전과 같이 이 알고리즘은 매우 쓸만한 정확도인 83.7%로 수렴하는데, 이전보다 훨씬 더 적은 코드를 가지고 가능합니다. Gluon은 단순하게 구현할 경우 만날 수 있는 수치 안정성을 넘어서 특별한 예방을 포함하고 있기에, 모델을 직접 구현할 때 만날 수 있는 많은 일반적인 위험을 피할 수 있게 해줍니다.

## Problems

## 문제

1. Try adjusting the hyper-parameters, such as batch size, epoch, and learning rate, to see what the results are.
1. Why might the test accuracy decrease again after a while? How could we fix this?
1. batch size, epoch, learning rate와 같은 hyper-parameter를 변경하면서 어떤 결과가 나오는지 보세요.
1. 학습이 진행될 때 어느 정도 지나면 테스트 정확도가 감소할까요? 어떻게 고칠 수 있나요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2337)

![](../img/qr_softmax-regression-gluon.svg)
