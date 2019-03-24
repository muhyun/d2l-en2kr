# Implementation of Multilayer Perceptron from Scratch

Now that we learned how multilayer perceptrons (MLPs) work in theory, let's implement them. First, import the required packages or modules.

multilayer perceptron (MLP)가 어떻게 작동하는지 이론적으로 배웠으니, 직접 구현해보겠습니다. 우선 관련 패키지와 모듈을 import 합니다.

```{.python .input  n=9}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import nd
from mxnet.gluon import loss as gloss
```

We continue to use the Fashion-MNIST data set. We will use the Multilayer Perceptron for image classification.

이 예제에서도 Fashion-MNIST 데이터셋을 사용해서, 이미지를 분류하는데 multilayer perceptron을 사용하겠습니다.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initialize Model Parameters

We know that the dataset contains 10 classes and that the images are of $28 \times 28 = 784$ pixel resolution. Thus the number of inputs is 784 and the number of outputs is 10. Moreover, we use an MLP with one hidden layer and we set the number of hidden units to 256, but we could have picked some other value for this *hyperparameter*, too. Typically one uses powers of 2 since things align more nicely in memory.

이미 알아봤듯이 이 데이터셋은 10개의 클래스로 구분되어 있고, 각 이미지는  $28 \times 28 = 784$ 픽셀의 해상도를 가지고 있습니다. 따라서, 입력은 개수는 784개이고, 출력은 10개가 됩니다. 우리는 한개의 hidden 래이어를 갖는 MLP를 만들어보겠는데, 이 hidden 래이어는 256개의 hidden unit을 갖도록 하겠습니다. 만약 원한다면 hyperparameter인 hidden unit 개수를 다르게 설정할 수도 있습니다. 일반적으로, unit 의 개수는 메모리에 잘 배치될 수 있도록 2의 지수승의 숫자로 선택합니다.

```{.python .input  n=3}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

## Activation Function

Here, we use the underlying `maximum` function to implement the ReLU, instead of invoking `ReLU` directly.

여기서 `ReLU` 를 직접 호출하는 대신, ReLU를 `maximum` 함수를 이용해서 정의합니다.

```{.python .input  n=4}
def relu(X):
    return nd.maximum(X, 0)
```

## The model

As in softmax regression, using `reshape` we change each original image to a length vector of  `num_inputs`. We then implement implement the MLP just as discussed previously.

Softmax regression에서 그랬던 것처럼, `reshape` 함수를 이용해서 원래 이미지를 `num_inputs` 크기의 백터로 변환한 다음에 앞에서 설명한 대로 MLP를 구현합니다.

```{.python .input  n=5}
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2
```

## The Loss Function

For better numerical stability, we use Gluon's functions, including softmax calculation and cross-entropy loss calculation. We discussed the intricacies of that in the [previous section](mlp.md). This is simply to avoid lots of fairly detailed and specific code (the interested reader is welcome to look at the source code for more details, something that is useful for implementing other related functions).

더 나은 계산의 안정성을 위해서, softmax 계산과 cross-entropy loss 계산은 Gluon 함수를 이용하겠습니다. 왜 그런지는 [앞 절](mlp.md)에서 이 함수의 구현에 대한 복잡성을 이야기했으니 참고 바랍니다. Gluon 함수를 이용하면 코드를 안전하게 구현하기 위해서 신경써야하는 많은 세밀한 것들을 간단하게 피할 수 있습니다. (자세한 내용이 궁금하다면 소스 코드를 살펴보기 바랍니다. 소스 코드을 보면 다른 관련 함수를 구현하는데 유용한 것들을 배울 수도 있습니다.)

```{.python .input  n=6}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## Training

Steps for training the Multilayer Perceptron are no different from Softmax Regression training steps.  In the `d2l` package, we directly call the `train_ch3` function, whose implementation was introduced [here](softmax-regression-scratch.md). We set the number of epochs to 10 and the learning rate to 0.5.

Multilayer perceptron을 학습시키는 단계는 softmax regression 학습과 같습니다. `g2l` 패키지에서 제공하는 `train_ch3` 함수를 직접 호출합니다. 이 함수의 구현은 [여기](softmax-regression-scratch.md) 를 참고하세요. 총 epoch 수는 10으로 learning rate는 0.5로 설정합니다.

```{.python .input  n=7}
num_epochs, lr = 10, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
```

To see how well we did, let's apply the model to some test data. If you're interested, compare the result to corresponding [linear model](softmax-regression-scratch.md).

학습이 잘 되었는지 확인하기 위해서, 모델을 테스트 데이터에 적용해 보겠습니다. 이 모델의 성능이 궁금하다면, 동일한 분류를 수행하는 [linear 모델](softmax-regression-scratch.md) 의 결과와 비교해보세요.

```{.python .input}
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [truelabel + '\n' + predlabel
          for truelabel, predlabel in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
```

This looks slightly better than before, a clear sign that we're on to something good here.

이전보다 조금 성능이 좋아 보이는 것이, MLP를 사용하는 것이 좋은 것임을 알 수 있습니다.

## Summary

We saw that implementing a simple MLP is quite easy, when done manually. That said, for a large number of layers this can get quite complicated (e.g. naming the model parameters, etc).

간단한 MLP는 직접 구현하는 것이 아주 쉽다는 것을 확인했습니다. 하지만, 많은 수의 래이어를 갖는 경우에는 굉장히 복잡해질 수 있습니다. (예를 들면 모델 파라메터 이름을 정하는 것 등)

## Problems

1. Change the value of the hyper-parameter `num_hiddens` in order to see the result effects.
1. Try adding a new hidden layer to see how it affects the results.
1. How does changing the learning rate change the result.
1. What is the best result you can get by optimizing over all the parameters (learning rate, iterations, number of hidden layers, number of hidden units per layer)?
1. `num_hiddens` hyperparameter를 변경해서 결과가 어떻게 영향을 받는지 확인해보세요.
1. 새로운 hidden 래이어를 추가해서 어떤 영향을 미치는지 확인해보세요.
1. learning rate를 변경하면 결과가 어떻게 되나요?
1. 모든 hyperparameter (learing rate, epoch 회수, hidden layer 개수, 각 래이어의 hidden unit 개수)의 조합을 통해서 얻을 수 있는 가장 좋은 결과는 무엇인가요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2339)

![](../img/qr_mlp-scratch.svg)
