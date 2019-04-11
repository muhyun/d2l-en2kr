# Weight Decay

# Weight 감퇴시키기 (Weight decay)

In the previous section, we encountered overfitting and the need for capacity control. While increasing the training data set may mitigate overfitting, obtaining additional training data is often costly, hence it is preferable to control the complexity of the functions we use. In particular, we saw that we could control the complexity of a polynomial by adjusting its degree. While this might be a fine strategy for problems on one-dimensional data, this quickly becomes difficult to manage and too coarse. For instance, for vectors of dimensionality $D​$ the number of monomials of a given degree $d​$ is ${D -1 + d} \choose {D-1}​$. Hence, instead of controlling for the number of functions we need a more fine-grained tool for adjusting function complexity.

앞 절에서 우리는 overfitting에 대해서 알아봤고, 이를 해결하기 위해서 용량 제어 (capacity control)의 필요성에 대해서도 이야기했습니다. 학습 데이터셋의 양을 늘리는 것은 overfitting 문제를 해결할 수도 있지만, 학습 데이터를 추가로 확보하는 것은 일반적으로 어려운 일입니다. 그렇기 때문에, 사용하는 함수의 복잡도를 조정하는 것을 더 선호합니다. 구체적으로는 차수를 조정해서 다항식의 복잡도를 조절할 수 있는 것을 확인했습니다. 이 방법은 일차원 데이터를 다루는 문제에 대해서는 좋은 전략이 될 수 있지만, 이 방법은 쉽게 복잡해지기 때문에 관리가 어려워질 수 있고, 너무 투박한 방법입니다. 예를 들면, $D$ 차원 벡터의 경우, $d$  차수에 대한 단항의 개수는  ${D -1 + d} \choose {D-1}$ 가 됩니다. 따라서, 여러 함수에 대한 제어를 하는 것보다는 함수의 복잡도를 조절할 수 있는 보다 정교한 툴이 필요합니다.

## Squared Norm Regularization

One of the most commonly used techniques is weight decay. It relies on the notion that among all functions $f​$ the function $f = 0​$ is the simplest of all. Hence we can measure functions by their proximity to zero. There are many ways of doing this. In fact there exist entire branches of mathematics, e.g. in functional analysis and the theory of Banach spaces which are devoted to answering this issue.

가장 많이 사용하는 기법 중에 하나로 weight decay가 있습니다. 이 방법은 모든 함수 $f$ 들 중에서 $f=0$ 이 가장 간단한 형태라는 것에 착안하고 있습니다. 따라서, 0와 얼마나 가까운 가를 통해서 함수를 측정할 수 있습니다. 이를 측정하는 방법은 다양한데 사실은 별도의 수학 분야가 존재합니다. 예를 들면, 이 문제에 대한 답을 찾는 것에 목적을 두고 있는 함수 분석과 Banach 공간 이론 (the theory of Banach spaces)를 들 수 있습니다.

For our purpose something much simpler will suffice:
A linear function $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ can be considered simple if its weight vector is small. We can measure this via $\|\mathbf{w}\|^2$. One way of keeping the weight vector small is to add its value as a penalty to the problem of minimizing the loss. This way if the weight vector becomes too large, the learning algorithm will prioritize minimizing $\mathbf{w}$ over minimizing the training error. That's exactly what we want. To illustrate things in code, consider the previous section on [“Linear Regression”](linear-regression.md). There the loss is given by

우리의 목적을 위해서는 아주 간단한 것을 사용해도 충분합니다:

선형 함수 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$  에서 weight vector가 작을 경우 ''이 함수는 간단하다''라고 간주합니다. 이것은  $\|\mathbf{w}\|^2$ 로 측정될 수 있습니다. weight vector를 작게 유지하는 방법 중에 하나는 loss를 최소화하는 문제에 그 값을 penalty로 더하는 것입니다. 이렇게 하면, weight vector가 너무 커지면, 학습 알고리즘은 학습 오류를 최소화하는 것보다 $\mathbf{w}$ 를 최소화하는데 우선 순위를 둘 것입니다. 이것이 우리가 정확히 원하는 것입니다. 코드에서 이를 설명하기 위해서,  앞 절의 [“Linear Regression”](linear-regression.md) 를 고려해보면, loss는 다음과 같이 주어집니다.

$$l(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Recall that $\mathbf{x}^{(i)}$ are the observations, $y^{(i)}$ are labels, and $(\mathbf{w}, b)$ are the weight and bias parameters respectively. To arrive at the new loss function which penalizes the size of the weight vector we need to add $\|\mathbf{w}\|^2$, but how much should we add? This is where the regularization constant (hyperparameter) $\lambda$ comes in:

위 수식에서 $\mathbf{x}^{(i)}$ 는 관찰들이고,  $y^{(i)}$ 는 label, $(\mathbf{w}, b)$ 는 weight와 bias 파라메터들입니다. weight vector의 크기에 대한 패널티를 주는 새로운 loss 함수를 만들기 위해서,  $\|\mathbf{w}\|^2$ 를 더합니다. 하지만, 얼마나 더해야 할까요? 이를 조절하는 정규화 상수(regularization constant)인  $\lambda$  hyperparameter가 그 역할을 합니다.

$$l(\mathbf{w}, b) + \frac{\lambda}{2} \|\boldsymbol{w}\|^2$$

$\lambda \geq 0$ governs the amount of regularization. For $\lambda = 0$ we recover the previous loss function, whereas for $\lambda > 0$ we ensure that $\mathbf{w}$ cannot grow too large. The astute reader might wonder why we are squaring the weight vector. This is done both for computational convenience since it leads to easy to compute derivatives, and for statistical performance, as it penalizes large weight vectors a lot more than small ones. The stochastic gradient descent updates look as follows:

$\lambda \geq 0$  는 정규화(regularzation)을 정도를 조절합니다.  $\lambda = 0$ 인 경우, 원래의 loss 함수가되고,  $\lambda > 0$ 이면,  $\mathbf{w}$ 가 너무 커지지 않도록 강제합니다. 통찰력이 있는 분은 weight vector를 왜 제곱을 하는지 의아해할 것입니다. 이는 두가지 이유 때문인데, 하나는 미분 계산이 쉬워지기 때문에 연산의 편의성을 위함이고, 다른 하나는 작은 weight vector들 보다 큰 weight vector에 더 많은 패널티를 부여하는 것으로 통계적인 성능 향상을 얻기 위하는 것입니다. Stochastic gradient descent 업데이트는 다음과 같이 이뤄집니다.
$$
\begin{aligned}
w & \leftarrow \left(1- \frac{\eta\lambda}{|\mathcal{B}|} \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),
\end{aligned}
$$

As before, we update $\mathbf{w}$ in accordance to the amount to which our estimate differs from the observation. However, we also shrink the size of $\mathbf{w}$ towards $0$, i.e. the weight 'decays'. This is much more convenient than having to pick the number of parameters as we did for polynomials. In particular, we now have a continuous mechanism for adjusting the complexity of $f$. Small values of $\lambda$ correspond to fairly unconstrained $\mathbf{w}$ whereas large values of $\lambda$ constrain $\mathbf{w}$ considerably. Since we don't want to have large bias terms either, we often add $b^2$ as penalty, too.

이전과 같이, 관찰된 값과 예측된 값의 차이에 따라서  $\mathbf{w}$ 를 업데이트합니다. 하지만,  $\mathbf{w}$ 의 크기를  $0$ 과 가까워지게 줄이고 있습니다. 즉, weight를 쇠약하게(decay) 만듭니다. 이것은 다항식에 파라메터 개수를 선택하는 것보다 더 편한 방법입니다. 특히, $f$ 의 복잡도를 조절하는 연속성이 있는 방법을 갖게 되었습니다. 작은 $\lambda$ 값은  $\mathbf{w}$ 를 적게 제약하는 반면, 큰 값은  $\mathbf{w}$ 를 많이 제약합니다. bias 항 역시 큰 값을 갖기를 원하지 않기 때문에,  $b^2$ 를 패널티로 더하기도 합니다.

## High-dimensional Linear Regression

## 고차원 선형 회귀

For high-dimensional regression it is difficult to pick the 'right' dimensions to omit. Weight-decay regularization is a much more convenient alternative. We will illustrate this below. But first we need to generate some data via

고차원 regression에서 생략할 정확한 차원을 선택하기 어려운데, weight-decay regularization은 아주 간편한 대안이 됩니다. 왜 그런지를 지금부터 설명하겠습니다.. 우선, 아래 공식을 사용해서 데이터를 생성합니다.

$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01)$$

That is, we have additive Gaussian noise with zero mean and variance 0.01. In order to observe overfitting more easily we pick a high-dimensional problem with $d = 200$ and a deliberatly low number of training examples, e.g. 20. As before we begin with our import ritual (and data generation).

즉, 이 식에서는 평균이 0이고 표준편차가 0.01인 Gaussian 노이즈를 추가했습니다. overfitting을 더 잘 재현하기 위해서, 차원 $d$ 가 200인 고차원 문제를 선택하고, 적은 양의 학습 데이터 (20개)를 사용하겠습니다. 이전과 같이 필요한 패키지를 import 합니다.

```{.python .input  n=2}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]
```

## Implementation from Scratch

## 처음부터 구현하기

Next, we will show how to implement weight decay from scratch. For this we simply add the $\ell_2$ penalty as an additional loss term after the target function. The squared norm penalty derives its name from the fact that we are adding the second power $\sum_i x_i^2$. There are many other related penalties. In particular, the $\ell_p$ norm is defined as

다음으로는 weight decay를 직접 구현해보겠습니다. 이를 위해서, 간단하게 target 함수 다음에  $\ell_2$ 패널티를 추가 loss 항목으로 더합니다. Squared norm 패널티라는 이름은 제곱수를 더하는 것,  $\sum_i x_i^2$, 으로 부터 왔습니다. 이 외에도 여러가지 패널티들이 있습니다.  $\ell_p$ norm 은 다음과 같이 정의됩니다.

$$\|\mathbf{x}\|_p^p := \sum_{i=1}^d |x_i|^p$$

### Initialize Model Parameters
### 파라메터 초기화하기

First, define a function that randomly initializes model parameters. This function attaches a gradient to each parameter.

우선 모델 파라메터를 임의로 초기화하는 함수를 정의합니다. 이 함수는 각 파라메터에 gradient를 붙입니다.

```{.python .input  n=5}
def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

### Define $\ell_2$ Norm Penalty
### $\ell_2$ Norm Penalty 정의하기

A convenient way of defining this penalty is by squaring all terms in place and summing them up. We divide by $2$ to keep the math looking nice and simple.

이 패널티를 정의하는 간단한 방법은 각 항을 모두 제곱하고 이를 더하는 것입니다. 수식이 멋지고 간단하게 보이기 위해서 2로 나눕니다.

```{.python .input  n=6}
def l2_penalty(w):
    return (w**2).sum() / 2
```

### Define Training and Testing
### 학습 및 테스트 정의하기

The following defines how to train and test the model separately on the training data set and the test data set. Unlike the previous sections, here, the $\ell_2$ norm penalty term is added when calculating the final loss function. The linear network and the squared loss are as before and thus imported via `d2l.linreg` and `d2l.squared_loss` respectively.

아래 코드는 학습 데이터셋과 테스트 데이터셋을 이용해서 모델을 학습시키고 테스트하는 함수를 정의합니다. 이전 절의 예와는 다르게, 여기서는  $\ell_2​$ norm 패널티를 최종 loss 함수를 계산할 때 더합니다. 선형 네트워크와 squred loss는 이전과 같기 때문에, `d2l.linreg` 와 `d2l.squared_loss` 를 import 해서 사용하겠습니다.

```{.python .input  n=7}
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    train_features, train_labels), batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # The L2 norm penalty term has been added
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('l2 norm of w:', w.norm().asscalar())
```

### Training without Regularization
### Regularization 없이 학습하기

Next, let's train and test the high-dimensional linear regression model. When `lambd = 0` we do not use weight decay. As a result, while the training error decreases, the test error does not. This is a perfect example of overfitting.

자 이제 고차원의 선형 회귀(linear regression) 모델을 학습시키고 테스트해봅니다.  `lambd = 0`  인 경우에는 weight decay를 사용하지 않습니다. 그 결과로, 학습 오류가 줄어드는 반면, 테스트 오류는 줄어들지 않게 됩니다. 즉, overfitting의 완벽한 예제가 만들어졌습니다.

```{.python .input  n=8}
fit_and_plot(lambd=0)
```

### Using Weight Decay
### Weight decay 사용하기

The example below shows that even though the training error increased, the error on the test set decreased. This is precisely the improvement that we expect from using weight decay. While not perfect, overfitting has been mitigated to some extent. In addition, the $\ell_2$ norm of the weight $\mathbf{w}$ is smaller than without using weight decay.

아래 예는 학습 오류는 증가하는 반면, 테스트 오류는 감소하는 것을 보여줍니다. 이것은 weight decay를 사용하면서 예상한 개선된 결과입니다. 완벽하지는 않지만, overfitting 문제가 어느정도 해결되었습니다. 추가로, weight  $\mathbf{w}$ 에 대한  $\ell_2$ norm도 weight decay를 사용하지 않을 때보다 작아졌습니다.

```{.python .input  n=9}
fit_and_plot(lambd=3)
```

## Concise Implementation
## 간결한 구현

Weight decay in Gluon is quite convenient (and also a bit special) insofar as it is typically integrated with the optimization algorithm itself. The reason for this is that it is much faster (in terms of runtime) for the optimizer to take care of weight decay and related things right inside the optimization algorithm itself, since the optimizer itself needs to touch all parameters anyway.

Gluon에는 최적화 알고리즘에 weight decay가 통합되어 있어 더 편하게 적용할 수 있습니다. 그 이유는 optimzier가 모든 파라메터를 직접 다루기 때문에, optimzier가 weight decay를 직접 관리하고, 관련된 것을 최적화 알고리즘에서 다루는 것이 실행 속도면에서 더 빠르기 때문입니다.

Here, we directly specify the weight decay hyper-parameter through the `wd` parameter when constructing the `Trainer` instance. By default, Gluon decays weight and bias simultaneously. Note that we can have *different* optimizers for different sets of parameters. For instance, we can have a `Trainer` with weight decay and one without to take care of $\mathbf{w}$ and $b$ respectively.

아래 예제에서는 `Trainer`  인스턴스를 생성할 때, `wd` 파타메터를 통해서 weight decay hyperparamer를 직접 지정합니다. Gluon의 기본 설정은 weight와 bias를 모두 decay 시킵니다. 다른 종류의 파라메터에 대해서 다른 optimizer를 사용할 수 있습니다. 예를 들면,  $\mathbf{w}$ 에는 weight decay 적용하는 `Trainer` 를 하나 만들고,  $b$  에는 weight decay를 적용하지 않은 다른 `Trainer` 를 각각 만들 수 있습니다.

```{.python .input}
def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    # The weight parameter has been decayed. Weight names generally end with
    # "weight".
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd',
                              {'learning_rate': lr, 'wd': wd})
    # The bias parameter has not decayed. Bias names generally end with "bias"
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd',
                              {'learning_rate': lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            # Call the step function on each of the two Trainer instances to
            # update the weight and bias separately
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net[0].weight.data().norm().asscalar())
```

The plots look just the same as when we implemented weight decay from scratch (but they run a bit faster and are a bit easier to implement, in particular for large problems).

그래프는 weight decay를 직접 구현해서 얻었던 것과 아주 비슷하게 생겼습니다. 하지만, 더 빠르고 더 구현하기 쉬웠고, 큰 문제의 경우에는 더욱 그렇습니다.

```{.python .input}
fit_and_plot_gluon(0)
```

```{.python .input}
fit_and_plot_gluon(3)
```

So far we only touched upon what constitutes a simple *linear* function. For nonlinear functions answering this question is way more complex. For instance, there exist [Reproducing Kernel Hilbert Spaces](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) which allow one to use many of the tools introduced for linear functions in a nonlinear context. Unfortunately, algorithms using them do not always scale well to very large amounts of data. For all intents and purposes of this book we limit ourselves to simply summing over the weights for different layers, e.g. via $\sum_l \|\mathbf{w}_l\|^2$, which is equivalent to weight decay applied to all layers.

지금까지 우리는 간단한 선형 함수를 구성하는 것들만을 다뤘습니다. 비선형 함수에 대해서 이것들을 다루는 것은 훨씬 더 복잡합니다. 예를 들어,  [Reproducing Kernel Hilbert Spaces](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) 라는 것이 있는데, 이를 이용하면 선형 함수에서 사용한 많은 도구들을 비선형에서 사용할 수 있게 해줍니다. 하지만 안타깝게도, 사용되는 알고리즘들이 데이터가 매우 많은 경우 잘 동작하지 않는 확장성 문제가 있습니다. 따라서, 이 책의 목적을 위해서 우리는 각 레이어의 weight 들을 단순히 더하는 방법, $\sum_l \|\mathbf{w}_l\|^2$ 을 사용하겠습니다. 이렇게 하는 것은 전체 레이어에 weight decay를 적용하는 것과 같습니다.


## Summary
## 요약

* Regularization is a common method for dealing with overfitting. It adds a penalty term to the loss function on the training set to reduce the complexity of the learned model.
* One particular choice for keeping the model simple is weight decay using an $\ell_2$ penalty. This leads to weight decay in the update steps of the learning algorithm.
* Gluon provides automatic weight decay functionality in the optimizer by setting the hyperparameter `wd`.
* You can have different optimizers within the same training loop, e.g. for different sets of parameters.
* 정규화(regularization)은 overfitting을 다루는 일반적인 방법입니다. 학습된 모델의 복잡도를 줄이기 위해서 학습 데이터에 대한 loss 함수의 값에 패널티 항목을 더합니다.
* 모델을 간단하게 유지하는 방법으로  $\ell_2$ 패널티를 사용하는 weight decay를 선택했습니다. 이를 통해서, 학습 알고리즘의 업데이트 단계에서 weight decay가 적용됩니다.
* Gluon은 optimizer에 hyperparamer `wd` 를 설정하는 것으로 weight decay 기능을 자동으로 추가할 수 있습니다.
* 같은 학습에서 파라메터마다 다른 optimizer를 적용할 수 있습니다.


## Problems
## 문제

1. Experiment with the value of $\lambda$ in the estimation problem in this page. Plot training and test accuracy as a function of $\lambda$. What do you observe?
1. Use a validation set to find the optimal value of $\lambda$. Is it really the optimal value? Does this matter?
1. What would the update equations look like if instead of $\|\mathbf{w}\|^2$ we used $\sum_i |w_i|$ as our penalty of choice (this is called $\ell_1$ regularization).
1. We know that $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Can you find a similar equation for matrices (mathematicians call this the [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm))?
1. Review the relationship between training error and generalization error. In addition to weight decay, increased training, and the use of a model of suitable complexity, what other ways can you think of to deal with overfitting?
1. In Bayesian statistics we use the product of prior and likelihood to arrive at a posterior via $p(w|x) \propto p(x|w) p(w)$. How can you identify $p(w)$ with regularization?
1. 이 장에서의 예측 문제에서  $\lambda$ 값을 실험해보세요.  $\lambda$ 에 대한 함수의 형태로 학습 정확도와 테스트 정확도를 도식화해보세요. 어떤 것이 관찰되나요?
1. 검증 데이터셋을 이용해서 최적의 $\lambda$ 값을 찾아보세요. 찾은 값이 진짜 최적 값인가요? 진짜 값을 찾는 것이 중요한가요?
1. 패널티 항목으로 $\|\mathbf{w}\|^2$ 대신  $\sum_i |w_i|$ 를 사용하면 업데이트 공식이 어떻게 될까요? (이는  $\ell_1$ regularzation이라고 합니다.)
1. $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$ 입니다. 행렬에서 비슷한 공식을 찾아볼 수 있나요? (수학자들은 이를 [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) 이라고 합니다)
1. 학습 오류와 일반화 오류의 관계를 복습해보세요. weight decay, 학습 데이터셋 늘리기, 적당한 복잡도를 갖는 모델 사용하기 외에, overfitting 을 다를 수 있는 방법이 어떤 것들이 있을까요?
1. **베이시안 통계에서,  prior 와  likelihood 곱을 이용해서 posteror를 구할 수 있습니다.  $p(w|x) \propto p(x|w) p(w)$.  $p(w)​$ 가 regularization와 어떻게 동일할까요?**

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2342)

![](../img/qr_weight-decay.svg)
