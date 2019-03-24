# Parameter Management

The ultimate goal of training deep networks is to find good parameter values for a given architecture. When everything is standard, the `nn.Sequential` class is a perfectly good tool for it. However, very few models are entirely standard and most scientists want to build things that are novel. This section shows how to manipulate parameters. In particular we will cover the following aspects:

딥 네트워크 학습의 최종 목표는 주어진 아키텍처에 가장 잘 맞는 파라메터 값들을 찾는 것입니다. 일반적인 것 또는 표준에 준하는 것들을 다룰 때는 `nn.Sequential` 클래스가 이를 위한 완벽한 도구가 될 수 있습니다. 하지만, 소수의 모델이 완전히 표준이고, 대부분의 과학자들은 독창적인 것을 만들기를 원합니다. 이 절에서는 파라메터를 다루는 방법에 대해서 살펴보겠습니다. 좀 더 자세하게는 아래와 같은 것들을 포함합니다.

* Accessing parameters for debugging, diagnostics, to visualize them or to save them is the first step to understanding how to work with custom models.
* Secondly, we want to set them in specific ways, e.g. for initialization purposes. We discuss the structure of parameter initializers.
* Lastly, we show how this knowledge can be put to good use by building networks that share some parameters.
* 디버깅이나 분석을 위해서 파라메터를 접근하고, 그것들을 시각화하거나 저장하는 것을 통해서 커스텀 모델을 어떻게 만들어야하는지 이해를 시작하겠습니다.
* 다음으로는 초기화 목적 등을 위해서 특별한 방법으로 파라메터들을 설정해야 하는데, 이를 위해서 파라메터 초기화 도구의 구조에 대해서 논의합니다.
* 마지막으로 일부 파라메터를 공유하는 네트워크를 만들면서 이 내용들이 어떻게 적용되는지 보겠습니다.

As always, we start from our trusty Multilayer Perceptron with a hidden layer. This will serve as our choice for demonstrating the various features.

지금까지 그랬듯이 hidden 래이어를 갖는 multilayer perceptron으로 부터 시작하겠습니다. 이를 이용해서 다양한 특징들을 살펴봅니다.

```{.python .input  n=1}
from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # Use the default initialization method

x = nd.random.uniform(shape=(2, 20))
net(x)  # Forward computation
```

## Parameter Access

In the case of a Sequential class we can access the parameters with ease, simply by indexing each of the layers in the network. The params variable then contains the required data. Let's try this out in practice by inspecting the parameters of the first layer.

Sequential 클래스의 경우, 네트워크의 각 래이어의 인덱스를 사용해서 파라메터를 쉽게 접근할 수 있습니다. params 변수가 필요한 데이터를 가지고 있습니다. 자 그럼 첫번째 래이어의 파라메터를 조사하는 것을 직접해보겠습니다.

```{.python .input  n=2}
print(net[0].params)
print(net[1].params)
```

The output tells us a number of things. Firstly, the layer consists of two sets of parameters: `dense0_weight` and `dense0_bias`, as we would expect. They are both single precision and they have the necessary shapes that we would expect from the first layer, given that the input dimension is 20 and the output dimension 256. In particular the names of the parameters are very useful since they allow us to identify parameters *uniquely* even in a network of hundreds of layers and with nontrivial structure. The second layer is structured accordingly.

위 코드의 수행 결과는 많은 것을 우리에게 알려줍니다. 첫번째 정보는 예상대로 이 래이어는 파라메터들의 두개의 세트, `dense0_weight` 와 `dense0_bias`, 로 구성되어 있는 것을 확인할 수 있습니다. 이 값들은 모두 single precision이고, 입력 디멘전이 20이고 출력 디멘전이 256인 첫번째 래이어에 필요한 shape를 갖고 있습니다. 특히, 파라메터들의 이름이 주어지는데 이는 아주 유용합니다. 이름을 사용하면 간단하지 않은 구조를 갖는 수백개의 래이어들로 구성된 네트워크에서 파라메터를 쉽게 지정할 수 있기 때문입니다. 두번째 래이어도 같은 방식으로 구성되어 있는 것을 확인할 수 있습니다.

### Targeted Parameters

In order to do something useful with the parameters we need to access them, though. There are several ways to do this, ranging from simple to general. Let's look at some of them.

파라메터를 가지고 뭔가 유용한 일을 하기를 원한다면 이 값들을 접근할 수 있어야 합니다. 간단한 방법부터 일반적인 방법까지 다양한 방법이 있는데, 몇가지를 살펴보겠습니다.

```{.python .input  n=3}
print(net[1].bias)
print(net[1].bias.data())
```

The first returns the bias of the second layer. Sine this is an object containing data, gradients, and additional information, we need to request the data explicitly. Note that the bias is all 0 since we initialized the bias to contain all zeros. Note that we can also access the parameters by name, such as `dense0_weight`. This is possible since each layer comes with its own parameter dictionary that can be accessed directly. Both methods are entirely equivalent but the first method leads to much more readable code.

첫번째 코드는 두번째 래이어의 bias를 출력합니다. 이는 데이터, gradient 그리고 추가적인 정보를 가지고 있는 객체이기에, 우리는 데이터를 명시적으로 접근해야합니다. 우리는 bias를 모두 0으로 초기화했기 때문에 bias가 모두 0임을 기억해두기 바랍니다. 이 값은 파라메터의 이름, `dense0_weight`, 을 이용해서 직접 접근할 수도 있습니다. 이렇게 할 수 있는 이유는 모든 래이어는 직접 접근할 수 있는 고유의 파라메터 dictionary를 갖고있기 때문입니다. 이 두 방법은 완전이 동일하나, 첫번째 방법이 조금더 읽기 쉽습니다.

```{.python .input  n=4}
print(net[0].params['dense0_weight'])
print(net[0].params['dense0_weight'].data())
```

Note that the weights are nonzero. This is by design since they were randomly initialized when we constructed the network. `data` is not the only function that we can invoke. For instance, we can compute the gradient with respect to the parameters. It has the same shape as the weight. However, since we did not invoke backpropagation yet, the values are all 0.

weight들이 모두 0이 아닌 값으로 되어 있음을 주목하세요. 우리가 네트워크를 만들 때, 이 값들은 난수값으로 초기화했기 때문에 그렇습니다. `data`  함수만 있는 것이 아닙니다. 예를 들어 파라메터에 대해서 gradient를 계산하고자 할 수도 있습니다. 이 결과는 weight와 같은 shape을 갖게됩니다. 하지만, back propagation을 아직 실행하지 않았기 때문에 이 값들은 모두 0으로 보여질 것입니다.

```{.python .input  n=5}
net[0].weight.grad()
```

### All Parameters at Once

Accessing parameters as described above can be a bit tedious, in particular if we have more complex blocks, or blocks of blocks (or even blocks of blocks of blocks), since we need to walk through the entire tree in reverse order to how the blocks were constructed. To avoid this, blocks come with a method `collect_params` which grabs all parameters of a network in one dictionary such that we can traverse it with ease. It does so by iterating over all constituents of a block and calls `collect_params` on subblocks as needed. To see the difference consider the following:

위 방법으로 파라메터를 접근하는 것은 다소 지루할 수 있습니다. 특히, 더 복잡한 블럭들을 갖거나, 블럭들로 구성된 블럭 (심지어는 블럭들을 블럭들의 블럭)으로 구성된 네트워크인 경우, 블럭들이 어떻게 생성되었는지 알기 위해서 전체 트리를 모두 뒤져봐야하는 경우가 그런 예입니다. 이를 피하기 위해서, 블럭은 `collect_params` 라는 메소드를 제공하는데 이를 이용하면 네트워크의 모든 파라메터를 하나의 dictionary에 담아주고, 쉽게 조회할 수 있습니다. 이는 내부적으로 블럭의 모든 구성 요소들을 방문하면서 필요할 경우 서브블럭들에 `collect_params` 함수를 호출하는 식으로 동작합니다. 차이를 확인하기 위해서 아래 코드를 살보보겠습니다.

```{.python .input  n=6}
# parameters only for the first layer
print(net[0].collect_params())
# parameters of the entire network
print(net.collect_params())
```

This provides us with a third way of accessing the parameters of the network. If we wanted to get the value of the bias term of the second layer we could simply use this:

이렇게 해서 네트워크의 파라메터를 접근하는 세번째 방법을 배웠습니다. 두번째 래이어의 bias 값을 확인하는 코드는 아래와 같이 간단하게 작성할 수 있습니다.

```{.python .input  n=7}
net.collect_params()['dense1_bias'].data()
```

Throughout the book we'll see how various blocks name their subblocks (Sequential simply numbers them). This makes it very convenient to use regular expressions to filter out the required parameters.

이 책에서 설명을 계속하면서, 블럭들의 하위 블럭에 이름이 어떻게 부여되는지 보게될 것입니다. (그 중에, Sequential의 경우는 숫자를 할당합니다.) 이름 할당 규칙은 필요한 파라메터만 필터링하는 정규식을 사용할 수 있게해서 아주 편리합니다.

```{.python .input  n=8}
print(net.collect_params('.*weight'))
print(net.collect_params('dense0.*'))
```

### Rube Goldberg strikes again

Let's see how the parameter naming conventions work if we nest multiple blocks inside each other. For that we first define a function that produces blocks (a block factory, so to speak) and then we combine these inside yet larger blocks.

블럭들이 중첩되어 있는 경우 파라메터의 이름이 어떤식으로 매겨지는지 보겠습니다. 이를 위해서 우리는 블럭들을 생성하는 함수(block factory 라고 불릴 수 있는) 를 정의하고, 이를 이용해서 더 큰 블럭들이 블럭을 포함시켜보겠습니다.

```{.python .input  n=20}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(x)
```

Now that we are done designing the network, let's see how it is organized. `collect_params` provides us with this information, both in terms of naming and in terms of logical structure.

네트워크를 설계했으니, 어떻게 구성되는지 확인해봅니다. `collect_params` 를 이용하면 이름과 논리적인 구조에 대한 정보를 얻을 수 있습니다.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

Since the layers are hierarchically generated, we can also access them accordingly. For instance, to access the first major block, within it the second subblock and then within it, in turn the bias of the first layer, we perform the following.

래이어들이 계층적으로 생성되어 있으니, 우리도 래이어들을 그렇게 접근할 수 있습니다. 예를 들어서, 첫번째 큰 블럭의 두번째 하위 블럭의 첫번째 래이어의 bias 값은 다음과 같이 접근이 가능합니다.

```{.python .input}
rgnet[0][1][0].bias.data()
```

## Parameter Initialization

Now that we know how to access the parameters, let's look at how to initialize them properly. We discussed the need for [Initialization](../chapter_deep-learning-basics/numerical-stability-and-init.md) in the previous chapter. By default, MXNet initializes the weight matrices uniformly by drawing from $U[-0.07, 0.07]$ and the bias parameters are all set to $0$. However, we often need to use other methods to initialize the weights. MXNet's `init` module provides a variety of preset initialization methods, but if we want something out of the ordinary, we need a bit of extra work.

자 이제 파라메터를 어떻게 접근할 수 있는지 알게되었으니, 파라메터를 어떻게 적절하게 초기화할 수 있을지를 살펴볼 차례입니다. 이전 장에서 [초기화](../chapter_deep-learning-basics/numerical-stability-and-init.md) 가 왜 필요한지를 설명했습니다. 기본 설명으로는 MXNet은 weight 행렬은  $U[-0.07, 0.07]$ 을 따르는 균일한 난수로, bias 파라메터는 모두 0으로 설정합니다. 하지만, 때로는 weight 값을 다르게 초기화해야할 필요가 있습니다. MXNet의 `init` 모듈은 미리 설정된 다양한 초기화 방법들을 제공하는데, 만약 특별한 방법으로 초기화하는 것이 필요하다면 몇 가지 추가적인 일이 필요합니다.

### Built-in Initialization

Let's begin with the built-in initializers. The code below initializes all parameters with Gaussian random variables.

빌트인 초기화 방법들을 우선 살펴보겠습니다. 아래 코드는 모든 파라메터를 Gaussian 랜덤 변수로 초기화하는 예제입니다.

```{.python .input  n=9}
# force_reinit ensures that the variables are initialized again, regardless of
# whether they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

If we wanted to initialize all parameters to 1, we could do this simply by changing the initializer to `Constant(1)`.

만약 파라메터들을 모두 1로 초기화하고 싶다면, 초기화 방법을 `Constant(1)` 로 바꾸기만 하면됩니다.

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

If we want to initialize only a specific parameter in a different manner, we can simply set the initializer only for the appropriate subblock (or parameter) for that matter. For instance, below we initialize the second layer to a constant value of 42 and we use the `Xavier` initializer for the weights of the first layer.

만약 특정 파라메터만 다른 방법으로 초기화를 하고 싶다면, 해당하는 서브블럭에 초기화 함수를 지정하는 것으로 간단히 구현할 수 있습니다. 예를 들어, 아래 코드는 두번째 래이어를 42라는 값으로 초기화하고, 첫번째 해이어의 weight들은 `Xavier` 초기화 방법을 적용하고 있습니다.

```{.python .input  n=11}
net[1].initialize(init=init.Constant(42), force_reinit=True)
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[1].weight.data()[0,0])
print(net[0].weight.data()[0])
```

### Custom Initialization

Sometimes, the initialization methods we need are not provided in the `init` module. At this point, we can implement a subclass of the `Initializer` class so that we can use it like any other initialization method. Usually, we only need to implement the `_init_weight` function and modify the incoming NDArray according to the initial result. In the example below, we  pick a decidedly bizarre and nontrivial distribution, just to prove the point. We draw the coefficients from the following distribution:

때로는 우리가 필요한 초기화 방법이 `init` 모듈에 없을 수도 있습니다. 이 경우에는, `Initializer` 클래스의 하위 클래스를 정의해서 다른 초기화 메소드와 같은 방법으로 사용할 수 있습니다. 보통은, `_init_weight` 함수만 구현하면 됩니다. 이 함수는 입력받은 NDArray를 원하는 초기값으로 바꿔줍니다. 아래 예제에서는 이를 잘 보여주기 위해서 다소 이상하고 특이한 분포를 사용해서 값을 초기화합니다.
$$
\begin{aligned}
    w \sim \begin{cases}
        U[5, 10] & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U[-10, -5] & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

```{.python .input  n=12}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

If even this functionality is insufficient, we can set parameters directly. Since `data()` returns an NDArray we can access it just like any other matrix. A note for advanced users - if you want to adjust parameters within an `autograd` scope you need to use `set_data` to avoid confusing the automatic differentiation mechanics.

이 기능이 충분하지 않을 경우에는, 파라메터 값을 직접 설정할 수도 있습니다. `data()` 는 NDArray를 반환하기 때문에, 이를 이용하면 일반적인 행렬처럼 사용하면 됩니다. 고급 사용자들을 위해서 조금 더 설명하면, `autograd` 범위안에서 파라메터를 조정하는 경우에는, 자동 미분 기능이 오작동하지 않도록 `set_data` 를 사용해야하는 것을 기억해두세요.

```{.python .input  n=13}
net[0].weight.data()[:] += 1
net[0].weight.data()[0,0] = 42
net[0].weight.data()[0]
```

## Tied Parameters

In some cases, we want to share model parameters across multiple layers. For instance when we want to find good word embeddings we may decide to use the same parameters both for encoding and decoding of words. We discussed one such case when we introduced [Blocks](model-construction.md). Let's see how to do this a bit more elegantly. In the following we allocate a dense layer and then use its parameters specifically to set those of another layer.

다른 어떤 경우에는, 여러 래이어들이 모델 파라메터를 공유하는 것이 필요하기도 합니다. 예를 들면, 좋은 단어 임베딩을 찾는 경우, 단어 인코딩과 디코딩에 같은 파라메터를 사용하도록 하는 결정할 수 있습니다. 이런 경우는  [Blocks](model-construction.md)에서도 소개되었습니다. 이것을 보다 깔끔하게 구현하는 방법을 알아보겠습니다. 아래 코드에서는 dense 래이어를 하나 정의하고, 다른 래이어에 파라매터값을 동일하게 설정하는 것을 보여주고 있습니다.

```{.python .input  n=14}
net = nn.Sequential()
# We need to give the shared layer a name such that we can reference its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0,0] = 100
# Make sure that they're actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

The above example shows that the parameters of the second and third layer are tied. They are identical rather than just being equal. That is, by changing one of the parameters the other one changes, too. What happens to the gradients is quite ingenious. Since the model parameters contain gradients, the gradients of the second hidden layer and the third hidden layer are accumulated in the `shared.params.grad( )` during backpropagation.

위 예제는 두번째, 세번째 래이어의 파라메터가 묶여있는 것(tied)을 보여줍니다. 이 파라메터들은 값이 같은 수준이 아니라, 동일합니다. 즉, 하나의 파라메터를 바꾸면 다른 파라메터의 값도 함께 바뀝니다. gradient들에 일어나는 현상은 아주 독창적입니다. 모델은 파라메터는 gradient를 갖고 있기 때문에, 두번째와 세번째 래이어의 gradient들은 back propagation 단계에서 `shared.params.grad()` 함수에 의해서 누적됩니다.

## Summary

* We have several ways to access, initialize, and tie model parameters.
* We can use custom initialization.
* Gluon has a sophisticated mechanism for accessing parameters in a unique and hierarchical manner.
* 모델 파라메터를 접근하고, 초기화하고, 서로 묶는 다양한 방법이 있습니다.
* 커스텀 초기활를 사용할 수 있습니다.
* Gluon은 독특하고 계층적인 방법으로 파라메터를 접근하는 정교한 방법을 제공합니다.


## Problems

1. Use the FancyMLP definition of the [previous section](model-construction.md) and access the parameters of the various layers.
1. Look at the [MXNet documentation](http://beta.mxnet.io/api/gluon-related/mxnet.initializer.html) and explore different initializers.
1. Try accessing the model parameters after `net.initialize()` and before `net(x)` to observe the shape of the model parameters. What changes? Why?
1. Construct a multilayer perceptron containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.
1. Why is sharing parameters a good idea?
1.  [이전 절](model-construction.md) 의 FancyMLP 정의를 사용해서, 다양한 래이어의 파라메터를 접근해보세요.
1.  [MXNet documentation](http://beta.mxnet.io/api/gluon-related/mxnet.initializer.html) 의 다양한 초기화 방법들을 살펴보세요.
1. `net.initialize()` 수행 후와 `net(x)` 수행 전에 모델 파라메터를 확인해서, 모델 파라메터들의 shape를 관찰해보세요. 무엇인 바뀌어 있고, 왜 그럴까요?
1. 파라메터를 공유하는 래이어를 갖는 multilayer perceptron을 만들어서 학습을 시켜보세요. 학습 과정을 수행하면서 모델 각 래이어의 파라메터들과 gradient 값을 관찰해보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2326)

![](../img/qr_parameters.svg)
