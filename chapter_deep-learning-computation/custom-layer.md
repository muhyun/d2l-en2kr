# Custom Layers

One of the reasons for the success of deep learning can be found in the wide range of layers that can be used in a deep network. This allows for a tremendous degree of customization and adaptation. For instance, scientists have invented layers for images, text, pooling, loops, dynamic programming, even for computer programs. Sooner or later you will encounter a layer that doesn't exist yet in Gluon, or even better, you will eventually invent a new layer that works well for your problem at hand. This is when it's time to build a custom layer. This section shows you how.

딥러닝의 성공 요인 중에 하나는 딥 네트워크에서 사용할 수 있는 다양한 종류의 래이어가 있다는 점에서 찾아볼 수 있습니다. 즉, 다양한 형태의 래이어를 사용해서 많은 종류의 커스터마이징와 다양한 문제에 적용이 가능하게 되었습니다. 예를 들면, 과학자들이 이미지, 텍스트, 풀링, loop, 동적 프로그램밍, 그리고 심지어는 컴퓨터 프로그램을 위한 래이어를 발명해왔습니다. 앞으로도 Gluon에 현재 존재하지 않은 새로운 래이어를 만나게될 것이고, 어쩌면 여러분이 만난 문제를 해결하기 위해서 새로운 래이어를 직접 발명할지도 모릅니다. 자 그럼 커스텀 래이어를 만들어 보는 것을 이 절에서 배워보겠습니다.

## Layers without Parameters

Since this is slightly intricate, we start with a custom layer (aka Block) that doesn't have any inherent parameters. Our first step is very similar to when we [introduced blocks](model-construction.md) previously. The following `CenteredLayer` class constructs a layer that subtracts the mean from the input. We build it by inheriting from the Block class and implementing the `forward` method.

커스텀 래이어를 만드는 것은 다소 복잡할 수 있기 때문에, 파라메터를 계승 받지 않는 커스텀 래이어 (또는 Block)를 만드는 것부터 시작해보겠습니다. 첫번째 시작은 이전에 [introduced blocks](model-construction.md) 에서 소개했던 것과 비슷합니다. 아래 `CenteredLayer` 클래스는 입력에서 평균을 빼는 것을 계산하는 래이어를 정의합니다. 우리는 이것을 Block 클래스를 상속하고, `forward` 메소드를 구현해서 만듭니다.

```{.python .input  n=1}
from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```

To see how it works let's feed some data into the layer.

어떻게 동작하는지 보기 위해서, 데이터를 래이어에 입력해봅니다.

```{.python .input  n=2}
layer = CenteredLayer()
layer(nd.array([1, 2, 3, 4, 5]))
```

We can also use it to construct more complex models.

우리는 이를 사용해서 더 복잡한 모델을 만들 수도 있습니다.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

Let's see whether the centering layer did its job. For that we send random data through the network and check whether the mean vanishes. Note that since we're dealing with floating point numbers, we're going to see a very small albeit typically nonzero number.

그럼 이 가운데로 만들어주는 래이어가 잘 작동하는지 보겠습니다. 이를 위해서 난수 데이터를 생성하고, 네트워크에 입력한 후 평균만큼 값이 조정되는지 확입합니다. 우리가 다루는 변수가 실수형이기 때문에, 아죽 작지만 0이 아닌 숫자를 보게될 것임을 염두하세요.

```{.python .input  n=4}
y = net(nd.random.uniform(shape=(4, 8)))
y.mean().asscalar()
```

## Layers with Parameters

Now that we know how to define layers in principle, let's define layers with parameters. These can be adjusted through training. In order to simplify things for an avid deep learning researcher the `Parameter` class and the `ParameterDict` dictionary provide some basic housekeeping functionality. In particular, they govern access, initialization, sharing, saving and loading model parameters. For instance, this way we don't need to write custom serialization routines for each new custom layer.

래이어를 어떻게 정의하는지 원리를 알게되었으니, 파라메터를 갖는 래이어를 정의해보겠습니다. 이 파라메터들은 학습을 통해서 조정될 값들입니다. 딥러닝 연구자들의 일을 편하게 만들어 주기위해서, `Parameter`  클래스와 `ParameterDict` dictionary는 많이 사용하는 기능을 제공하고 있습니다. 이 클래스들은 접근을 관리하고, 초기화를 하고, 공유를 하고, 모델 파라메터를 저장하고 로딩하는 기능을 관리해줍니다. 예를 들면, 새로운 커스텀 래이어를 만든 때 매번 직렬화(serialization) 루틴을 작성할 필요가 없습니다. 

For instance, we can use the member variable `params` of the `ParameterDict` type that comes with the Block class. It is a dictionary that maps string type parameter names to model parameters in the `Parameter` type.  We can create a `Parameter` instance from `ParameterDict` via the `get` function.

다른 예로는, Block 클래스와 함께 제공되는  `ParameterDict` 타입인 `params` 를 사용할 수도 있습니다. 이 dictionary는 문자 파입의 파라메터 이름을 `Parameter` 타입의 모델 파라메터로 매핑하는 기능을 제공합니다.  `ParameterDict` 의 `get` 함수를 사용해서 `Parameter` 인스턴스를 생성하는 것도 가능합니다.

```{.python .input  n=7}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```

Let's use this to implement our own version of the dense layer. It has two parameters - bias and weight. To make it a bit nonstandard, we bake in the ReLu activation as default. Next, we implement a fully connected layer with both weight and bias parameters.  It uses ReLU as an activation function, where `in_units` and `units` are the number of inputs and the number of outputs, respectively.

Dense 래이어를 직접 구현해보겠습니다. 이 래이어는 두 파라메터, weight와 bais, 를 갖습니다. 약간 특별하게 만들기 위해서, ReLU activation 함수를 기본으로 적용하도록 만들어봅니다. weight와 bias 파라메터를 갖는 fully connected 래이어를 구현하고, ReLU를 activation 함수로 추가합니다. `in_units`와 `units` 는 각각 입력과 출력의 개수입니다.

```{.python .input  n=19}
class MyDense(nn.Block):
    # units: the number of outputs in this layer; in_units: the number of
    # inputs in this layer
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

Naming the parameters allows us to access them by name through dictionary lookup later. It's a good idea to give them instructive names. Next, we instantiate the `MyDense` class and access its model parameters.

파라메터에 이름을 부여하는 것은 이후에 dictionary 조회를 통해서 원하는 파라메터를 직접 접근할 수 있도록해줍니다. 그렇기 때문에, 잘 설명하는 이름을 정하는 것이 좋은 생각입니다. 자 이제 `MyDense` 클래스의 인스턴스를 만들고 모델 파라메터들을 직접 확인해봅니다.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

We can directly carry out forward calculations using custom layers.

커스텀 래이어의 forward 연산을 수행합니다.

```{.python .input  n=20}
dense.initialize()
dense(nd.random.uniform(shape=(2, 5)))
```

We can also construct models using custom layers. Once we have that we can use it just like the built-in dense layer. The only exception is that in our case size inference is not automagic. Please consult the [MXNet documentation](http://www.mxnet.io) for details on how to do this.

커스텀 래이어를 이용해서 모델은 만들어 보겠습니다. 만들어진 모델은 기본으로 제공되는 dense 래이어처럼 사용할 수 있습니다. 하나 다른 점은 입력, 출력의 크기를 자동으로 계산하는 것이 없다는 점입니다. 어떻게 이 기능을 구현할 수 있는지는 [MXNet documentation](http://www.mxnet.io) 를 참고하세요.

```{.python .input  n=19}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))
```

## Summary

* We can design custom layers via the Block class. This is more powerful than defining a block factory, since it can be invoked in many contexts.
* Blocks can have local parameters.
* Block 클래스를 이용해서 커스텀 래이어를 만들 수 있습니다. 이 방법은 블럭 팩토리를 정의하는 것보다 더 강력한 방법인데, 그 이유는 다양한 context들에서 불려질 수 있기 때문입니다.
* 블럭들은 로컬 파라매터를 갖을 수 있습니다.


## Problems

1. Design a layer that learns an affine transform of the data, i.e. it removes the mean and learns an additive parameter instead.
1. Design a layer that takes an input and computes a tensor reduction, i.e. it returns $y_k = \sum_{i,j} W_{ijk} x_i x_j​$.
1. Design a layer that returns the leading half of the Fourier coefficients of the data. Hint - look up the `fft` function in MXNet.
1. 데이터에 대해서 affine 변환을 학습하는 래이어를 디자인하세요. **예를 들면, 평균 값을 빼고, 대신 더할 파라메터를 학습합니다.**
1. 입력을 받아서 텐서 축소를 하는 래이어를 만들어 보세요. 즉, $y_k = \sum_{i,j} W_{ijk} x_i x_j$ 를 반환합니다.
1. 데어터에 대한 퓨리에 계수의 앞에서 반을 리텅하는 래이어를 만들어보세요. 힌트 - MXnet의 `fft` 함수를 참고하세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2328)

![](../img/qr_custom-layer.svg)
