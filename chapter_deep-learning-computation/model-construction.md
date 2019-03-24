# Layers and Blocks

One of the key components that helped propel deep learning is powerful software. In an analogous manner to semiconductor design where engineers went from specifying transistors to logical circuits to writing code we now witness similar progress in the design of deep networks. The previous chapters have seen us move from designing single neurons to entire layers of neurons. However, even network design by layers can be tedious when we have 152 layers, as is the case in ResNet-152, which was proposed by [He et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) in 2016 for computer vision problems.
Such networks have a fair degree of regularity and they consist of *blocks* of repeated (or at least similarly designed) layers. These blocks then form the basis of more complex network designs. In short, blocks are combinations of one or more layers. This design is aided by code that generates such blocks on demand, just like a Lego factory generates blocks which can be combined to produce terrific artifacts.

딥러닝이 유명해질 수 있었던 중요 요소들 중에 하나는 바로 강력한 소프트웨어입니다. 반도체 설계를 하는데 엔지니어들이 논리 회로를 트랜지스터로 구현하던 것에서 코드를 작성하는 것으로 넘어간 것과 같은 일이 딥 네트워크 설계에도 비슷하게 일어나고 있습니다. 앞 장들은 단일 뉴런으로 부터 뉴런으로 구성된 전체 래이어들로 옮겨가는 것을 보여줬습니다. 하지만, 컴퓨터 비전 문제를 풀기 위해서 2016년에 [He et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) 에 의해서 제안된 ResNet-152의 경우처럼 152개의 래이어들을 갖는 네트워크를 래이어들을 사용한 네트워크 설계 방법 조차도 지루할 수 있습니다.

이런 네트워크는 많은 정도로 반복적되는 부분을 갖고, 반복되는 (또는 비슷하게 설계된) 래이어들의 *블럭*들로 구성됩니다. 이들 블럭들은 더 복잡한 네트워크 디자인을 구성하는 기본 요소가 됩니다. 간략하게 말하면, 블럭은 하나 또는 그 이상의 래이어의 조합니다. 마치 레고 공장이 만든 블럭을 이용해서 멋진 구조물을 만들 수 있는 것처럼, 이 디자인은 요청에 따라서 브럭을 생성하는 코드의 도움으로 만들어질 수 있습니다.

We start with very simple block, namely the block for a multilayer perceptron, such as the one we encountered [previously](../chapter_deep-learning-basics/mlp-gluon.md). A common strategy would be to construct a two-layer network as follows:

아주 간단한 블럭부터 살펴보겠습니다. 이 블럭은 [앞 장](../chapter_deep-learning-basics/mlp-gluon.md) 에서 본 multilayer perception을 위한 것입니다. 일반적인 방법으로 두개의 래이어를 갖는 네트워크를 다음과 같이 만들 수 있습니다.

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

x = nd.random.uniform(shape=(2, 20))

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

This generates a network with a hidden layer of 256 units, followed by a ReLu activation and another 10 units governing the output. In particular, we used the `nn.Sequential` constructor to generate an empty network into which we then inserted both layers. What exactly happens inside `nn.Sequential` has remained rather mysterious so far. In the following we will see that this really just constructs a block. These blocks can be combined into larger artifacts, often recursively. The diagram below shows how:

이 코드는 256개의 unit들을 갖는 hidden 래이어 한개를 포함한 네트워크를 생성합니다. hidden 래이어는 ReLU activation으로 연결되어 있고, 결과 래이어의 10개 unit들로 연결되어 있습니다. 여기서 우리는 `nn.Sequential` 생성자를 사용해서 빈 네트워크를 만들고, 그 다음에 래이어들을 추가했습니다. 아직은 `nn.Sequential` 내부에서 어떤 일이 벌어지는 지는 미스테리로 남아있습니다. 아래 내용을 통해서 이것은 실제로 블럭을 생성하고 있는 것을 확인할 것입니다. 이 블럭들은 더 큰 결과물로 합쳐지는데 때로는 재귀적으로 합쳐지기도 합니다. 아래 그림은 이 것이 어떻게 일어나는지 보여줍니다.

![Multiple layers are combined into blocks](../img/blocks.svg)

In the following we will explain the various steps needed to go from defining layers to defining blocks (of one or more layers). To get started we need a bit of reasoning about software. For most intents and purposes a block behaves very much like a fancy layer. That is, it provides the following functionality:

래이어를 정의하는 것부터 (하나 또는 그 이상이 래이어들을 갖는) 블럭을 정의하는 데 필요한 다양한 절차에 대해서 설명하겠습니다. 블럭은 멋진 래이어와 비슷하게 동작합니다. 즉, 블럭은 아래 기능을 제공합니다.

1. It needs to ingest data (the input).
1. It needs to produce a meaningful output. This is typically encoded in what we will call the `forward` function. It allows us to invoke a block via `net(X)` to obtain the desired output. What happens behind the scenes is that it invokes `forward` to perform forward propagation.
1. It needs to produce a gradient with regard to its input when invoking `backward`. Typically this is automatic.
1. It needs to store parameters that are inherent to the block. For instance, the block above contains two hidden layers, and we need a place to store parameters for it.
1. Obviously it also needs to initialize these parameters as needed.
1. 데이터 (입력을) 받아야합니다.
1. 의미있는 결과를 출력해야 합니다. 이는 `forward` 라고 불리는 함수에서 처리합니다. 원하는 output을 얻기 위해서 `net(x)` 를 통해서 블럭을 수행할 수도 있는데, 실제로는 forward propagation을 수행하는 `forward` 함수를 호추합니다.
1. `backward` 함수가 호출되면 입력에 대해서 gradient를 생성해야합니다. 일반적으로 이것은 자동으로 이뤄집니다.
1. 블럭에 속한 파라메터들을 저장해야합니다. 예를 들면, 위 블럭은 두개의 hidden 래이어를 갖는데, 파라메터를 저장할 공간이 있어야 합니다.

## A Custom Block

The `nn.Block` class provides the functionality required for much of what we need. It is a model constructor provided in the `nn` module, which we can inherit to define the model we want. The following inherits the Block class to construct the multilayer perceptron mentioned at the beginning of this section. The `MLP` class defined here overrides the `__init__` and `forward` functions of the Block class. They are used to create model parameters and define forward computations, respectively. Forward computation is also forward propagation.

`nn.Block` 클래스는 우리가 필요로하는 기능들을 제공합니다. `nn` 모듈에서 제공하는 모델 생성자로, 우리가 원하는 모델을 정의하기 위해서 상속하는 클래스입니다. 아래 코드는 이 절을 시작할 때 언급한 multilayer perceptron을 생성하기 위해서 Block 클래스를 상속하고 있습니다. 여기서  `MLP` 클래스는 Block 클래스의  `__init__` 과 `forward` 함수를 오버라이드하고 있습니다. 이 함수들은 각각 모델 파라메터들을 생성하고 forward 계산을 정의하는 함수입니다. Forward 연산은 forward propagation을 의미합니다.

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.output = nn.Dense(10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input x
    def forward(self, x):
        return self.output(self.hidden(x))
```

Let's look at it a bit more closely. The `forward` method invokes a network simply by evaluating the hidden layer `self.hidden(x)` and subsequently by evaluating the output layer `self.output( ... )`. This is what we expect in the forward pass of this block.

조금 더 자세히 살펴보겠습니다. `forward` 메소드는 hidden 래이어 `self.hidden(x)` 를 계산하고, 그 값을 이용해서 output 래이어 `self.output(…)` 을 계산합니다. 이것이 이 블럭의 forward 연산에서 해야하는 일입니다.

In order for the block to know what it needs to evaluate, we first need to define the layers. This is what the `__init__` method does. It first initializes all of the Block-related parameters and then constructs the requisite layers. This attached the corresponding layers and the required parameters to the class. Note that there is no need to define a backpropagation method in the class. The system automatically generates the `backward` method needed for back propagation by automatically finding the gradient. The same applies to the `initialize` method, which is generated automatically. Let's try this out:

블럭이 어떤 값을 사용해서 계산을 수행해야하는지를 알기 위해서, 우리는 우선 래이어들을 정의해야합니다. 이는 `__init__` 메소드가 하는 일입니다. 블럭과 관련된 모든 파라메터들을 초기화하고, 필요한 래이어를 생성합니다. 그리고, 관련 래이어들과 클래스에 필요한 파라메터들을 정의합니다. 시스템은 gradient를 자동으로 계산해주는 `backward` 메소드를 자동으로 생성해줍니다.   `initialize` 메소드도 자동으로 생성됩니다. 한번 수행해보겠습니다.

```{.python .input  n=2}
net = MLP()
net.initialize()
net(x)
```

As explained above, the block class can be quite versatile in terms of what it does. For instance, its subclass can be a layer (such as the `Dense` class provided by Gluon), it can be a model (such as the `MLP` class we just derived), or it can be a part of a model (this is what typically happens when designing very deep networks). Throughout this chapter we will see how to use this with great flexibility.

위에서 설명했듯이, 블럭 클래스는 무엇을 하는지에 따라서 아주 다르게 정의될 수 있습니다. 예를 들어, 그것의 하위 클래이스가 (Gluon에서 제공하는 `Dense` 클래스와 같은) 래이어가 될 수도 있고, (우리가 막 정의한 `MLP` 클래스와 같은) 모델이 될 수도 있습니다. 또는 다른 모델의 일부가 될 수도 있습니다. 이는 아주 깊은 네트워크를 디자인할 때 사용되는 방법입니다. 이 장을 통해서 우리는 이것을 아주 유연하게 사용할 수 있는 방법에 대해서 알아보겠습니다.

## A Sequential Block

The Block class is a generic component describing dataflow. In fact, the Sequential class is derived from the Block class: when the forward computation of the model is a simple concatenation of computations for each layer, we can define the model in a much simpler way. The purpose of the Sequential class is to provide some useful convenience functions. In particular, the `add` method allows us to add concatenated Block subclass instances one by one, while the forward computation of the model is to compute these instances one by one in the order of addition.

Block 클래스는 데이터흐름을 기술하는 일반 컴포넌트립니다. 사실 Sequential 클래스는 Block 클래스로부터 정의됩니다. 모델을 forward 연산은 각 래이어에 대한 연산의 단순한 연결이기 때문에, 우리는 모델을 아주 간단한 방법으로 정의할 수 있습니다. Sequential 클래스의 목적은 유용한 편의 함수들을 제공하는 것에 있습니다. 특히, `add` 메소드는 연결된 Block 하위클래스의 인스턴스를 하나씩 더할 수 있게 해주고, 모델의 forward 연산은 이 인스턴들을 더하기 순서대로 계산합니다.

Below, we implement a `MySequential` class that has the same functionality as the Sequential class. This may help you understand more clearly how the Sequential class works.

아래 코드에서 `MySequential` 클래스를 정의했는데, 이는 Sequential 클래스와 같은 기능을 제공합니다. 이를 통해서 Sequential 클래스가 어떻게 동작하는 이해하는데 도움이 될 것입니다.

```{.python .input  n=3}
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume it has
        # a unique name. We save it in the member variable _children of the
        # Block class, and its type is OrderedDict. When the MySequential
        # instance calls the initialize function, the system automatically
        # initializes all members of _children
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            x = block(x)
        return x
```

At its core is the `add` method. It adds any block to the ordered dictionary of children. These are then executed in sequence when forward propagation is invoked. Let's see what the MLP looks like now.

`add` 메소드가 핵심입니다. 이 메소드는 순서가 있는 dictionary에 블럭을 추가하는 일을 합니다. forward propagation이 호출되면 이 블럭들은 순서대로 수행됩니다. MLP가 어떻게 구현되는지 보겠습니다.

```{.python .input  n=4}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

Indeed, it is no different than It can observed here that the use of the `MySequential` class is no different from the use of the Sequential class described in the [“Concise Implementation of Multilayer Perceptron”](../chapter_deep-learning-basics/mlp-gluon.md) section.

실제로,  [“Concise Implementation of Multilayer Perceptron”](../chapter_deep-learning-basics/mlp-gluon.md) 에서 Sequential 클래스를 사용한 것과 `MySequential` 클래스를 사용한 것이 다르지 않다는 것을 볼 수 있습니다.


## Blocks with Code

Although the Sequential class can make model construction easier, and you do not need to define the `forward` method, directly inheriting the Block class can greatly expand the flexibility of model construction. In particular, we will use Python's control flow within the forward method. While we're at it, we need to introduce another concept, that of the *constant* parameter. These are parameters that are not used when invoking backprop. This sounds very abstract but here's what's really going on. Assume that we have some function

Sequential 클래스가 모델 생성을 쉽게 해주고 `forward` 메소스를 별도로 구현할 필요없게 해주지만, Block 클래스를 직접 상속하면 더 유연한 모델 생성을 할 수 있습니다. 특히, forward 메소스에서 Python의 제어 흐름을 이용하는 것을 예로 들어보겠습니다. 설명하기에 앞서서 *constant* 파라메터라는 개념에 대해서 알아보겠습니다. 이 파라메터들은 back propagation이 호출되었을 때 사용되지는 않습니다. 추상적으로 들릴 수 있지만, 실제 일어나는 일이 그렇습니다. 어떤 함수가 있다고 가정합니다.

$$f(\mathbf{x},\mathbf{w}) = 3 \cdot \mathbf{w}^\top \mathbf{x}.$$

In this case 3 is a constant parameter. We could change 3 to something else, say $c$ via

이 경우, 3이 constant 파라메터입니다. 우리는 3을 다른 값, 예를 들어  $c$ 로 바꿔서 다음과 같이 표현할 수 있습니다.

$$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}.$$

Nothing has really changed, except that we can adjust the value of $c$. It is still a constant as far as $\mathbf{w}$ and $\mathbf{x}$ are concerned. However, since Gluon doesn't know about this beforehand, it's worth while to give it a hand (this makes the code go faster, too, since we're not sending the Gluon engine on a wild goose chase after a parameter that doesn't change). `get_constant` is the method that can be used to accomplish this. Let's see what this looks like in practice.

 $c$ 의 값을 조절할 수 있게된 것 이외에는 바뀐게 없습니다.  $\mathbf{w}$ 와 $\mathbf{x}$ 만을 생각해보면 여전히 상수입니다. 하지만, Gluon은 이것을 미리 알지 못하기 때문에, 도움을 주는 것이 필요합니다. 이렇게 하는 것은 Gluon이 변하지 않는 파라메터에 대해서는 신경쓰지 않도록 할 수 있기 때문에 코드가 더 빠르게 수행되게 해줍니다. `get_constant` 메소드을 이용하면 됩니다. 실제 어떻게 구현되는지 살펴보겠습니다.

```{.python .input  n=5}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # Random weight parameters created with the get_constant are not
        # iterated during training (i.e. constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # Use the constant parameters created, as well as the relu and dot
        # functions of NDArray
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.dense(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()
```

In this `FancyMLP` model, we used constant weight `Rand_weight` (note that it is not a model parameter), performed a matrix multiplication operation (`nd.dot<`), and reused the *same* `Dense` layer. Note that this is very different from using two dense layers with different sets of parameters. Instead, we used the same network twice. Quite often in deep networks one also says that the parameters are *tied* when one wants to express that multiple parts of a network share the same parameters. Let's see what happens if we construct it and feed data through it.

`FancyMLP` 모델에서 `rand_weight`라는 상수 weight을 정의했습니다. (이 변수는 모델 파라에터는 아니다라는 것을 알아두세요). 그리고, 행렬 곱하기 연산 (`nd.dot()`)을 수행하고, 같은 `Dense` 래이어를 재사용합니다. 서로 다른 파라메터 세트를 사용한 두 개의 dense 래이어를 사용했던것과 다른 형태로 구현되었음을 주목하세요. 우리는 대신, 같은 네트워크를 두 번 사용했습니다.  네트워크의 여러 부분이 같은 파라메터를 공유하는 경우 딥 네트워크에서 이 것을 파라메터가 서로 묶여 있다(tied)라고 말하기도 합니다. 이 클래스에 대한 인스턴스를 만들어서 데이터를 입력하면 어떤 일이 일어나는지 보겠습니다.

```{.python .input  n=6}
net = FancyMLP()
net.initialize()
net(x)
```

There's no reason why we couldn't mix and match these ways of build a network. Obviously the example below resembles more a chimera, or less charitably, a [Rube Goldberg Machine](https://en.wikipedia.org/wiki/Rube_Goldberg_machine). That said, it combines examples for building a block from individual blocks, which in turn, may be blocks themselves. Furthermore, we can even combine multiple strategies inside the same forward function. To demonstrate this, here's the network.

네트워크를 만들 때 이런 방법을 섞어서 사용하지 않을 이유가 없습니다. 아래 예제를 보면 어쩌면 키메라와 닮아 보일 수도 있고 조금 다르게 말하면, [Rube Goldberg Machine](https://en.wikipedia.org/wiki/Rube_Goldberg_machine) 와 비슷하다고 할 수도 있습니다. 즉, 개별적인 블럭을 합쳐서 블럭을 만들고 이렇게 만들어진 블럭이 다시 블럭으로 사용될 수 있는 것을 예제를 다음과 같이 만들어 볼 수 있습니다. 더 나아가서는 같은 forward 함수 안에서 여러 전략을 합치는 것도 가능합니다. 아래 코드가 그런 예입니다.

```{.python .input  n=7}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FancyMLP())

chimera.initialize()
chimera(x)
```

## Compilation

The avid reader is probably starting to worry about the efficiency of this. After all, we have lots of dictionary lookups, code execution, and lots of other Pythonic things going on in what is supposed to be a high performance deep learning library. The problems of Python's [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well known. In the context of deep learning it means that we have a super fast GPU (or multiple of them) which might have to wait until a puny single CPU core running Python gets a chance to tell it what to do next. This is clearly awful and there are many ways around it. The best way to speed up Python is by avoiding it altogether.

여러분이 관심이 많다면 이런 접근 방법에 대한 효율에 대한 의심을 할 것입니다. 결국에는 많은 dictionary 참조, 코드 수행과 다른 Python 코드들 수행하면서 성능이 높은 딥러닝 라이브러리를 만들어야 합니다. Python의 Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock) 은 아주 잘 알려진 문제로, 아주 성능이 좋은 GPU를 가지고 있을지라도 단일 CPU 코어에서 수행되는 Python 프로그램이 다음에 무엇을 해야할지를 알려주지 기달려야하기 때문에 딥러닝 환경에서 성능에 안좋은 영향을 미칩니다. 당연하게도 아주 나쁜 상황이지만, 이를 우회하는 여러 방법들이 존재합니다. Python 속도를 향상시키는 방법은 이 모든것을 모두 제거하는 것이 최선입니다.

Gluon does this by allowing for [Hybridization](../chapter_computational-performance/hybridize.md). In it, the Python interpreter executes the block the first time it's invoked. The Gluon runtime records what is happening and the next time around it short circuits any calls to Python. This can accelerate things considerably in some cases but care needs to be taken with control flow. We suggest that the interested reader skip forward to the section covering hybridization and compilation after finishing the current chapter.

Gluon은  [Hybridization](../chapter_computational-performance/hybridize.md) 기능을 통해서 해결하고 있습니다. Python 코드 블럭이 처음 수행되면 Gluon 런터임은 무엇이 수행되었는지를 기록하고, **이후에 수행될 때는 Python을 호출하지 않고 빠른 코드를 수행합니다. ** 이 방법은 속도를 상당히 빠르게 해주지만, 제어 흐름을 다루는데 주의를 기울여야 합니다. Hybridization과 compilation에 대해서 더 관심이 있다면 이 장을 마치고, 해당 내용이 있는 절을 읽어보세요.


## Summary

* Layers are blocks
* Many layers can be a block
* Many blocks can be a block
* Code can be a block
* Blocks take are of a lot of housekeeping, such as parameter initialization, backprop and related issues.
* Sequential concatenations of layers and blocks are handled by the eponymous `Sequential` block.
* 래이어들은 블럭입니다.
* 많은 래이어들이 하나의 블럭이 될 수 있습니다.
* 많은 블럭들이 하나의 블럭이 될 수 있습니다.
* 코드도 블럭이 될 수 있습니다.
* 블럭은 파라메터 초기화, back propagation, 또는 관련된 일을 대신 처리해줍니다.
* 래이어들과 블럭들을 순차적으로 연결하는 것은 `Sequential` 블럭에 의해서 처리됩니다.

## Problems

1. What kind of error message will you get when calling an `__init__` method whose parent class not in the `__init__` function of the parent class?
1. What kinds of problems will occur if you remove the `asscalar` function in the `FancyMLP` class?
1. What kinds of problems will occur if you change `self.net` defined by the Sequential instance in the `NestMLP` class to `self.net = [nn.Dense(64, activation='relu'), nn. Dense(32, activation='relu')]`?
1. Implement a block that takes two blocks as an argument, say `net1` and `net2` and returns the concatenated output of both networks in the forward pass (this is also called a parallel block).
1. Assume that you want to concatenate multiple instances of the same network. Implement a factory function that generates multiple instances of the same block and build a larger network from it.
1. `__init__` 메소드를 호출하면 어떤 오류 메시지가 나오나요? 상위 클래스의 `__init__` 메소드 안에 없는 상위 클래스는 무엇이니가요?
1. `FancyMLP` 클래스에서 `asscalar` 함수를 삭제하면 어떤 문제가 발생하나요? 
1. `NestMLP` 클래스에서 Sequential 클래스의 인스턴스로 정의된 `self.net` 을 `self.net = [nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu')]` 로 바꾸면 어떤 문제가 발생하나요?
1. 두 블럭 (`net1` 과 `net2`)를 인자로 받아서 forward pass의 두 네트워크의 결과를 연결해서 반환하는 블럭을 작성해보세요. (이는 parallel 블럭이라고 합니다)
1. 같은 네트워크의 여러 인스턴스를 연결하고자 가정합니다. 같은 블럭의 여러 인스턴스를 생성하는 factory 함수를 작성하고, 이를 사용해서 더 큰 네트워크를 만들어 보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2325)

![](../img/qr_model-construction.svg)
