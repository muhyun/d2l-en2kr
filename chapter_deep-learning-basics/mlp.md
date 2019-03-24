# Multilayer Perceptron

In the previous chapters we showed how you could implement multiclass logistic regression (also called softmax regression) for classifying images of clothing into the 10 possible categories. This is where things start to get fun. We understand how to wrangle data, coerce our outputs into a valid probability distribution (via `softmax`), how to apply an appropriate loss function, and how to optimize over our parameters. Now that we’ve covered these preliminaries, we can extend our toolbox to include deep neural networks.

이전 절들에서 옷 이미지를 10개의 카테고리 중에 어디에 속하는지를 예측하는 multiclass logistic regression (또는 softmax regression)을 구현해봤습니다. 여기서부터 재미있는 것들이 시작됩니다. 데이터를 다루고, 출력값을 유효한 확률 분포로 바꾸고, 적합한 loss 함수를 적용하고, 파라메터를 최적화하는 방법에 대해서 알아봤습니다. 기본적인 것들을 익혔으니, 이제 deep neural network를 포함하도록 우리의 도구 상자를 확장해보겠습니다.

## Hidden Layers

Recall that before, we mapped our inputs directly onto our outputs through a single linear transformation via

이전의 것을 기억해보면, 단일 선형 변환 (single linear transformation)을 통해서 입력들을 바로 출력으로 매핑을 했고, 이는 아래과 같이 표현할 수 있습니다.
$$
\hat{\mathbf{o}} = \mathrm{softmax}(\mathbf{W} \mathbf{x} + \mathbf{b})
$$

![Single layer perceptron with 5 output units.](../img/singlelayer.svg)

If our labels really were related to our input data by an approximately linear function, then this approach might be adequate. But linearity is a *strong assumption*. Linearity means that given an output of interest, for each input, increasing the value of the input should either drive the value of the output up or drive it down, irrespective of the value of the other inputs.

만약 라벨 (label)들이 대략적인 선형 함수로 입력 데이터와 연관을 지을 수 있다면, 이 방법은 적절할 수 있습니다. 하지만, 이 선형성은 *너무 강한 가정*입니다. 선형성은 각 입력에 대해서, 입력값을 증가하면 다른 입력값과는 상관없이 결과값이 커지거나 작아지는 것을 의미합니다. 

### From one to many

Imagine the case of classifying cats and dogs based on black and white images. That’s like saying that for each pixel, increasing its value either increases the probability that it depicts a dog or decreases it. That’s not reasonable. After all, the world contains both black dogs and black cats, and both white dogs and white cats.

검정색이나 희색 이미지을 이용해서 강아지나 고양이를 분류하는 캐이스를 생각해봅시다. 각 픽셀의 값을 증가시키면 강아지라고 판별할 확률값을 높이거나 내려가는 경우를 생각해봅시다. 이는 합당하지 않습니다. 왜냐하면, 이렇게 된다면 결국 강아지는 모두 검정색이고 고양이는 모두 횐색이거나 그 반대라는 것을 의미하기 때문입니다.

Teasing out what is depicted in an image generally requires allowing more complex relationships between our inputs and outputs, considering the possibility that our pattern might be characterized by interactions among the many features. In these cases, linear models will have low accuracy. We can model a more general class of functions by incorporating one or more hidden layers. The easiest way to do this is to stack a bunch of layers of neurons on top of each other. Each layer feeds into the layer above it, until we generate an output. This architecture is commonly called a “multilayer perceptron”. With an MLP, we stack a bunch of layers on top of each other. Here's an example:

이미지에 무엇이 있는지 알아내기 위해서는 입력과 출력간의 매우 복잡한 관계와 이를 위해서는 패턴이 많은 feature들 사이의 관계를 통해서 특정 지어지는 가능성을 고려해야합니다. 이런 경우에는, 선형 모델의 정확도는 낮을 것입니다. 우리는 한개 이상의 hidden 래이어를 함께 사용해서 더 일반적인 함수들을 이용한 모델을 만들 수 있습니다. 이를 구현하는 가장 쉬운 방법은 각 래이어 위에 다른 래이어들을 쌓는 것입니다. 각 래이어의 결과는 그 위의 래이어의 입력으로 연결되는데, 이는 마지막 output 래이어까지 반복됩니다. 이런 아키텍쳐는 일반적으로 "multilayer perception"이라고 불립니다. 즉, MLP는 여러 래이어를 연속해서 쌓아올립니다. 예를 들면 다음과 같습니다.

![Multilayer perceptron with hidden layers. This example contains a hidden layer with 5 hidden units in it. ](../img/mlp.svg)

In the multilayer perceptron above, the number of inputs and outputs is 4 and 3 respectively, and the hidden layer in the middle contains 5 hidden units. Since the input layer does not involve any calculations, there are a total of 2 layers in the multilayer perceptron. The neurons in the hidden layer are fully connected to the inputs within the input layer. The neurons in the output layer and the neurons in the hidden layer are also fully connected. Therefore, both the hidden layer and the output layer in the multilayer perceptron are fully connected layers.

위 multilayer perception에서는 입력이 4개, 출력이 3개, 중간의 hidden 래이어는 5개의 hidden unit이 있습니다. 입력 래이어는 어떤 연산를 수행하지 않기 때문에, 이 multilayer perception은 총 2개의 래이어를 갖습니다. hidden 래이어의 neuron들은 입력 래이어의 입력들과 모두 연결되어 있습니다. output 래이어의 뉴런과 hidden 래이어의 뉴런들도 모두 연결되어 있습니다. 따라서, 이 multilayer perceptron의 hidden 래이어와 output 래이어 모두 fully connecter 래이어입니다.

### From linear to nonlinear

Let us write out what is happening mathematically in the picture above, e.g. for multiclass classification.

다중 클래스 분류의 경우 위 그림이 수학적으로 어떻게 정의되는지 보겠습니다.
$$
\begin{aligned}
    \mathbf{h} & = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 \\
    \mathbf{o} & = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2 \\
    \hat{\mathbf{y}} & = \mathrm{softmax}(\mathbf{o})
\end{aligned}
$$

The problem with the approach above is that we have gained nothing over a simple single layer perceptron since we can collapse out the hidden layer by an equivalently parametrized single layer perceptron using $\mathbf{W} = \mathbf{W}_2 \mathbf{W}_1$ and $\mathbf{b} = \mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2$.

위 방법의 문제점은 hidden 래이어를 다음과 같이 single layer perceptron 식으로 재구성할 수 있기 때문에, mutilayer가 아닌 단순한 single layer perceptron이라는 점입니다.

$$\mathbf{o} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2 = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1) \mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2) = \mathbf{W} \mathbf{x} + \mathbf{b}$$

To fix this we need another key ingredient - a nonlinearity $\sigma$ such as $\mathrm{max}(x,0)$ after each layer. Once we do this, it becomes impossible to merge layers. This yields

이를 해결하는 방법은 모든 래이어의 다음에  $\mathrm{max}(x,0)$ 와 같은 비선형 함수  $\sigma$ 를 추가하는 것입니다. 이렇게 하면, 래이어들을 합치는 것이 더이상 불가능해집니다. 즉,
$$
\begin{aligned}
    \mathbf{h} & = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \\
    \mathbf{o} & = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2 \\
    \hat{\mathbf{y}} & = \mathrm{softmax}(\mathbf{o})
\end{aligned}
$$

Clearly we could continue stacking such hidden layers, e.g. $\mathbf{h}_1 = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$ and $\mathbf{h}_2 = \sigma(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$ on top of each other to obtain a true multilayer perceptron.

이렇게 하면, 여러개 hidden 래이어들을 쌓는 것도 가능합니다. 즉,  $\mathbf{h}_1 = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$ 과 $\mathbf{h}_2 = \sigma(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$ 를 각각 연결해서 진짜 multilayer perceptron을 만들 수 있습니다. 

Multilayer perceptrons can account for complex interactions in the inputs because the hidden neurons depend on the values of each of the inputs. It’s easy to design a hidden node that that does arbitrary computation, such as, for instance, logical operations on its inputs. And it’s even widely known that multilayer perceptrons are universal approximators. That means that even for a single-hidden-layer neural network, with enough nodes, and the right set of weights, it could model any function at all! Actually learning that function is the hard part. And it turns out that we can approximate functions much more compactly if we use deeper (vs wider) neural networks. We’ll get more into the math in a subsequent chapter, but for now let’s actually build an MLP. In this example, we’ll implement a multilayer perceptron with two hidden layers and one output layer.

hidden 뉴런들이 각 입력의 값에 의존하고 있기 때문에 multilayer perceptron은 입력들 사이의 복잡한 상호 작용을 설명할 수 있습니다. 예를 들면, 입력들에 대한 논리 연산과 같이 임의의 연산을 수행하는 hidden node를 만드는 것도 쉽습니다. multilayer perceptron이 보편적인 approximator라는 것이 잘 알려져 있습니다. 이것의 의미는 다음과 같습니다. 한개의 hidden layer를 갖는 multilayer perceptron이라도, 충분이 많은 node와, 정확한 weight을 설정할 수 있다면 모든 함수에 대한 모델을 만들 수 있다는 것입니다. 사실 그 함수를 학습하는 것은 매우 어려운 부분입니다. 그리고, 더 깊은 (또는 더 넓은) 뉴럴 네트워크를 이용한다면 함수를 더욱 더 간결하게 추정할 수도 있습니다. 수학적인 부분은 이어질 장들에서 다루겠고, 이 장에서는 MLP를 실제로 만들어 보겠습니다. 아래 예제에서는 2개의 hidden 래이어와 1개의 output 래이어를 갖는 multilayer perceptron을 구현해보겠습니다.

### Vectorization and mini-batch

When given a mini-batch of samples we can use vectorization to gain better efficiency in implementation. In a nutshell, we replace vectors by matrices. As before, denote by $\mathbf{X}$ the matrix of inputs from a minibatch. Then an MLP with two hidden layers can be expressed as

샘플들이 미니 배치로 주어지는 경우에는 백터화를 통해서 구현의 효율성을 높일 수 있습니다. 요약하면, 백터를 행렬로 바꿀 것입니다.  $\mathbf{X}$ 는 미니 배치의 입력 행렬에 대한 표기입니다. 2개의 hidden 래이어를 갖는 MLP는 다음과 같이 표현됩니다.
$$
\begin{aligned}
    \mathbf{H}_1 & = \sigma(\mathbf{W}_1 \mathbf{X} + \mathbf{b}_1) \\
    \mathbf{H}_2 & = \sigma(\mathbf{W}_2 \mathbf{H}_1 + \mathbf{b}_2) \\
    \mathbf{O} & = \mathrm{softmax}(\mathbf{W}_3 \mathbf{H}_2 + \mathbf{b}_3)
\end{aligned}
$$

This is easy to implement and easy to optimize. With some abuse of notation we define the nonlinearity $\sigma$ to apply to its inputs on a row-wise fashion, i.e. one observation at a time, often one coordinate at a time. This is true for most activation functions (the [batch normalization](../chapter_convolutional-neural-networks/batch-norm.md) is a notable exception from that rule).

위 MLP는 구현을 쉽게 할 수 있고, 최적화도 쉽게할 수 있습니다. 표기법을 조금 남용해서, nonlinearlity  $\sigma$ 를 정의하고, 이를 row 단위로 입력에 적용하겠습니다. 즉, 한번에 하나의 관찰 또는 한번에 한 좌표씩 적용합니다. 실제 대부분 activation 함수는 이렇게 적용합니다.  ([batch normalization](../chapter_convolutional-neural-networks/batch-norm.md) 은 이 규칙에 대한 예외 중에 하나입니다.)

## Activation Functions

Let us look a bit more at examples of activation functions. After all, it is this alternation between linear and nonlinear terms that makes deep networks work. A rather popular choice, due to its simplicity of implementation and its efficacy is the ReLu function.

activation 함수의 예를 더 알아보겠습니다. 결국 deep network를 동작시키는 것은 linear 항목과 nonlinear 항목들을 서로 교차시키는 것입니다. 구현의 간단함과 좋은 효과로 유명한 ReLU 함수가 있습니다. 

### ReLU Function

The ReLU (rectified linear unit) function provides a very simple nonlinear transformation. Given the element $x$, the function is defined as

ReLU (rectified linear unit) 함수는 아주 간단한 nonlinear 변환입니다. 주어진 $x$ 에 대해서, ReLU 함수는 다음과 같이 정의됩니다.

$$\mathrm{ReLU}(x) = \max(x, 0).$$

It can be understood that the ReLU function retains only positive elements and discards negative elements. To get a better idea of what it looks like it helps to plot it. For convenience we define a plotting function `xyplot` to take care of the gruntwork.

ReLU 함수는 양수만 그대로 두고, 음수는 버리고 0으로 바꿔주는 역할을 합니다. 더 잘 이해하기 위해서 도표로 그려보겠습니다. 간단한 방법으로, 도표를 그리는 함수 `xyplot` 를 정의하겠습니다.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import autograd, nd

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
```

Then, we can plot the ReLU function using the `relu` function provided by NDArray. As you can see, the activation function is a two-stage linear function.

그리고, 이를 사용해서 ReLU 함수를 NDArray에서 제공하는 `relu` 함수를 이용해서 도식화합니다. 보이는 것처럼 activation function은 두 개의 linear 함수로 보입니다.

```{.python .input  n=2}
x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()
xyplot(x, y, 'relu')
```

Obviously, when the input is negative, the derivative of ReLU function is 0; when the input is positive, the derivative of ReLU function is 1. Note that the ReLU function is not differentiable when the input is 0. Instead, we pick its left-hand-side (LHS) derivative 0 at location 0. The derivative of the ReLU function is plotted below.

당연히 입력이 음수일 경우 ReLU 함수의 미분 값은 0이고, 입력이 양수면  ReLU 함수의 미분 값이 1이됩니다. 하지만, 입력이 0일 경우에는 ReLU 함수는 미분이 불가능하기 때문에, 입력이 0일 때는 left-hand-side (LHS) 미분인 0으로 선택합니다. ReLU 함수의 미분은 다음과 같습니다.

```{.python .input  n=3}
y.backward()
xyplot(x, x.grad, 'grad of relu')
```

Note that there are many variants to the ReLU function, such as the parameterized ReLU (pReLU) of [He et al., 2015](https://arxiv.org/abs/1502.01852). Effectively it adds a linear term to the ReLU, so some information still gets through, even when the argument is negative.

ReLU 함수는 다양한 변형이 있는데, 예를 들면 [He et al., 2015](https://arxiv.org/abs/1502.01852). parameterized ReLu (pReLU)가 있습니다. 이는 ReLU 선형 항목을 추가해서, 입력이 음수일 경우에도 정보가 전달될 수 있도록 만들고 있습니다.

$$\mathrm{pReLU}(x) = \max(0, x) - \alpha x$$

The reason for using the ReLU is that its derivatives are particularly well behaved - either they vanish or they just let the argument through. This makes optimization better behaved and it reduces the issue of the vanishing gradient problem (more on this later).

ReLU를 사용하는 이유는 값이 사라지거나 그대로 전달하게 하는 식으로 미분이 아주 잘 작동하기 때문입니다. 이런 특징으로 최적화가 더 잘 되고, (나중에 설명할) vanishing gradient 문제를 줄여줍니다.

### Sigmoid Function

The Sigmoid function can transform the value of an element in $\mathbb{R}$ to the interval $(0,1)$.

sigmoid 함수는 실수 값을 (0,1) 사이의 값으로 변환해줍니다.

$$\mathrm{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

The Sigmoid function was commonly used in early neural networks, but is currently being replaced by the simpler ReLU function. In the "Recurrent Neural Network" chapter, we will describe how to utilize the function’s ability to control the flow of information in a neural network thanks to its capacity to transform the value range between 0 and 1. The derivative of the Sigmoid function is plotted below. When the input is close to 0, the Sigmoid function approaches a linear transformation.

sigmoid 함수는 이전의 뉴럴 네트워크에서 일반적으로 사용되었으나, 현재는 더 간단한 ReLU 함수로 대체되었습니다. "Recurrent Neural Network"장에서는 이 함수가 0과 1사이의 값으로 변환해주는 특징을 이용해서 뉴럴 네트워크에서 정보의 전달을 제어하는데 어떻게 사용하는지 보겠습니다. Sigmoid 함수의 미분은 아래 그림과 같습니다. 입력이 0에 가까워지면, Sigmoid 함수는 선형 변환에 가까워집니다.

```{.python .input  n=4}
with autograd.record():
    y = x.sigmoid()
xyplot(x, y, 'sigmoid')
```

The derivative of Sigmoid function is as follows:

Sigmoid 함수의 미분은 아래와 같습니다.

$$\frac{d}{dx} \mathrm{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \mathrm{sigmoid}(x)\left(1-\mathrm{sigmoid}(x)\right).$$

The derivative of Sigmoid function is plotted below. When the input is 0, the derivative of the Sigmoid function reaches a maximum of 0.25; as the input deviates further from 0, the derivative of Sigmoid function approaches 0.

Sigmoid 함수의 미분은 아래와 같이 생겼습니다. 입력이 0이면, Sigmoid 함수의 미분의 최대값인 0.25가 됩니다. 입력값이 0에서 멀어지면, Sigmoid 함수의 미분값은 0으로 접근합니다.

```{.python .input  n=5}
y.backward()
xyplot(x, x.grad, 'grad of sigmoid')
```

### Tanh Function

The Tanh (Hyperbolic Tangent) function transforms the value of an element to the interval between -1 and 1:

Tanh (Hyperbolic Tangent) 함수는 값을 -1와 1사이 값으로 변환합니다.

$$\text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

We can then plot the Tanh function. As the input nears 0, the Tanh function approaches linear transformation. Although the shape of the function is similar to that of the Sigmoid function, the Tanh function is symmetric at the origin of the coordinate system.

Tanh 함수를 도식화를 아래와 같이 할 수 있습니다. 입력이 0에 가까워지면, Tanh 함수는 선형 변환에 가까워 집니다. 생긴 모양이 Sigmoid 함수와 비숫하지만, Tanh 함수는 좌표의 원점을 기준으로 대칭인 형태를 띕니다.

```{.python .input  n=6}
with autograd.record():
    y = x.tanh()
xyplot(x, y, 'tanh')
```

The derivative of the Tanh function is:

Tanh 함수의 미분은 다음과 같습니다.

$$\frac{d}{dx} \mathrm{tanh}(x) = 1 - \mathrm{tanh}^2(x).$$

The derivative of Tanh function is plotted below. As the input nears 0, the derivative of the Tanh function approaches a maximum of 1; as the input deviates away from 0, the derivative of the Tanh function approaches 0.

Tanh 함수의 미분은 아래와 같이 그려지는데, 입력이 0과 가까우면 Tanh 함수의 미분은 최대값이 1에 근접하게됩니다. 입력값이 0에서 멀어지면, Tanh 함수의 미분은 0에 접근합니다.

```{.python .input  n=7}
y.backward()
xyplot(x, x.grad, 'grad of tanh')
```

In summary, we have a range of nonlinearities and now know how to layer them to build quite powerful network architectures. As a side note, we have now pretty much reached the state of the art in deep learning, anno 1990. The main difference is that we have a powerful deep learning framework which lets us build models in a few lines of code where previously thousands of lines of C and Fortran would have been needed.

요약하면, 여러 종류의 nonlinearlity를 살펴봤고, 아주 강력한 네트워크 아키텍처를 만들기 위해서 어떻게 사용해야하는지도 알아봤습니다. 부수적으로 말하자면, 여기까지의 소개한 기법을 사용하면 1990년대의 최첨단의 deep learning을 만들 수 있습니다. 이전과 다른 점은 예전에는 C 또는 Fortran 언어를 이용해서 수천 줄의 코드로 만들어야 했던 모델을 지금은 강력한 deep learning framework가 있어서 몇줄의 코드로 만들 수 있다는 점입니다.

## Summary

* The multilayer perceptron adds one or multiple fully connected hidden layers between the output and input layers and transforms the output of the hidden layer via an activation function.
* Commonly used activation functions include the ReLU function, the Sigmoid function, and the Tanh function.
* Multilayer perceptron은 입력과 출력 래이어에 한개 이상의 fully connected hidden 래이어를 추가하고, 각 hidden 래이어의 결과에 activation 함수를 적용하는 것입니다.
* 일반적으로 사용되는 activation 함수는 ReLU 함수, Sigmoid 함수,  Tanh 함수가 있습니다.


## Problems

1. Compute the derivative of the Tanh and the pReLU activation function.
1. Show that a multilayer perceptron using only ReLU (or pReLU) constructs a continuous piecewise linear function.
1. Show that $\mathrm{tanh}(x) + 1 = 2 \mathrm{sigmoid}(2x)$.
1. Assume we have a multilayer perceptron *without* nonlinearities between the layers. In particular, assume that we have $d$ input dimensions, $d$ output dimensions and that one of the layers had only $d/2$ dimensions. Show that this network is less expressive (powerful) than a single layer perceptron.
1. Assume that we have a nonlinearity that applies to one minibatch at a time. What kinds of problems to you expect this to cause?
1. Tanh,  pReLU activation 함수의 미분을 구하시오.
1. ReLU (또는 pReLU) 만을 사용해서 만든 multlayer perceptron은 연속된 piecewise linear function임을 증명하세요.
1. $\mathrm{tanh}(x) + 1 = 2 \mathrm{sigmoid}(2x)$ 임을 증명하세요.
1. layer 사이에 nonlinearity 없이 만든 multilayer perceptron이 있다고 가정합니다.  $d$  입력 차원,  $d$ 출력 차원, 그리고 다른 layer는  $d/2$ 차원을 찾는다고 했을 때, 이 네트워크는 single layer perceptron 보다 강하지 않다는 것을 증명하세요.
1. 한번에 하나의 미니배치에 적용하는 nonlinearity가 있다고 가정합니다. 이렇게 해서 발생하는 문제가 무엇이 있을까요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2338)

![](../img/qr_mlp.svg)
