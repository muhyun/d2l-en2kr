# Forward Propagation, Back Propagation, and Computational Graphs

In the previous sections we used a mini-batch stochastic gradient descent optimization algorithm to train the model. During the implementation of the algorithm, we only calculated the forward propagation of the model, which is to say, we calculated the model output for the input, then called the auto-generated `backward` function to then finally calculate the gradient through the `autograd` module. The automatic gradient calculation, when based on back-propagation, significantly simplifies the implementation of the deep learning model training algorithm. In this section we will use both mathematical and computational graphs to describe forward and back propagation. More specifically, we will explain forward and back propagation through a sample model with a single hidden layer perceptron with $\ell_2$ norm regularization. This section will help understand a bit better what goes on behind the scenes when we invoke a deep network.

앞에서 우리는 모델을 학습 시키는 방법으로 mini-batch stochastic gradient descent 최적화 알고리즘을 사용했습니다. 이를 구현할 때, 우리는 모델의 forward propagation을 계산하면서 입력에 대한 모델의 결과만을 계산했습니다. 그리고, 자동으로 생성된 `backward` 함수를 호출함으로  `autograd` 을 이용해서 gradient를 계산합니다. back-propagation을 이용하는 경우 자동으로 gradient를 계산하는 함수를 이용함으로 딥러닝 학습 알고리즘 구현이 굉징히 간단해졌습니다. 이 절에서는 forward propagation과 backward propagation을 수학적이고 연산적인 그래프를 사용해서 설명하겠습니다. 더 정확하게는 한개의 hidden 래이어를 갖는 multilayer perceptron에 $\ell_2$ norm regularization을 적용한 간단한 모델을 이용해서 forwad prop와 backward prop을 설명합니다. 이 절은딥 러닝을 수행할 때 어떤 일이 일어나고 있는지에 대해서 더 잘 이해할 수 있도록 해줄 것입니다.

## Forward Propagation

Forward propagation refers to the calculation and storage of intermediate variables (including outputs) for the neural network within the models in the order from input layer to output layer. In the following we work in detail through the example of a deep network with one hidden layer step by step. This is a bit tedious but it will serve us well when discussing what really goes on when we call `backward`.

Forward propagation은 뉴럴 네트워크 모델의 input 래이어부터 ouput 래이어까지 순서대로 변수들을 계산하고 저장하는 것을 의미합니다. 지금부터 한개의 hidden 래이어를 갖는 딥 네트워크를 예로 들어 단계별로 어떻게 계산되는지 설명해겠습니다. 다소 지루할 수 있지만, `backward` 를 호출했을 때, 어떤 일이 일어나는지 논의할 때 도움이 될 것입니다.

For the sake of simplicity, let’s assume that the input example is $\mathbf{x}\in \mathbb{R}^d$ and there is no bias term. Here the intermediate variable is:

간단하게 하기 위해서, 입력은 $d$ 차원의 실수 공간  $\mathbf{x}\in \mathbb{R}^d$ 으로 부터 선택되고, bias 항목은 생략하겠습니다. 중간 변수는 다음과 같이 정의됩니다.

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x}$$

$\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ is the weight parameter of the hidden layer. After entering the intermediate variable $\mathbf{z}\in \mathbb{R}^h$ into the activation function $\phi$ operated by the basic elements, we will obtain a hidden layer variable with the vector length of $h$,

$\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ 은 hidden 래이어의 weight 파라매터입니다. 중간 변수 $\mathbf{z}\in \mathbb{R}^h$ 를 activation 함수  $\phi$ 에 입력해서 백터 길이가  $h$ 인 hidden 래이어 변수를 얻습니다.

$$\mathbf{h}= \phi (\mathbf{z}).$$

The hidden variable $\mathbf{h}$ is also an intermediate variable. Assuming the parameters of the output layer only possess a weight of $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$, we can obtain an output layer variable with a vector length of $q$,

Hidden 변수 $\mathbf{h}$ 도 중간 변수입니다. Output 래이어가 weight $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ 만을 사용한다고 가정하면, 백터 길이가 $q$ 인 output 래이어의 변수를 다음과 같이 계산할 수 있습니다.

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Assuming the loss function is $l$ and the example label is $y$, we can then calculate the loss term for a single data example,

Loss 함수를 $l$ 이라고 하고, 샘플 label을 $y$ 라고 가정하면, 하나의 데이터 샘플에 대한 loss 값을 다음과 같이 계산할 수 있습니다.

$$L = l(\mathbf{o}, y).$$

According to the definition of $\ell_2$ norm regularization, given the hyper-parameter $\lambda$, the regularization term is

 $\ell_2$ norm regularization의 정의에 따라서, hyper-parameter $\lambda$ 가 주어졌을 때, 정규화 (regularization) 항목은 다음과 같습니다.

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$

where the Frobenius norm of the matrix is equivalent to the calculation of the $L_2$ norm after flattening the matrix to a vector. Finally, the model's regularized loss on a given data example is

여기서 행렬의 Frobenius norm은 행렬을 백터로 바꾼 후 계산하는 $L_2$ norm 과 같습니다. 마지막으로, 한개의 데이터 샘플에 대한 모델의 regularized loss 값을 계산합니다.

$$J = L + s.$$

We refer to $J$ as the objective function of a given data example and refer to it as the ‘objective function’ in the following discussion.

$J$ 를 주어진 데이터 샘플에 대한 objective 함수라고 하며, 앞으로 이를 'objective function'이라고 하겠습니다.


## Computational Graph of Forward Propagation

Plotting computational graphs helps us visualize the dependencies of operators and variables within the calculation. The figure below contains the graph associated with the simple network described above. The lower left corner signifies the input and the upper right corner the output. Notice that the direction of the arrows (which illustrate data flow) are primarily rightward and upward.

연산 그래프를 도식화하면 연산에 포함된 연산자와 변수들 사이의 관계를 시작화하는데 도움이 됩니다. 아래 그림은 위에서 정의한 간단한 네트워크의 그래프입니다. 왼쪽 아래는 입력이고, 오른쪽 위는 출력입니다. 데이터의 흐름을 표시하는 화살표의 방향이 오른쪽과 위로 향해있습니다.

![Compute Graph](../img/forward.svg)


## Back Propagation

Back propagation refers to the method of calculating the gradient of neural network parameters. In general, back propagation calculates and stores the intermediate variables of an objective function related to each layer of the neural network and the gradient of the parameters in the order of the output layer to the input layer according to the ‘chain rule’ in calculus. Assume that we have functions $\mathsf{Y}=f(\mathsf{X})$ and $\mathsf{Z}=g(\mathsf{Y}) = g \circ f(\mathsf{X})$, in which the input and the output $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ are tensors of arbitrary shapes. By using the chain rule we can compute the derivative of $\mathsf{Z}$ wrt. $\mathsf{X}$ via

Back propagation은 뉴럴 네트워크의 파라메터들에 대한 gradient를 계산하는 방법을 의미합니다. 일반적으로는 back propagation은 뉴럴 네트워크의 각 래이어와 관련된 objective 함수의 중간 변수들과 파라매터들의 gradient를 output 래이어에서 input 래이어 순으로 계산하고 저장합니다. 이는 미적분의 'chain rule'을 따르기 때문입니다. 임의의 모양을 갖는 입력과 출력 tensor $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ 들을 이용해서 함수 $\mathsf{Y}=f(\mathsf{X})$  와 $\mathsf{Z}=g(\mathsf{Y}) = g \circ f(\mathsf{X})$ 를 정의했다고 가정하고, Chain rule을 사용하면,  $\mathsf{X}$ 에 대한  $\mathsf{Z}$ 의 미분은 다음과 같이 정의됩니다.

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Here we use the $\text{prod}$ operator to multiply its arguments after the necessary operations, such as transposition and swapping input positions have been carried out. For vectors this is straightforward: it is simply matrix-matrix multiplication and for higher dimensional tensors we use the appropriate counterpart. The operator $\text{prod}$ hides all the notation overhead.

여기서 $\text{prod}$ 연산은 transposotion이나 입력 위치 변경과 같이 필요한 연산을 수항한 후 곱을 수행하는 것을 의미합니다. 백터의 경우에는 이것은 직관적입니다. 단순히 행렬-행렬 곱샘이고, 고차원의 텐서의 경우에는 새로 대응하는 원소들 간에 연산을 수행합니다. $\text{prod}$ 연산자는 이 모든 복잡한 개념을 감춰주는 역할을 합니다.

The parameters of the simple network with one hidden layer are $\mathbf{W}^{(1)}$ and $\mathbf{W}^{(2)}$. The objective of back propagation is to calculate the gradients $\partial J/\partial \mathbf{W}^{(1)}$ and $\partial J/\partial \mathbf{W}^{(2)}$. To accompish this we will apply the chain rule and calculate in turn the gradient of each intermediate variable and parameter. The order of calculations are reversed relative to those performed in forward propagation, since we need to start with the outcome of the compute graph and work our way towards the parameters. The first step is to calculate the gradients of the objective function $J=L+s$ with respect to the loss term $L$ and the regularization term $s$.

하나의 hidden 래이어를 갖는 간단한 네트워크의 파라매터는 $\mathbf{W}^{(1)}$ 와 $\mathbf{W}^{(2)}$ 이고, back propagation은 미분값 $\partial J/\partial \mathbf{W}^{(1)}$ 와 $\partial J/\partial \mathbf{W}^{(2)}$ 를 계산하는 것입니다. 이를 위해서 우리는 chain rule을 적용해서 각 중간 변수와 파라매터에 대한 gradient를 계산합니다. 연산 그래프의 결과로부터 시작해서 파라매터들에 대한 gradient를 계산해야하기 때문에, forward propagation과는 반대 방향으로 연산을 수행합니다. 첫번째 단계는 loss 항목 $L$ 과 regularization 항목 $s$ 에 대해서 objective 함수 $J=L+s$ 의 gradient를 계산하는 것입니다.

$$\frac{\partial J}{\partial L} = 1 \text{ and } \frac{\partial J}{\partial s} = 1$$

Next we compute the gradient of the objective function with respect to variable of the output layer $\mathbf{o}$ according to the chain rule.

그 다음, output layer $o$ 의 변수들에 대한 objective 함수의 gradient를 chain rule을 적용해서 구합니다.
$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q
$$

Next we calculate the gradients of the regularization term with respect to both parameters.

이제 두 파라메터에 대해서 regularization 항목의 gradient를 계산합니다.

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\text{ and }
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}$$

Now we are able calculate the gradient $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ of the model parameters closest to the output layer. Using the chain rule yields:

이제 우리는 output 래이어와 가장 가까운 모델 파라메터들에 대해서 objective 함수의 gradient $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ 를 계산할 수 있습니다. Chain rule을 적용하면 다음과 같이 계산됩니다.
$$
\frac{\partial J}{\partial \mathbf{W}^{(2)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)
= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}
$$

To obtain the gradient with respect to $\mathbf{W}^{(1)}$ we need to continue back propagation along the output layer to the hidden layer. The gradient $\partial J/\partial \mathbf{h}\in \mathbb{R}^h$ of the hidden layer variable is

 $\mathbf{W}^{(1)}$ 에 대한 gradient를 계산하기 위해서, output 래이어로부터 hidden 래이어까지 back propagation을 계속 해야합니다. Hidden 래이어 변수에 대한 gradient $\partial J/\partial \mathbf{h}\in \mathbb{R}^h$ 는 다음과 같습니다.
$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Since the activation function $\phi$ applies element-wise, calculating the gradient $\partial J/\partial \mathbf{z}\in \mathbb{R}^h$ of the intermediate variable $\mathbf{z}$ requires the use of the element-wise multiplication operator. We denote it by $\odot$.

Activation 함수  $\phi$ 는 각 원소별로 적용되기 때문에, 중간 변수 $\mathbf{z}$ 에 대한 gradient $\partial J/\partial \mathbf{z}\in \mathbb{R}^h$ 를 계산하기 위해서는 element-wise multiplication 연산자를 사용해야합니다. 우리는 이 연산을 $\odot$ 로 표현하겠습니다.
$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Finally, we can obtain the gradient $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ of the model parameters closest to the input layer. According to the chain rule, we get

마지막으로, input 래이어와 가장 가까운 모델 파라메터에 대한 gradient  $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ 를 chain rule을 적용해서 다음과 같이 계산합니다.
$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$

## Training a Model

When training networks, forward and backward propagation depend on each other. In particular, for forward propagation we traverse the compute graph in the direction of dependencies and compute all the variables on its path. These are then used for backpropagation where the compute order on the graph is reversed. One of the consequences is that we need to retain the intermediate values until backpropagation is complete. This is also one of the reasons why backpropagation requires significantly more memory than plain 'inference' - we end up computing tensors as gradients and need to retain all the intermediate variables to invoke the chain rule. Another reason is that we typically train with minibatches containing more than one variable, thus more intermediate activations need to be stored.

네트워크를 학습시킬 때, forward propagation과 backward propagation은 서로 의존하는 관계입니다. 특히 forward propagation은 연관되는 관계를 따라서 그래프를 계산하고, 그 경로의 모든 변수를 계산합니다. 이것들은 연산이 반대 방향인 back propagation에서 다시 사용됩니다. 그 결과 중에 하나로 back propagation을 완료할 때까지 중간 값들을 모두 가지고 있어야하는 것이 있습니다. 이것이 backpropagation이 단순 예측을 수행할 때보다 훨씬 더 많은 메모리를 사용하는 이유들 중에 하나입니다. 즉, chain rule을 적용하기 위해서 모든 중간 변수를 저장하고 있어야, gradient인 텐서(tensor)들을 계산할 수 있습니다. 메모리를 더 많이 사용하는 다른 이유는 모델을 학습 시킬 때 미니 배치 형태로 하기 때문에, 더 많은 중간 activation들을 저장해야하는 것이 있습니다.

## Summary

* Forward propagation sequentially calculates and stores intermediate variables within the compute graph defined by the neural network. It proceeds from input to output layer.
* Back propagation sequentially calculates and stores the gradients of intermediate variables and parameters within the neural network in the reversed order.
* When training deep learning models, forward propagation and back propagation are interdependent.
* Training requires significantly more memory and storage.
* Forwards propagation은 뉴럴 네트워크의 그래프를 계산하기 위해서 중간 변수들을 순서대로 계산하고 저장합니다. 즉, input 래이어부터 시작해서 output 래이어까지 처리합니다.
* Back propagation은 중간 변수와 파라매터에 대한 gradient를 반대 방향으로 계산하고 저장합니다.
* Deep learning 모델을 학습시킬 때,  forward propagation과 back propagation은 상호 의존적입니다.
* 학습은 상당히 많은 메모리와 저장 공간을 요구합니다.


## Problems

1. Assume that the inputs $\mathbf{x}$ are matrices. What is the dimensionality of the gradients?
1. Add a bias to the hidden layer of the model described in this chapter.
    * Draw the corresponding compute graph.
    * Derive the forward and backward propagation equations.
1. Compute the memory footprint for training and inference in model described in the current chapter.
1. Assume that you want to compute *second* derivatives. What happens to the compute graph? Is this a good idea?
1. Assume that the compute graph is too large for your GPU.
    * Can you partition it over more than one GPU?
    * What are the advantages and disadvantages over training on a smaller minibatch?
1. 입력  $\mathbf{x}$ 가 행렬이라고 가정하면, gradient의 차원이 어떻게 되나요?
1. 이 절에서 설명한 모델의 hidden layer에 bias를 추가하고,
    - 연산 그래프를 그려보세요
    - forward propagation과 backward propagation 공식을 유도해보세요.
1. 이 절에 사용한 모델에 대해서 학습과 예측에 사용되는 메모리 양을 계산해보세요.
1. 2차 미분을 계산해야한다고 가정합니다. 그래프 연산에 어떤일이 생길까요? 좋은 아이디어인가요?
1. 연산 그래프가 사용 중인 GPU에 비해서 너무 크다고 가정합니다.
    - 한개 이상의 GPU로 나눌 수 있나요?
    - 작은 미니배치로 학습을 할 경우 장점과 단점이 무엇인가요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2344)

![](../img/qr_backprop.svg)
