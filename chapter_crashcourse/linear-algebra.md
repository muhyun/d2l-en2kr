# Linear algebra

Now that you can store and manipulate data,
let's briefly review the subset of basic linear algebra
that you'll need to understand most of the models.
We'll introduce all the basic concepts,
the corresponding mathematical notation,
and their realization in code all in one place.
If you're already confident in your basic linear algebra,
feel free to skim or skip this chapter.

자 이제 데이터를 저장하고 조작하는 방법을 배웠으니, 모델에 대한 대부분을 이해하는데 필요한 기초적인 선형 대수 일부를 간단하게 살펴보겠습니다. 기초적인 개념, 관련된 수학 표기법, 그리고 코드로의 구현까지 모두 소개할 것입니다. 기본 선형 대수에 익숙하다면, 이 절은 빨리 읽거나 다음 절로 넘어가도 됩니다.

```{.python .input}
from mxnet import nd
```

## Scalars

If you never studied linear algebra or machine learning,
you're probably used to working with one number at a time.
And know how to do basic things like add them together or multiply them.
For example, in Palo Alto, the temperature is $52​$ degrees Fahrenheit.
Formally, we call these values $scalars​$.
If you wanted to convert this value to Celsius (using metric system's more sensible unit of temperature measurement),
you'd evaluate the expression $c = (f - 32) * 5/9​$ setting $f​$ to $52​$.
In this equation, each of the terms $32​$, $5​$, and $9​$ is a scalar value.
The placeholders $c​$ and $f​$ that we use are called variables
and they stand in for unknown scalar values.

선형대수나 머신러닝을 배워본 적이 없다면, 아마도 한번에 하나의 숫자를 다루는데 익숙할 것입니다. 예를 어 팔로 알토의 기온이 화씨 52도입니다. 공식 용어를 사용하면 이 값은 *스칼라(scalar)* 입니다. 이 값을 섭씨로 바꾸기를 원한다면,  ​$c = (f - 32) * 5/9$ 공식에  ​$f$ 값으로  ​$52$ 대입하면됩니다. 이 공식에서 각 항들  ​$32$, ​$5$, ​$9$ 은 스칼라 값입니다. 플래이스 홀더 ​$c$ 와 ​$f$ 를 변수라고 부르고, 아직 정해지지 않은 스칼라 값들을 위해 있습니다.

In mathematical notation, we represent scalars with ordinary lower cased letters ($x​$, $y​$, $z​$).
We also denote the space of all scalars as $\mathcal{R}​$.
For expedience, we're going to punt a bit on what precisely a space is,
but for now, remember that if you want to say that $x​$ is a scalar,
you can simply say $x \in \mathcal{R}​$.
The symbol $\in​$ can be pronounced "in" and just denotes membership in a set.

수학적인 표기법으로는 우리는 스칼라를 소문자($x$, $y$, $z$)로 표기 합니다. 또한 모든 스칼라에 대한 공간은  $\mathcal{R}$로 적습니다. 공간이 정확히 무엇인지를 알아보겠지만, 편의상 지금은 $x$ 가 스칼라라고 이야기하는 것은 $x \in \mathcal{R}$ 로 표현하기로 하겠습니다.

In MXNet, we work with scalars by creating NDArrays with just one element.
In this snippet, we instantiate two scalars and perform some familiar arithmetic operations with them, such as addition, multiplication, division and exponentiation.

MXNet에서 스칼라는 하나의 원소를 갔는 NDArray로 표현됩니다. 아래 코드에서는 두개의 스칼라를 생성하고, 친숙한 수치 연산 - 더하기, 빼기, 나누기, 그리고 제곱을 수행합니다.

```{.python .input}
x = nd.array([3.0])
y = nd.array([2.0])

print('x + y = ', x + y)
print('x * y = ', x * y)
print('x / y = ', x / y)
print('x ** y = ', nd.power(x,y))
```

We can convert any NDArray to a Python float by calling its `asscalar` method. Note that this is typically a bad idea. While you are doing this, NDArray has to stop doing anything else in order to hand the result and the process control back to Python. And unfortunately Python isn't very good at doing things in parallel. So avoid sprinkling this operation liberally throughout your code or your networks will take a long time to train.

`asscalar` 메소르를 이용하면 NDArray를 Python의 float 형으로 변환할 수 있습니다. 하지만 이렇게 하는 것은 좋은 아이디어가 아님을 알아두세요. 그 이유는 이를 수행하는 동안, 결과과 프로세스 제어를 Python에게 줘야하기 때문에 NDArray는 다른 것들을 모두 멈춰야합니다. 아쉽게도, Python은 병렬로 일을 처리하는데 좋지 못합니다. 이런 연산을 코드나 네트워크에서 수행한다면 학습하는데 오랜 시간이 걸릴 것입니다.

```{.python .input}
x.asscalar()
```

## Vectors

You can think of a vector as simply a list of numbers, for example ``[1.0,3.0,4.0,2.0]``.
Each of the numbers in the vector consists of a single scalar value.
We call these values the *entries* or *components* of the vector.
Often, we're interested in vectors whose values hold some real-world significance.
For example, if we're studying the risk that loans default,
we might associate each applicant with a vector
whose components correspond to their income,
length of employment, number of previous defaults, etc.
If we were studying the risk of heart attack in hospital patients,
we might represent each patient with a vector
whose components capture their most recent vital signs,
cholesterol levels, minutes of exercise per day, etc.
In math notation, we'll usually denote vectors as bold-faced,
lower-cased letters ($\mathbf{u}​$, $\mathbf{v}​$, $\mathbf{w})​$.
In MXNet, we work with vectors via 1D NDArrays with an arbitrary number of components.

백터를 ``[1.0,3.0,4.0,2.0]`` 처럼 숫자들의 리스트로 생각할 수 있습니다. 백터의 각 숫자는 하나의 스칼라 변수로 이뤄져있습니다. 이 숫자들을 우리는 백터의 *원소* 또는 *구성요소* 라고 부릅니다. 종종 우리는 실제 세상에서 중요한 값을 답은 백터들에 관심을 갖습니다. 예를 들어 채무 불이행 위험을 연구하고 있다면, 모든 지원자를 원소가 수입, 재직 기간, 이전의 불이행 횟수 등을 포함한 백터와 연관을 지을지도 모릅니다. 병원의 환자들이 심장 마비 위험을 연구하는 사람은, 환자들을 최근 바이탈 사인, 콜레스테롤 지수, 하루 운동 시간 등을 원소로 갖는 백터로 표한할 것입니다. 수학적인 표기 법을 이용할 때 백터는 굵은 글씨체로 소문자  ($\mathbf{u}$, $\mathbf{v}$, $\mathbf{w})$ 를 사용해서 표현합니다. MXNet에서는 임의의 숫자를 원소로 갖는 1D NDArray를 이용해서 백터를 다루게 됩니다.

```{.python .input}
x = nd.arange(4)
print('x = ', x)
```

We can refer to any element of a vector by using a subscript.
For example, we can refer to the $4​$th element of $\mathbf{u}​$ by $u_4​$.
Note that the element $u_4​$ is a scalar,
so we don't bold-face the font when referring to it.
In code, we access any element $i​$ by indexing into the ``NDArray``.

첨자를 사용해서 백터의 요소를 가르킬 수 있습니다. 즉, $\mathbf{u}$ 의 4번째 요소는  $u_4$ 로 표현합니다.  $u_4$ 는 스칼라이고, 따라서 굵은 글씨가 아닌 폰트로 표기하는 것을 유의하세요. 코드에는 `NDArray` 의  $i$ 번째  인덱스로 이를 지정합니다.

```{.python .input}
x[3]
```

## Length, dimensionality and shape

Let's revisit some concepts from the previous section. A vector is just an array of numbers. And just as every array has a length, so does every vector.
In math notation, if we want to say that a vector $\mathbf{x}​$ consists of $n​$ real-valued scalars,
we can express this as $\mathbf{x} \in \mathcal{R}^n​$.
The length of a vector is commonly called its $dimension​$.
As with an ordinary Python array, we can access the length of an NDArray
by calling Python's in-built ``len()`` function.

앞 절에서 소개한 개념 몇개를 다시 살펴보겠습니다. 백터는 숫자들의 배열입니다. 모든 배열은 길이를 갖듯이, 백터도 길이를 갖습니다. 백터 $\mathbf{x}$ 가 $n$ 개의 실수값을 갖는 스칼라들로 구성되어 있다면, 이는 수학적인 표현으로 $\mathbf{x} \in \mathcal{R}^n$ 로 적습니다. 백터의 길이는 일반적으로 차원($dimension$) 이라고 합니다. Python 배열 처럼, NDArray의 길이도 Python의 내장 함수 `len()` 를 통해서 얻을 수 있습니다.

We can also access a vector's length via its `.shape` attribute.
The shape is a tuple that lists the dimensionality of the NDArray along each of its axes.
Because a vector can only be indexed along one axis, its shape has just one element.

백터의 길이는 `.shape` 속성으로도 얻을 수 있습니다. shape은 NDArray 객체의 각 축에 대한 차원의 목록으로 표현됩니다. 백터는 축이 하나이기 때문에, 백터의 shape는 하나의 숫자로 표현됩니다.

```{.python .input}
x.shape
```

Note that the word dimension is overloaded and this tends to confuse people.
Some use the *dimensionality* of a vector to refer to its length (the number of components).
However some use the word *dimensionality* to refer to the number of axes that an array has.
In this sense, a scalar *would have* $0$ dimensions and a vector *would have* $1$ dimension.

차원(dimension)이라는 단어가 여러가지 의미로 사용되기 때문에, 헷깔릴 수 있습니다. 어떤 경우에는 백터의 *차원(dimensionality)* 를 백터의 길이 (원소들의 수)로 사용하기도 하고, 어떤 경우에는 배열의 축의 개수로 사용되기도 합니다. 후자의 경우에는 스칼라는 0 차원을 갖고, 백터는 1차원을 갖습니다.

**To avoid confusion, when we say *2D* array or *3D* array, we mean an array with 2 or 3 axes respectively. But if we say *$n$-dimensional* vector, we mean a vector of length $n$.**

**혼동을 줄이기 위해서, 우리는 2D 배열 또는 3D 배열이라고 말할 때, 축이 각각 2개 3개인 배열을 의미하도록 합니다. 하지만 막약 *n-*차원 백터라고 하는 경우에는, 길이가 *n* 인 백터를 의미합니다.**

```{.python .input}
a = 2
x = nd.array([1,2,3])
y = nd.array([10,20,30])
print(a * x)
print(a * x + y)
```

## Matrices

Just as vectors generalize scalars from order $0​$ to order $1​$,
matrices generalize vectors from $1D​$ to $2D​$.
Matrices, which we'll typically denote with capital letters ($A​$, $B​$, $C​$),
are represented in code as arrays with 2 axes.
Visually, we can draw a matrix as a table,
where each entry $a_{ij}​$ belongs to the $i​$-th row and $j​$-th column.

백터가 오더 0 인 스칼라를 오더 1로 일반화하는 것처럼, 행렬은 1$D$에서 2$D$ 로 백터를 일반화합니다. 일반적으로 대문자 ($A$, $B$, $C$)로 표현하는 형렬은 코드에서는 축이 2개인 배열로 표현합니다. 시각화한다면, 행렬은 원소 $a_{ij}$ 가 $i$-열, $j$-행에 속하는 표로 그릴 수 있습니다. 

$$A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1m} \\
 a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{pmatrix}​$$

We can create a matrix with $n$ rows and $m$ columns in MXNet
by specifying a shape with two components `(n,m)`
when calling any of our favorite functions for instantiating an `ndarray`
such as `ones`, or `zeros`.

MXNet에서는 $n$ 행, $m$ 열을 갖는 행렬을 만드는 방법은 두 요소를 갖는 `(n,m)`  shape을 이용해서 `ones` 또는 `zeros` 함수를 호출을 통해  `ndarray` 를 얻는 것입니다.

```{.python .input}
A = nd.arange(20).reshape((5,4))
print(A)
```

Matrices are useful data structures: they allow us to organize data that has different modalities of variation. For example, rows in our matrix might correspond to different patients, while columns might correspond to different attributes.

형렬은 유용한 자료 구조입니다. 행렬을 이용해서 서로 다른 양식의 변형을 갖는 데이터를 구성할 수 있습니다. 예를 들어보면, 행렬의 행들은 서로 다른 환자에 대한 정보를, 열은 서로 다른 속성에 대한 정보를 의미할 수 있습니다.

We can access the scalar elements $a_{ij}$ of a matrix $A$ by specifying the indices for the row ($i$) and column ($j$) respectively. Leaving them blank via a `:` takes all elements along the respective dimension (as seen in the previous section).

형렬 $A$ 의 스칼라 원소 $a_{ij}$ 을 지정하는 방법은 행($i$)과 열($j$)에 대한 인덱스를 지정하면 됩니다. `:` 를 사용해서 공백으로 두면, 해당 차원의 모든 원소를 의미합니다. (앞 절에서 봤던 방법입니다.)

We can transpose the matrix through `T`. That is, if $B = A^T$, then $b_{ij} = a_{ji}$ for any $i$ and $j$.

행렬을 전치하는 방법은 `T` 를 이용합니다. 전치 행렬은 만약 $B = A^T$ 이면, 모든 $i$ 과 $j$ 에 대해서 $b_{ij} = a_{ji}$ 인 행렬을 의미합니다.

```{.python .input}
print(A.T)
```

## Tensors

Just as vectors generalize scalars, and matrices generalize vectors, we can actually build data structures with even more axes. Tensors give us a generic way of discussing arrays with an arbitrary number of axes. Vectors, for example, are first-order tensors, and matrices are second-order tensors.

백터가 스칼라를 일반화하고, 행렬이 백터를 일반화하는 것처럼 더 많은 축을 갖는 자료 구조를 만들 수 있습니다. 텐서(tensor)는 임의의 개수의 축을 갖는 행렬을 표현하는 일반적인 방법을 제공합니다. 예를 들어 백터는 1차 오더(order) 텐서이고, 행렬을 2차 오더(order) 텐서입니다.

Using tensors will become more important when we start working with images, which arrive as 3D data structures, with axes corresponding to the height, width, and the three (RGB) color channels. But in this chapter, we're going to skip past and make sure you know the basics.

3D 자료 구조를 갖는 이미지를 다룰 때 텐서를 사용하는 것은 아주 중요하게 됩니다. 즉, 각 축이 높이, 넓이, 그리고 세가지 색(RGB) 채널을 의미합니다. **이 장에서는 기본적인 것을을 확실하게 아는 것을 목표로 하겠습니다.**

```{.python .input}
X = nd.arange(24).reshape((2, 3, 4))
print('X.shape =', X.shape)
print('X =', X)
```

## Basic properties of tensor arithmetic

Scalars, vectors, matrices, and tensors of any order have some nice properties that we'll often rely on.
For example, as you might have noticed from the definition of an element-wise operation,
given operands with the same shape,
the result of any element-wise operation is a tensor of that same shape.
Another convenient property is that for all tensors, multiplication by a scalar
produces a tensor of the same shape.
In math, given two tensors $X​$ and $Y​$ with the same shape,
$\alpha X + Y​$ has the same shape
(numerical mathematicians call this the AXPY operation).

스칼라, 백터, 행렬, 그리고 어떤 오더를 같는 텐서 우리가 자주 사용할 유용한 특성들을 가지고 있습니다. 원소에 따른 연산(element-wise operatio)의 정의의서 알 수 있듯이, 같은 shape 들에 대해서 연산을 수행하면, 원소에 따른 연산의 경과는 같은 shape을 갖는 텐서입니다. 또다른 유용한 특성은 모든 텐서에 대해서 스칼라를 곱하면 결과는 같은 shape의 텐서입니다. 수학적으로 표현하면, 같은 shape의 두 텐서 $X$와 $Y$가 있다면 $\alpha X + Y$ 는 같은 shape을 갖습니다. 

```{.python .input}
a = 2
x = nd.ones(3)
y = nd.zeros(3)
print(x.shape)
print(y.shape)
print((a * x).shape)
print((a * x + y).shape)
```

Shape is not the the only property preserved under addition and multiplication by a scalar. These operations also preserve membership in a vector space. But we'll postpone this discussion for the second half of this chapter because it's not critical to getting your first models up and running.

더하기와 스칼라 곱으로 보전되는 특성이 shape 뿐만은 아닙니다. 이 연산들은 백터 공간의 맴버쉽을 보존해줍니다. 하지만, 여러분의 첫번째 모델을 만들어서 수행하는데 중요하지 않기 때문에, 이장의 뒤에서 설명하겠습니다.

## Sums and means

The next more sophisticated thing we can do with arbitrary tensors
is to calculate the sum of their elements.
In mathematical notation, we express sums using the $\sum$ symbol.
To express the sum of the elements in a vector $\mathbf{u}$ of length $d$,
we can write $\sum_{i=1}^d u_i$. In code, we can just call ``nd.sum()``.

임의의 텐서들로 수행할 수 있는 조금더 복잡한 것은 각 요소의 합을 구하는 것입니다. 수학기호로는 합을  $\sum$ 로 표시합니다. 길이가 $d$ 인 백터 $\mathbf{u}$ 의 요소들의 합은  $\sum_{i=1}^d u_i$ 로 표현하고, 코드에서는 `nd.sum()` 만 호출하면 됩니다.

```{.python .input}
print(x)
print(nd.sum(x))
```

We can similarly express sums over the elements of tensors of arbitrary shape. For example, the sum of the elements of an $m \times n​$ matrix $A​$ could be written $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}​$.

임의의 shape을 갖는 텐서의 우너소들의 합을 비슷하게 표현할 수 있습니다. 예를 들어 $m \times n$ 행렬 $A$ 의 원소들의 합은 $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$ 이고, 코드로는 다음과 같습니다.

```{.python .input}
print(A)
print(nd.sum(A))
```

A related quantity is the *mean*, which is also called the *average*.
We calculate the mean by dividing the sum by the total number of elements.
With mathematical notation, we could write the average
over a vector $\mathbf{u}​$ as $\frac{1}{d} \sum_{i=1}^{d} u_i​$
and the average over a matrix $A​$ as  $\frac{1}{n \cdot m} \sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}​$.
In code, we could just call ``nd.mean()`` on tensors of arbitrary shape:

합과 관련된 것으로 *평균(mean)* 이 있습니다. (*average*라고도 합니다.) 평균은 합을 원소들의 개수로 나눠서 구합니다. 어떤 백터  $\mathbf{u}$ 의 평균을 수학 기호로 표현하면  $\frac{1}{d} \sum_{i=1}^{d} u_i$ 이고, 행렬  $A$ 에 대한 평균은 s  $\frac{1}{n \cdot m} \sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$ 이 됩니다. 코드로 구현하면, 임의의 shape을 갖는 텐서의 평균은 `nd.mean()` 을 호출해서 구합니다.

```{.python .input}
print(nd.mean(A))
print(nd.sum(A) / A.size)
```

## Dot products

So far, we've only performed element-wise operations, sums and averages. And if this was all we could do, linear algebra probably wouldn't deserve its own chapter. However, one of the most fundamental operations is the dot product. Given two vectors $\mathbf{u}​$ and $\mathbf{v}​$, the dot product $\mathbf{u}^T \mathbf{v}​$ is a sum over the products of the corresponding elements: $\mathbf{u}^T \mathbf{v} = \sum_{i=1}^{d} u_i \cdot v_i​$.

지금까지는 원소들 사이에 연산인 더하기와 평균에 대해서 살펴봤습니다. 이 연산들이 우리가 할 수 있는 전부라면, 선형대수를 별도의 절로 만들어서 설명할 필요가 없을 것입니다. 즉, 가장 기본적인 연산들 중에 하나로 점곱(dot product)가 있습니다. 두 백터, $\mathbf{u}$ 와 $\mathbf{v}$, 가 주어졌을 때, 점곱,  $\mathbf{u}^T \mathbf{v}$ ,은 요소들끼리 곱을 한 결과에 대한 합이됩니다. 즉, $\mathbf{u}^T \mathbf{v} = \sum_{i=1}^{d} u_i \cdot v_i$.

```{.python .input}
x = nd.arange(4)
y = nd.ones(4)
print(x, y, nd.dot(x, y))
```

Note that we can express the dot product of two vectors ``nd.dot(x, y)`` equivalently by performing an element-wise multiplication and then a sum:

두 백터의 점곱  ``nd.dot(x, y)`` , 은 원소들끼지의 곱을 수행한 후, 합을 구하는 것과 동일합니다.

```{.python .input}
nd.sum(x * y)
```

Dot products are useful in a wide range of contexts. For example, given a set of weights $\mathbf{w}$, the weighted sum of some values ${u}$ could be expressed as the dot product $\mathbf{u}^T \mathbf{w}$. When the weights are non-negative and sum to one $\left(\sum_{i=1}^{d} {w_i} = 1\right)$, the dot product expresses a *weighted average*. When two vectors each have length one (we'll discuss what *length* means below in the section on norms), dot products can also capture the cosine of the angle between them.

점곱은 다양한 경우에 유용하게 사용됩니다. 예를 들어, weight들의 집합 $\mathbf{w}$ 에 대해서, 어떤 값 $u$ 의 가중치를 적용한 합은 점곱인 $\mathbf{u}^T \mathbf{w}$으로 계산될 수 있습니다. weight들이 0 또는 양수이고, 합이 1  $\left(\sum_{i=1}^{d} {w_i} = 1\right)$ 인 경우, 행렬의 곱은 *가중치 평균(weighted average)* 를 나타냅니다. 길이가 1인 두 백터 (길이가 무엇인지는 아래에서 norm을 설명할때 다룹니다)가 있을 때, 점곱을 통해서 두 백터 사이의 코사인 각을 구할 수 있습니다.

## Matrix-vector products

Now that we know how to calculate dot products we can begin to understand matrix-vector products. Let's start off by visualizing a matrix $A​$ and a column vector $\mathbf{x}​$.

점곱을 어떻게 계산하는지 알아봤으니, 행렬-백터 곱을 알아볼 준비가 되었습니다. 우선 행렬 $A$ 와 열백터  $\mathbf{x}$ 를 시각적으로 표현하는 것으로 시작합니다.

$$A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1m} \\
 a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{pmatrix},\quad\mathbf{x}=\begin{pmatrix}
 x_{1}  \\
 x_{2} \\
\vdots\\
 x_{m}\\
\end{pmatrix} ​$$

We can visualize the matrix in terms of its row vectors

행렬을 다시 행백터 형태로 표현이 가능합니다.

$$A=
\begin{pmatrix}
\mathbf{a}^T_{1} \\
\mathbf{a}^T_{2} \\
\vdots \\
\mathbf{a}^T_n \\
\end{pmatrix},$$

where each $\mathbf{a}^T_{i} \in \mathbb{R}^{m}​$
is a row vector representing the $i​$-th row of the matrix $A​$.

여기서 각  $\mathbf{a}^T_{i} \in \mathbb{R}^{m}$ 는 행렬의 $i$ 번째 행을 표시하는 행 백터입니다.

Then the matrix vector product $\mathbf{y} = A\mathbf{x}$ is simply a column vector $\mathbf{y} \in \mathbb{R}^n$ where each entry $y_i$ is the dot product $\mathbf{a}^T_i \mathbf{x}$.

그러면 행렬-백터 곱 $\mathbf{y} = A\mathbf{x}$ 은 컬럼 백터  $\mathbf{y} \in \mathbb{R}^n$ 이며, 각 원소 $y_i$ 는 점곱 $\mathbf{a}^T_i \mathbf{x}$ 입니다.

$$A\mathbf{x}=
\begin{pmatrix}
\mathbf{a}^T_{1}  \\
\mathbf{a}^T_{2}  \\
 \vdots  \\
\mathbf{a}^T_n \\
\end{pmatrix}
\begin{pmatrix}
 x_{1}  \\
 x_{2} \\
\vdots\\
 x_{m}\\
\end{pmatrix}
= \begin{pmatrix}
 \mathbf{a}^T_{1} \mathbf{x}  \\
 \mathbf{a}^T_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^T_{n} \mathbf{x}\\
\end{pmatrix}$$

So you can think of multiplication by a matrix $A\in \mathbb{R}^{n \times m}$ as a transformation that projects vectors from $\mathbb{R}^{m}$ to $\mathbb{R}^{n}​$.

즉, 행렬 $A\in \mathbb{R}^{n \times m}$ 로 곱하는 것을 백터를 $\mathbb{R}^{m}$ 에서 $\mathbb{R}^{n}$로 사영시키는 변한으로도 생각할 수 있습니다.

These transformations turn out to be quite useful. For example, we can represent rotations as multiplications by a square matrix. As we'll see in subsequent chapters, we can also use matrix-vector products to describe the calculations of each layer in a neural network.

이런 변환은 아주 유용하게 쓰입니다. 예를 들면, 회전을 스퀘어 행렬의 곲으로 표현할 수 있습니다. 다음 절에서 보겠지만, 행렬-백터 곱을 뉴럴 네트워크의 각 레이어의 연산을 표현하는데도 사용합니다.

Expressing matrix-vector products in code with ``ndarray``, we use the same ``nd.dot()`` function as for dot products. When we call ``nd.dot(A, x)`` with a matrix ``A`` and a vector ``x``, MXNet knows to perform a matrix-vector product. Note that the column dimension of ``A`` must be the same as the dimension of ``x``.

`ndarray'` 를 이용해서 행렬-백터의 곱을 계산할 때는 점곱에서 사용했던 `nd.dot()` 함수를 동일하게 사용합니다. 행렬 `A` 와 백터 `x` 를 이용해서 `nd.dot(A,x)` 를 호출하면, MXNet은 행렬-백터 곱을 수행해야하다는 것을 압니다. `A` 의 열의 개수와 `x` 의 차원이 같아야한다는 점을 유의하세요.

```{.python .input}
nd.dot(A, x)
```

## Matrix-matrix multiplication

If you've gotten the hang of dot products and matrix-vector multiplication, then matrix-matrix multiplications should be pretty straightforward.

점곱과 행렬-백터 곱을 잘 이해했다면, 행렬-행렬 곱은 아주 간단할 것입니다.

Say we have two matrices, $A \in \mathbb{R}^{n \times k}$ and $B \in \mathbb{R}^{k \times m}$:

행렬 두개, $A \in \mathbb{R}^{n \times k}$,  $B \in \mathbb{R}^{k \times m}$, 가 있다고 하겠습니다.

$$A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{pmatrix},\quad
B=\begin{pmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{pmatrix}$$

To produce the matrix product $C = AB$, it's easiest to think of $A$ in terms of its row vectors and $B$ in terms of its column vectors:

행렬의 곱  $C = AB$ 를 계산하기 위해서,  $A$ 를 행 백터들로,  $B$ 를 열 백터들로로 생각하면 쉽습니다.

$$A=
\begin{pmatrix}
\mathbf{a}^T_{1} \\
\mathbf{a}^T_{2} \\
\vdots \\
\mathbf{a}^T_n \\
\end{pmatrix},
\quad B=\begin{pmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{pmatrix}.$$

Note here that each row vector $\mathbf{a}^T_{i}​$ lies in $\mathbb{R}^k​$ and that each column vector $\mathbf{b}_j​$ also lies in $\mathbb{R}^k​$.

각 행 백터  $\mathbf{a}^T_{i}$ 는  $\mathbb{R}^k$ 에 속하고, 각 열 백터 $\mathbf{b}_j$ 는  $\mathbb{R}^k$ 에 속한다는 것을 주의하세요.

Then to produce the matrix product $C \in \mathbb{R}^{n \times m}$ we simply compute each entry $c_{ij}$ as the dot product $\mathbf{a}^T_i \mathbf{b}_j$.

그러면, 행렬  $C \in \mathbb{R}^{n \times m}$ 의 각 원소 $c_{ij}$ 는  $\mathbf{a}^T_i \mathbf{b}_j$ 로 구해집니다.

$$C = AB = \begin{pmatrix}
\mathbf{a}^T_{1} \\
\mathbf{a}^T_{2} \\
\vdots \\
\mathbf{a}^T_n \\
\end{pmatrix}
\begin{pmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{pmatrix}
= \begin{pmatrix}
\mathbf{a}^T_{1} \mathbf{b}_1 & \mathbf{a}^T_{1}\mathbf{b}_2& \cdots & \mathbf{a}^T_{1} \mathbf{b}_m \\
 \mathbf{a}^T_{2}\mathbf{b}_1 & \mathbf{a}^T_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^T_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^T_{n} \mathbf{b}_1 & \mathbf{a}^T_{n}\mathbf{b}_2& \cdots& \mathbf{a}^T_{n} \mathbf{b}_m
\end{pmatrix}
$$

You can think of the matrix-matrix multiplication $AB$ as simply performing $m$ matrix-vector products and stitching the results together to form an $n \times m$ matrix. Just as with ordinary dot products and matrix-vector products, we can compute matrix-matrix products in MXNet by using ``nd.dot()``.

행렬-행렬 곱 $AB$ 을 단순히 $m$ 개의 행렬-백터의 곱을 수행한 후, 결과를 붙여서  $n \times m$ 행렬로 만드는 것으로 생각할 수도 있습니다. 일반적인 점곱과 행렬-백터 곱을 계산하는 것처럼 MXNet에서 행렬-행렬의 곱은 `nd.dot()` 으로 계산됩니다.

```{.python .input}
B = nd.ones(shape=(4, 3))
nd.dot(A, B)
```

## Norms

Before we can start implementing models,
there's one last concept we're going to introduce.
Some of the most useful operators in linear algebra are norms.
Informally, they tell us how big a vector or matrix is.
We represent norms with the notation $\|\cdot\|​$.
The $\cdot​$ in this expression is just a placeholder.
For example, we would represent the norm of a vector $\mathbf{x}​$
or matrix $A​$ as $\|\mathbf{x}\|​$ or $\|A\|​$, respectively.

모델을 구현하기 전에 배워야할 개념이 하나 더 있습니다. 선형대수에서 가장 유용한 연산 중에 놈(norm) 이 있습니다. 엄밀하지 않게는 설명하면, 놈은 백터나 행렬이 얼마나 큰지를 알려주는 개념입니다.  $\|\cdot\|$ 으로 놈을 표현하는데,  $\cdot$ 은 행렬이나 백터가 들어가 자리입니다. 예를 들면, 백터 $\mathbf{x}$ 나 행렬 $A$ 를 각각 $\|\mathbf{x}\|$ or $\|A\|$ 로 적습니다.

All norms must satisfy a handful of properties:

모든 놈은 다음 특성을 만족시켜야합니다.

1. $\|\alpha A\| = |\alpha| \|A\|$
1. $\|A + B\| \leq \|A\| + \|B\|$
1. $\|A\| \geq 0$
1. If $\forall {i,j}, a_{ij} = 0$, then $\|A\|=0$

To put it in words, the first rule says
that if we scale all the components of a matrix or vector
by a constant factor $\alpha​$,
its norm also scales by the *absolute value*
of the same constant factor.
The second rule is the familiar triangle inequality.
The third rule simply says that the norm must be non-negative.
That makes sense, in most contexts the smallest *size* for anything is 0.
The final rule basically says that the smallest norm is achieved by a matrix or vector consisting of all zeros.
It's possible to define a norm that gives zero norm to nonzero matrices,
but you can't give nonzero norm to zero matrices.
That's a mouthful, but if you digest it then you probably have grepped the important concepts here.

위 규칙을 말로 설명하면, 첫번째 규칙은 행렬이나 백터의 모든 원소에 상수 $\alpha$ 만큼 스캐일을 바꾸면, 놈도 그 상수의 *절대값* 만큼 스캐일이 바뀐다는 것입니다. 두번째 규칙은 친숙한 삼각부등식입니다. 세번째는 놈은 음수가 될 수 없다는 것입니다. 거의 모든 경우에 가장 작은 크기가 0이기에 이 규칙은 당연합니다. 마지막 규칙은 가장 작은 놈은 행렬 또는 백터가 0으로 구성되었을 경우라는 기본적인 것에 대한 것입니다. 0이 아닌 형렬에 놈이 0이 되도록 놈을 정의하는 것이 가능합니다. 하지만, 0인 행렬에 0이 아닌 놈이 되게하는 놈을 정의하는 것은 불가능합니다. 길게 설명했지만, 이해했다면 중요한 개념을 얻었을 것입니다.

If you remember Euclidean distances (think Pythagoras' theorem) from grade school,
then non-negativity and the triangle inequality might ring a bell.
You might notice that norms sound a lot like measures of distance.

수학시간에 배운 유클리디안 거리(Euclidean distance)를 기억한다면, 0이 아닌 것과 삼각부등식이 떠오를 것입니다. 놈이 거리를 측정하는 것과 비슷하다는 것은 인지했을 것입니다.

In fact, the Euclidean distance $\sqrt{x_1^2 + \cdots + x_n^2}​$ is a norm.
Specifically it's the $\ell_2​$-norm.
An analogous computation,
performed over the entries of a matrix, e.g. $\sqrt{\sum_{i,j} a_{ij}^2}​$,
is called the Frobenius norm.
More often, in machine learning we work with the squared $\ell_2​$ norm (notated $\ell_2^2​$).
We also commonly work with the $\ell_1​$ norm.
The $\ell_1​$ norm is simply the sum of the absolute values.
It has the convenient property of placing less emphasis on outliers.

사실 유클리디안 거리 $\sqrt{x_1^2 + \cdots + x_n^2}$ 는 놈입니다. 특히, 이를  $\ell_2$-놈이라고 합니다. 행렬의 각 원소에 대해서 유사하게 계산한 것   $\sqrt{\sum_{i,j} a_{ij}^2}$ 을 푸로베니우스 놈(Frobenius norm)이라고 합니다. 머신러닝에서는 자주 제곱  $\ell_2$ 놈을 사용합니다. ($\ell_2^2$ 로 표현합니다.)  $\ell_1$ 놈도 흔히 사용합니다.  $\ell_1$ 놈은 절대값들의 합으로, 특이점(outlier)에 덜 중점을 주는 편리한 특성이 있습니다.

To calculate the $\ell_2$ norm, we can just call ``nd.norm()``.

 $\ell_2$ 놈놈 계산은 `nd.norm()` 으로 합니다.

```{.python .input}
nd.norm(x)
```

To calculate the L1-norm we can simply perform the absolute value and then sum over the elements.

 $\ell_1$ 놈을 계산하는 방법은 각 원소의 절대값을 구한 후, 모두 합하는 것입니다.

```{.python .input}
nd.sum(nd.abs(x))
```

## Norms and objectives

While we don't want to get too far ahead of ourselves, we do want you to anticipate why these concepts are useful.
In machine learning we're often trying to solve optimization problems: *Maximize* the probability assigned to observed data. *Minimize* the distance between predictions and the ground-truth observations. Assign vector representations to items (like words, products, or news articles) such that the distance between similar items is minimized, and the distance between dissimilar items is maximized. Oftentimes, these objectives, perhaps the most important component of a machine learning algorithm (besides the data itself), are expressed as norms.

더 깊이 나가지는 않겠지만, 이 개념들이 왜 중요한지 궁금할 것입니다. 머신러닝에서 우리는 종종 최적화 문제를 풀기를 시도합니다: 관찰된 데이터에 할당된 확률을 *최대화*하기. 예측된 값과 실제 값의 차이를 *최소화*하기. 단어, 제품, 새로운 기사와 같은 아이템들에 가까운 아이템들의 거리가 최소화되는 백터를 할당하기. 자주 이 목적들 또는 아마도 머신러닝 알고리즘의 (데이터를 제외한) 가장 중요한 요소는 놈으로 표현됩니다.


## Intermediate linear algebra

If you've made it this far, and understand everything that we've covered,
then honestly, you *are* ready to begin modeling.
If you're feeling antsy, this is a perfectly reasonable place to move on.
You already know nearly all of the linear algebra required
to implement a number of many practically useful models
and you can always circle back when you want to learn more.

여러분이 여기까지 잘 따라오면서 모든 내용을 이해했다면, 솔직하게 여러분은 모델을 시작할 준비가 되었습니다. 먄약 조급함을 느낀다면, 이 절의 나머지는 넘어가도 됩니다. 실제로 적용할 수 있는 유용한 모델들을 구현하는데 필요한 모든 선형대수에 대해서 알아봤고, 더 알고 싶은면 다시 돌아올 수 있습니다.

But there's a lot more to linear algebra, even as concerns machine learning.
At some point, if you plan to make a career of machine learning,
you'll need to know more than we've covered so far.
In the rest of this chapter, we introduce some useful, more advanced concepts.

하지만, 머신러닝만 고려해봐도 더 많은 선형대수에 대한 내용이 있습니다. 이후 어느 시점에 여러분이 머신러닝 경력을 만들기를 원한다면, 여기서 다룬 것보다 더 많은 것을 알아야할 것입니다. 유용하고 더 어려운 개념을 소개하면서 이 절을 마치겠습니다.

### Basic vector properties

Vectors are useful beyond being data structures to carry numbers.
In addition to reading and writing values to the components of a vector,
and performing some useful mathematical operations,
we can analyze vectors in some interesting ways.

백터는 숫자를 담는 자료 구조보다 더 유용합니다. 백터의 원소에 숫자를 읽고 적는 것, 유용한 수학 연산을 수행하는 것과 더불어, 백터를 재미있는 방법으로 분석할 수 있습니다.

One important concept is the notion of a vector space.
Here are the conditions that make a vector space:

백터 공간의 개념은 중요한 개념입니다. 백터 공간이 되기에 필요한 조건은 다음과 같습니다.

* **Additive axioms** (we assume that x,y,z are all vectors):
  $x+y = y+x$ and $(x+y)+z = x+(y+z)$ and $0+x = x+0 = x$ and $(-x) + x = x + (-x) = 0$.
* **Multiplicative axioms** (we assume that x is a vector and a, b are scalars):
  $0 \cdot x = 0$ and $1 \cdot x = x$ and $(a b) x = a (b x)$.
* **Distributive axioms** (we assume that x and y are vectors and a, b are scalars):
  $a(x+y) = ax + ay$ and $(a+b)x = ax +bx$.

- **더하기 공리(Additive axioms)** (x,y,z가 모두 백터라고 가정합니다.):
  $x+y = y+x$ , $(x+y)+z = x+(y+z)$ , $0+x = x+0 = x$ 그리고 $(-x) + x = x + (-x) = 0$.
- **곱하기 공리(Multiplicative axioms)** (x는 백터이고 a, b는 스칼라입니다.):
  $0 \cdot x = 0$ , $1 \cdot x = x$ , $(a b) x = a (b x)$.
- **분배 공리(Distributive axioms)** (x와 y는 백터, a, b는 스칼라로 가정합니다.):
  $a(x+y) = ax + ay$ and $(a+b)x = ax +bx$.

### Special matrices

There are a number of special matrices that we will use throughout this tutorial. Let's look at them in a bit of detail:

이 책에서 사용할 특별한 행렬들이 있습니다. 그 행렬들에 대해서 조금 자세히 보겠습니다.

* **Symmetric Matrix** These are matrices where the entries below and above the diagonal are the same. In other words, we have that $M^\top = M$. An example of such matrices are those that describe pairwise distances, i.e. $M_{ij} = \|x_i - x_j\|$. Likewise, the Facebook friendship graph can be written as a symmetric matrix where $M_{ij} = 1$ if $i$ and $j$ are friends and $M_{ij} = 0$ if they are not. Note that the *Twitter* graph is asymmetric - $M_{ij} = 1$, i.e. $i$ following $j$ does not imply that $M_{ji} = 1$, i.e. $j$ following $i$.
* **Antisymmetric Matrix** These matrices satisfy $M^\top = -M$. Note that any arbitrary matrix can always be decomposed into a symmetric and into an antisymmetric matrix by using $M = \frac{1}{2}(M + M^\top) + \frac{1}{2}(M - M^\top)$.
* **Diagonally Dominant Matrix** These are matrices where the off-diagonal elements are small relative to the main diagonal elements. In particular we have that $M_{ii} \geq \sum_{j \neq i} M_{ij}$ and $M_{ii} \geq \sum_{j \neq i} M_{ji}$. If a matrix has this property, we can often approximate $M$ by its diagonal. This is often expressed as $\mathrm{diag}(M)$.
* **Positive Definite Matrix** These are matrices that have the nice property where $x^\top M x > 0$ whenever $x \neq 0$. Intuitively, they are a generalization of the squared norm of a vector $\|x\|^2 = x^\top x$. It is easy to check that whenever $M = A^\top A$, this holds since there $x^\top M x = x^\top A^\top A x = \|A x\|^2$. There is a somewhat more profound theorem which states that all positive definite matrices can be written in this form.
* **대칭 행렬(Symmetric Matrix)** 이 행렬들은 대각선 아래, 위의 원소들이 같은 값을 갖습니다. 즉,  $M^\top = M$ 입니다. 이런 예로는 pairwise 거리를 표현하는 행렬 $M_{ij} = \|x_i - x_j\|$이 있습니다. 패이스북 친구 관계를 대칭 행렬로 표현할 수 있습니다. $i$ 와 $j$ 가 친구라면 $M_{ij} = 1$ 이 되고, 친구가 아니라면 $M_{ij} = 0$ 로 표현하면 됩니다. 하지만, 트위터 그래프는 대칭이 아님을 주목해세요. $M_{ij} = 1$, 즉 $i$ 가 $j$ 를 팔로위하는 것이 꼭 $j$ 가 $i$ 를 팔로위하는 것, $M_{ji} = 1$, 은 아니기 때문입니다.
* **비대칭 행렬(Antisymmetric Matrix)**  $M^\top = -M$ 를 만족하는 행렬입니다. 임의의 행렬은 대칭 행렬과 비대칭 행렬로 분해될 수 있습니다. 즉, $M = \frac{1}{2}(M + M^\top) + \frac{1}{2}(M - M^\top)$ 로 표현될 수 있습니다.
* **대각 지배 행렬(Diagonally Dominant Matrix)** 대각 원소들 보다 대각이 아닌 원소들이 작은 행렬입니다.즉,  $M_{ii} \geq \sum_{j \neq i} M_{ij}$ 이고 $M_{ii} \geq \sum_{j \neq i} M_{ji}$ 입니다. 어떤 행렬이 이 특성을 갖는다면, 대각원로를 사용해서 $M$ 을 추정할 수 있고, 이를  $\mathrm{diag}(M)$ 로 표기합니다.
* **양의 정부호 행렬(Positive Definite Matrix)** 이 행렬은  $x \neq 0$ 이면,  $x^\top M x > 0$ 인 좋은 특성을 갖습니다. 직관적으로 설명하면, 백터의 제곱 놈,  $\|x\|^2 = x^\top x$, 의 일반화입니다. $M = A^\top A$ 이면 이 조건이 만족시킨다는 것을 쉽게 확인할 수 있습니다. 이유는  $x^\top M x = x^\top A^\top A x = \|A x\|^2$ 이기 때문입니다. 모든 양의 정부호 행렬은 이런 형태로 표현될 수있다는 더 심오한 이론이 있습니다.


## Summary

In just a few pages (or one Jupyter notebook) we've taught you all the linear algebra you'll need to understand a good chunk of neural networks. Of course there's a *lot* more to linear algebra. And a lot of that math *is* useful for machine learning. For example, matrices can be decomposed into factors, and these decompositions can reveal low-dimensional structure in real-world datasets. There are entire subfields of machine learning that focus on using matrix decompositions and their generalizations to high-order tensors to discover structure in datasets and solve prediction problems. But this book focuses on deep learning. And we believe you'll be much more inclined to learn more mathematics once you've gotten your hands dirty deploying useful machine learning models on real datasets. So while we reserve the right to introduce more math much later on, we'll wrap up this chapter here.

몇 장 (또는 Jupyter 노트북 한개)을 통해서 뉴럴 네트워크의 중요한 부분들을 이해하는데 필요한 모든 선형대수에 대해서 알아봤습니다. 물론 선형대수에는 더 많은 내용이 있고, 머신러닝에 유용하게 쓰입니다. 예를 들어, 형렬을 분해할 수 있고, 이 분해는 실세계의 데이터셋의 아래 차원의 구조를 알려주기도 합니다. 행렬 분해를 이용하는데 집중하는 머신러닝의 별도의 분야가 있습니다. 이를 이용해서 데이터의 구조를 밝히고 예측 문제를 풀기 위해서 고차원의 텐서를 일반화하기도 합니다. 하지만 이 책에서는 딥러닝에 집중합니다. 여러분이 실제 데이터를 사용해서 유용한 머신러닝 모델을 만들기 시작한다면, 수학에 대해서 더 관심을 갖게될 것이라고 믿습니다. 하지만 수학적인 내용은 나중에 더 설명하기로 하고, 이 절은 여기서 마무리하겠습니다.

If you're eager to learn more about linear algebra, here are some of our favorite resources on the topic

선행대수에 대해서 더 배우기를 원한다면, 유용한 교재들이 있습니다.

* For a solid primer on basics, check out Gilbert Strang's book [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/)
* Zico Kolter's [Linear Algebra Review and Reference](http://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf)
* 탄탄한 기초를 쌓고 싶으면,  Gilbert Strang의 책 [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/)를 참고하세요.
* Zico Kolter's [Linear Algebra Review and Reference](http://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf)

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2317)

![](../img/qr_linear-algebra.svg)
