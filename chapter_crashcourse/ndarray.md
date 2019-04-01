# Data Manipulation

It's impossible to get anything done if we can't manipulate data.
Generally, there are two important things we need to do with data:
(i) acquire it and (ii) process it once it's inside the computer.
There's no point in trying to acquire data if we don't even know how to store it, so let's get our hands dirty first by playing with synthetic data.

데이터를 변경할 수 없다면 아무것도 할 수 업습니다. 일반적으로 우리는 데이터를 사용해서 두가지 중요한 일을 합니다. (i) 데이터를 얻고, (ii) 컴퓨터에서 들어오면 처리하기. 데이터를 저장하는 방법을 모른다면 데이터를 얻는 것의 의미가 없으니, 합성된 데이터를 다루는 것부터 시작합겠습니다.

We'll start by introducing NDArrays, MXNet's primary tool for storing and transforming data. If you've worked with NumPy before, you'll notice that NDArrays are, by design, similar to NumPy's multi-dimensional array. However, they confer a few key advantages. First, NDArrays support asynchronous computation on CPU, GPU, and distributed cloud architectures. Second, they provide support for automatic differentiation. These properties make NDArray an ideal ingredient for machine learning.

MXNet에서 데이터를 저장하고 변경하는 주요 도구인 NDArray를 소개하겠습니다. NumPy를 사용해봤다면, NDArray가 NumPy의 다차원 배열과 디자인상으로 비슷하다는 것을 눈치챌 것입니다. 하지만, 주요한 장점들이 있습니다. 첫번째로는 NDArray는 CPU, GPU 그리고 분산 클라우드 아키텍처에서 비동기 연산을 지원합니다. 두번째는, 자동 미분을 지원합니다. 이 특징들 때문에 NDArray는 머신러닝에 이상적인 요소라고 할 수 있습니다.

## Getting Started

In this chapter, we'll get you going with the basic functionality. Don't worry if you don't understand any of the basic math, like element-wise operations or normal distributions. In the next two chapters we'll take another pass at NDArray, teaching you both the math you'll need and how to realize it in code. For even more math, see the ["Math"](../chapter_appendix/math.md) section in the appendix.

이 절에서는 여러분은 기본적인 것을 다룰 것입니다. 원소끼리의 연산 이나 표준 분포와 같은 기본적인 수학 내용을 이해하지 못해도 걱정하지 마세요. 다음 두 절에서 필요한 수학과 어떻게 코드로 구현하는지를 다룰 예정입니다. 수학에 대해서 더 알고 싶다면, 부록에  ["Math"](../chapter_appendix/math.md) 를 참고하세요.

We begin by importing MXNet and the `ndarray` module from MXNet. Here, `nd` is short for `ndarray`.

MXNet과 MXNet의 `ndarray` 모듈을 import 합니다. 여기서는 `ndarray` 를 `nd` 라고 별칭을 주겠습니다.

```{.python .input  n=1}
import mxnet as mx
from mxnet import nd
```

The simplest object we can create is a vector. `arange` creates a row vector of 12 consecutive integers.

우리가 만들 수 있는 가장 단순한 객체는 백터입니다. `arange` 는 12개의 연속된 정수를 갖는 행 백터를 생성합니다.

```{.python .input  n=2}
x = nd.arange(12)
x
```

From the property `<NDArray 12 @cpu(0)>` shown when printing `x` we can see that it is a one-dimensional array of length 12 and that it resides in CPU main memory. The 0 in `@cpu(0)`` has no special meaning and does not represent a specific core.

`x` 를 출력할 때 나온  `<NDArray 12 @cpu(0)>`  로 부터, 우리는 이것이 길이가 12인 일차원 배열이고, CPU의 메인메모리에 저장되어 있다는 것을 알 수 있습니다. `@cpu(0)`에서 0은 아무런 의미가 없고, 특정 코어를 의미하지도 않습니다.

We can get the NDArray instance shape through the `shape` property.

NDArray 인스턴스의 shape은 `shape` 속성으로 얻습니다.

```{.python .input  n=8}
x.shape
```

We can also get the total number of elements in the NDArray instance through the `size` property. Since we are dealing with a vector, both are identical.

`size` 속성은 NDArray 인스턴스의 원소 총 개수를 알려줍니다. 우리는 백터를 다루고 있기 때문에 두 결과는 같습니다.

```{.python .input  n=9}
x.size
```

In the following, we use the `reshape` function to change the shape of the line vector `x` to (3, 4), which is a matrix of 3 rows and 4 columns. Except for the shape change, the elements in `x` (and also its size) remain unchanged.

행 백터를 3행, 4열의 행렬로 바꾸기 위해서, 즉 shape을 바꾸기 위해서 `reshape` 함수를 사용합니다. shape이 바뀌는 것을 제외하고는 `x` 의 원소와 크기는 변하지 않습니다.

```{.python .input  n=3}
x = x.reshape((3, 4))
x
```

It can be awkward to reshape a matrix in the way described above. After all, if we want a matrix with 3 rows we also need to know that it should have 4 columns in order to make up 12 elements. Or we might want to request NDArray to figure out automatically how to give us a matrix with 4 columns and whatever number of rows that are needed to take care of all elements. This is precisely what the entry `-1` does in any one of the fields. That is, in our case
`x.reshape((3, 4))` is equivalent to `x.reshape((-1, 4))` and `x.reshape((3, -1))`.

위와 같이 행렬의 모양을 바꾸는 것은 좀 이상할 수 있습니다. 결국, 3개 행을 갖는 행렬을 원한다면 총 원소의 개수가 12개가 되기 위해서 열이 4가 되어야한다는 것을 알아야합니다. 또는, NDArray에게 행의 개수가 몇개이든지 모든 원소를 포함하는 열이 4개인 행렬을 자동으로 찾아내도록 요청하는 것도 가능합니다. 즉, 위 경우에는 `x.reshape((3, 4))` 는  `x.reshape((-1, 4))`  와  `x.reshape((3, -1))` 와 같습니다.

```{.python .input}
nd.empty((3, 4))
```

The `empty` method just grabs some memory and hands us back a matrix without setting the values of any of its entries. This is very efficient but it means that the entries can have any form of values, including very big ones! But typically, we'll want our matrices initialized.

`empty`  메소드는 shape에 따른 메모리를 잡아서 원소들의 값을 설정하지 않고 행렬를 반환합니다. 이는 아주 유용하지만, 원소들이 어떤 형태의 값도 갖을 수 있는 것을 의미합니다. 이는 매우 큰 값들일 수도 있습니다. 하지만, 일반적으로는 행렬을 초기화하는 것을 원합니다.

Commonly, we want one of all zeros. For objects with more than two dimensions mathematicians don't have special names - they simply call them tensors. To create one with all elements set to 0 and a shape of (2, 3, 4) we use

보통은 모두 0으로 초기화하기를 원합니다. 수학자들은 이차원 보다 큰 객체들에 대해서는 특별한 이름을 쓰지 않지만, 우리는 이것들은 텐서(tensor)라고 부르겠습니다. 모든 원소가 0이고 shape이 (2,3,4)인 텐서를 하나 만들기 위해서 다음과 같이 합니다.

```{.python .input  n=4}
nd.zeros((2, 3, 4))
```

Just like in NumPy, creating tensors with each element being 1 works via

NumPy 처럼, 모든 원소가 1인 텐서를 만드는 방법은 다음과 같습니다.

```{.python .input  n=5}
nd.ones((2, 3, 4))
```

We can also specify the value of each element in the NDArray that needs to be created through a Python list.

Python 리스트를 이용해서 NDArray의 각 원소 값을 지정하는 것도 가능합니다.

```{.python .input  n=6}
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
y
```

In some cases, we need to randomly generate the value of each element in the NDArray. This is especially common when we intend to use the array as a parameter in a neural network. The following  creates an NDArray with a shape of (3,4). Each of its elements is randomly sampled in a normal distribution with zero mean and unit variance.

어떤 경우에는, NDArray의 값을 임의로 채우기를 원할 때가 있습니다. 이는 특히 뉴럴 네트워크의 파라메터로 배열을 사용할 때 일반적입니다. 아래 코드는 shape이 (3,4) NDArray를 생성하고, 각 원소는 평균이 0이고 분산이 1인 표준 분포로부터 임의로 추출한 값을 갖습니다.

```{.python .input  n=7}
nd.random.normal(0, 1, shape=(3, 4))
```

## Operations

Oftentimes, we want to apply functions to arrays.
Some of the simplest and most useful functions are the element-wise functions.
These operate by performing a single scalar operation on the corresponding elements of two arrays.
We can create an element-wise function from any function that maps from the scalars to the scalars.
In math notations we would denote such a function as $f: \mathbb{R} \rightarrow \mathbb{R}​$.
Given any two vectors $\mathbf{u}​$ and $\mathbf{v}​$ *of the same shape*, and the function f,
we can produce a vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})​$
by setting $c_i \gets f(u_i, v_i)​$ for all $i​$.
Here, we produced the vector-valued $F: \mathbb{R}^d \rightarrow \mathbb{R}^d​$
by *lifting* the scalar function to an element-wise vector operation.
In MXNet, the common standard arithmetic operators (+,-,/,\*,\*\*)
have all been *lifted* to element-wise operations for identically-shaped tensors of arbitrary shape. We can call element-wise operations on any two tensors of the same shape, including matrices.

우리는 종종 배열에 함수를 적용할 필요가 있습니다. 아주 간단하고, 굉장히 유용한 함수 중에 하나로 원소끼리(element-wise) 함수가 있습니다. 이 연산은 두 배열의 동일한 위치에 있는 원소들에 스칼라 연산을 수행하는 것입니다. 스칼라를 스칼라로 매핑하는 함수를 사용하면 언제나 원소끼리(element-wise) 함수를 만들 수 있습니다. 수학 기호로는 이런 함수를  $f: \mathbb{R} \rightarrow \mathbb{R}$ 로 표현합니다. *같은 shape*의 두 백터 $\mathbf{u}$ 와 $\mathbf{v}$ 와 함수 f가 주어졌을 때, 모든 $i$ 에 대해서  $c_i \gets f(u_i, v_i)$ 을 갖는 백터  $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ 를 만들 수 있습니다. 즉, 우리는 스칼라 함수를 백터의 원소별로 적용해서 백터 함수 $F: \mathbb{R}^d \rightarrow \mathbb{R}^d$ 를 만들었습니다. MXNet에서는 일반적인 표준 산술 연산자들(+,-,/,\*,\*\*)은 shape이 무엇이든지 상관없이 두 텐선의 shape이 같을 경우 모두 원소별 연산로 간주되서 계산됩니다. 즉, 행렬을 포함한 같은 shape을 갖는 임의의 두 텐서에 대해서 원소별 연산을 수행할 수 있습니다.

```{.python .input}
x = nd.array([1, 2, 4, 8])
y = nd.ones_like(x) * 2
print('x =', x)
print('x + y', x + y)
print('x - y', x - y)
print('x * y', x * y)
print('x / y', x / y)
```

Many more operations can be applied element-wise, such as exponentiation:

제곱과 같은 더 많은 연산들이 원소별 연산으로 적용될 수 있습니다.

```{.python .input  n=12}
x.exp()
```

In addition to computations by element, we can also perform matrix operations, like matrix multiplication using the `dot` function. Next, we will perform matrix multiplication of `x` and the transpose of `y`. We define `x` as a matrix of 3 rows and 4 columns, and `y` is transposed into a matrix of 4 rows and 3 columns. The two matrices are multiplied to obtain a matrix of 3 rows and 3 columns (if you're confused about what this means, don't worry - we will explain matrix operations in much more detail in the chapter on [linear algebra](linear-algebra.md)).

원소별 연산과 더불어서, `dot` 함수를 이용한 형렬의 곱처럼 행렬의 연산들도 수행할 수 있습니다. 행렬 `x` 와 `y` 의 전치행렬에 대해서 행렬의 곱을 수행해보겠습니다. `x` 는 행이 3, 열이 4인 행렬이고, `y` 는 행이 4개 열이 3개를 갖도록 전치시킵니다. 두 행렬을 곱하면 행이 3, 열이 3인 행렬이 됩니다. (이것이 어떤 의미인지 햇깔려도 걱정하지 마세요. [linear algebra](linear-algebra.md) 절에서 행렬의 연산에 대한 것들을 설명할 예정입니다.)

```{.python .input  n=13}
x = nd.arange(12).reshape((3,4))
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
nd.dot(x, y.T)
```

We can also merge multiple NDArrays. For that, we need to tell the system along which dimension to merge. The example below merges two matrices along dimension 0 (along rows) and dimension 1 (along columns) respectively.

여러 NDArray들을 합치는 것도 가능합니다. 이를 위해서는 어떤 차원(dimension)을 따라서 합쳐야하는지를 알려줘야합니다. 아래 예제는 각각 차원 0 (즉, 행들)과 차원 1 (열들)을 따라서 두 행렬을 합칩니다.

```{.python .input}
nd.concat(x, y, dim=0)
nd.concat(x, y, dim=1)
```

Just like in NumPy, we can construct binary NDArrays by a logical statement. Take `x == y` as an example. If `x` and `y` are equal for some entry, the new NDArray has a value of 1 at the same position; otherwise, it is 0.

NumPy에서와 같이 논리 문장을 사용해서 이진 NDArray를 만들 수 있습니다.  `x == y` 를 예로 들어보겠습니다. 만약 `x` 와 `y` 가 같은 원소가 있다면, 새로운 NDArray는 그 위치에 1을 갖고, 다른 값이면 0을 갖습니다.

```{.python .input}
x == y
```

Summing all the elements in the NDArray yields an NDArray with only one element.

NDArray의 모든 요소를 더하면 하나의 원소를 갖는 NDArray가 됩니다. 

```{.python .input}
x.sum()
```

We can transform the result into a scalar in Python using the `asscalar` function. In the following example, the $\ell_2$ norm of `x` yields a single element NDArray. The final result is transformed into a scalar.

`asscalar` 함수를 이용해서 결과를 Python의 스칼라로 바꿀 수 있습니다. 아래 예제는 `x` 의  $\ell_2$ 놈 계산합니다. 이 결과는 하나의 원소를 갖는 NDArray이고, 이를 Python의 스칼라 값으로 바꿉니다.

```{.python .input}
x.norm().asscalar()
```

For stylistic convenience, we can write `y.exp()`, `x.sum()`, `x.norm()`, etc. also as `nd.exp(y)`, `nd.sum(x)`, `nd.norm(x)`.

표기를 편하기 해기 위해서  `y.exp()`, `x.sum()`, `x.norm()`, 등을 각각  `nd.exp(y)`, `nd.sum(x)`, `nd.norm(x)` 처럼 쓸 수도 있습니다.

## Broadcast Mechanism

In the above section, we saw how to perform operations on two NDArrays of the same shape. When their shapes differ, a broadcasting mechanism may be triggered analogous to NumPy: first, copy the elements appropriately so that the two NDArrays have the same shape, and then carry out operations by element.

위 절에서 우리는 같은 shape의 두 NDArray 객체에 대한 연산을 어떻게 수행하는지를 살펴봤습니다. 만약 shape이 다른 경우에는 NumPy와 같이 브로드케스팅 메카니즘이 적용됩니다: 즉, 두 NDArray가 같은 shape을 갖도록 원소들이 복사된 후, 원소별로 연산을 수행하게됩니다.

```{.python .input  n=14}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
a, b
```

Since `a` and `b` are (3x1) and (1x2) matrices respectively, their shapes do not match up if we want to add them. NDArray addresses this by 'broadcasting' the entries of both matrices into a larger (3x2) matrix as follows: for matrix `a` it replicates the columns, for matrix `b` it replicates the rows before adding up both element-wise.

`a`와 `b` 는 각각  (3x1),  (1x2) 행렬이기 때문에, 두 행렬을 더하기에는 shape이 일치하지 않습니다. NDArray는 이런 상황을 두 행렬의 원소들을 더 큰 행렬 (3x2)로 '브로드케스팅' 해서 해결합니다. 즉, 행렬 `a` 는 컬럼을 복제하고, 행렬 `b` 는 열을 복제한 후, 원소별 덧셈을 수행합니다.

```{.python .input}
a + b
```

## Indexing and Slicing

Just like in any other Python array, elements in an NDArray can be accessed by its index. In good Python tradition the first element has index 0 and ranges are specified to include the first but not the last. By this logic `1:3` selects the second and third element. Let's try this out by selecting the respective rows in a matrix.

다른 Python 배열처럼 NDArray의 원소들도 인덱스를 통해서 지정할 수 있습니다. Python에서 처럼 첫번째 원소의 인덱스는 0이고, 범위는 첫번째 원소는 포함하고 마지막은 포함하지 않습니다. 즉, `1:3` 은 두번째와 세번째 원소를 선택하는 범위입니다. 행렬에서 핼들을 선택하는 예는 다음과 같습니다.

```{.python .input  n=19}
x[1:3]
```

Beyond reading we can also write elements of a matrix.

값을 읽는 것말고도, 행렬의 원소값을 바꾸는 것도 가능합니다.

```{.python .input  n=20}
x[1, 2] = 9
x
```

If we want to assign multiple elements the same value, we simply index all of them and then assign them the value. For instance, `[0:2, :]` accesses the first and second rows. While we discussed indexing for matrices, this obviously also works for vectors and for tensors of more than 2 dimensions.

여러 원소에 같은 값을 할당하고 싶을 경우에는, 그 원소들에 대한 인덱스를 모두 지정해서 값을 할당하는 것으로 간단히 할 수 있습니다. 예를 들어 `[0:2, :]` 는 첫번째와 두번째 행을 의미합니다. 행렬에 대한 인덱싱을 이야기해왔지만, 백터나 2개 보다 많은 차원을 갖는 텐서에도 동일하게 적용됩니다.

```{.python .input  n=21}
x[0:2, :] = 12
x
```

## Saving Memory

In the previous example, every time we ran an operation, we allocated new memory to host its results. For example, if we write `y = x + y`, we will dereference the matrix that `y` used to point to and instead point it at the newly allocated memory. In the following example we demonstrate this with Python's `id()` function, which gives us the exact address of the referenced object in memory. After running `y = y + x`, we'll find that `id(y)` points to a different location. That's because Python first evaluates `y + x`, allocating new memory for the result and then subsequently redirects `y` to point at this new location in memory.

앞의 예제들 모두 연산을 수행할 때마다 새로운 메모리를 할당해서 결과를 저장합니다. 예를 들어,  `y = x + y` 를 수행하면, 원래의 행렬 `y` 에 대한 참조는 제거되고, 새로 할당된 메모리를 참조하도록 동작합니다. 다음 예제에서는, 객체의 메모리 주소를 반화하는 Python의 `id()` 함수를 이용해서 이를 확인해보겠습니다.  `y = x + y` 수행 후, `id(y)` 는 다른 위치를 가르키고 있습니다. 이렇게 되는 이유는 Python은  `y + x` 연산 결과를 새로운 메모리에 저장하고, `y` 가 새로운 메모리를 참조하도록 작동하기 때문입니다.

```{.python .input  n=15}
before = id(y)
y = y + x
id(y) == before
```

This might be undesirable for two reasons. First, we don't want to run around allocating memory unnecessarily all the time. In machine learning, we might have hundreds of megabytes of parameters and update all of them multiple times per second. Typically, we'll want to perform these updates *in place*. Second, we might point at the same parameters from multiple variables. If we don't update in place, this could cause a memory leak, and could cause us to inadvertently reference stale parameters.

이는 두가지 이유로 바람직하지 않을 수 있습니다. 첫번째로는 매번 불필요한 메모리를 할당하는 것을 원하지 않습니다. 머신러닝에서는 수백 메가 바이크의 패라메터들을 매초마다 여러번 업데이트를 수행합니다. 대부분의 경우 우리는 이 업데이트를 *같은 메모리(in-place)*를 사용해서 수행하기를 원합니다. 두번째는 여러 변수들이 같은 파라메터를 가르키고 있을 수 있습니다. 같은 메모리에 업데이트를 하지 않을 경우, 메모리 누수가 발생하고, 래퍼런스가 유효하지 않은 파라메터를 만드는 문제가 발생할 수 있습니다.

Fortunately, performing in-place operations in MXNet is easy. We can assign the result of an operation to a previously allocated array with slice notation, e.g., `y[:] = <expression>`. To illustrate the behavior, we first clone the shape of a matrix using `zeros_like` to allocate a block of 0 entries.

다행하게도 MXNet에서 같은 메모리 연산은 간단합니다. 슬라이스 표기법 `y[:] = <expression>` 을 이용하면 이전에 할당된 배열에 연산의 결과를 저장할 수 있습니다. `zeros_like` 함수를 사용해서 동일한 shape을 갖고 원소가 모두 0인 행렬을 하나 복사해서 이것이 어떻게 동작하는지 보겠습니다.

```{.python .input  n=16}
z = y.zeros_like()
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

While this looks pretty, `x+y` here will still allocate a temporary buffer to store the result of `x+y` before copying it to `y[:]`. To make even better use of memory, we can directly invoke the underlying `ndarray` operation, in this case `elemwise_add`, avoiding temporary buffers. We do this by specifying the `out` keyword argument, which every `ndarray` operator supports:

멋져 보이지만, `x+y` 는 결과값 계산하고 이를 `y[:]` 에 복사하기 전에 이 값을 저장하는 임시 버퍼를 여전히 할당합니다. 메모리를 더 잘 사용하기 위해서, `ndarray` 연산(이 경우는 `elemwise_add`)을 직접 호출해서 임시 버퍼의 사용을 피할 수 있습니다. 모든 `ndarray` 연산자가 제공하는  `out` 키워드를 이용하면 됩니다.

```{.python .input  n=17}
before = id(z)
nd.elemwise_add(x, y, out=z)
id(z) == before
```

If the value of `x ` is not reused in subsequent programs, we can also use `x[:] = x + y` or `x += y` to reduce the memory overhead of the operation.

`x` 값이 프로그램에서 더 이상 사용되지 않을 경우,  `x[:] = x + y` 이나  `x += y` 로 연산으로 인한 메모리 추가 사용을 줄일 수 있습니다.

```{.python .input  n=18}
before = id(x)
x += y
id(x) == before
```

## Mutual Transformation of NDArray and NumPy

Converting MXNet NDArrays to and from NumPy is easy. The converted arrays do *not* share memory. This minor inconvenience is actually quite important: when you perform operations on the CPU or one of the GPUs, you don't want MXNet having to wait whether NumPy might want to be doing something else with the same chunk of memory. The  `array` and `asnumpy` functions do the trick.

MXNet NDArray를 NumPy로 변환하는 것은 간단합니다. 변환된 배열은 메모리를 공유하지 않습니다. 이것은 사소하지만 아주 중요합니다. CPU 또는 GPU 하나를 사용해서 연산을 수행할 때, NumPy가 동일한 메모리에서 다른 일을 수행하는 것을 MXNet이 기다리는 것을 원하지 않기 때문입니다. `array` 와 `asnumpy` 함수를 이용하면 변환을 할 수 있습니다.

```{.python .input  n=22}
import numpy as np

a = x.asnumpy()
print(type(a))
b = nd.array(a)
print(type(b))
```

## Problems

1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of NDArray you can get.
1. Replace the two NDArrays that operate by element in the broadcast mechanism with other shapes, e.g. three dimensional tensors. Is the result the same as expected?
1. Assume that we have three matrices `a`, `b` and `c`. Rewrite `c = nd.dot(a, b.T) + c` in the most memory efficient manner.
1. 이 절의 코드를 실행하세요. 조건문 `x == y` 를  `x < y`  이나 `x > y` 로 바뀌서 결과가 어떻게되는지 확인하세요.
1. 다른 shape 행렬들에 브로드케스팅이 적용되는 연산을 수행하는 두 NDArray를 바꿔보세요. 예를 들면 3 차원 텐서로 바꿔보세요. 예상한 결과도 같나요?
1. 행렬 3개 `a`, `b`, `c` 가 있을 경우,  `c = nd.dot(a, b.T) + c` 를 가장 메모리가 효율적인 코드로 바꿔보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2316)

![](../img/qr_ndarray.svg)
