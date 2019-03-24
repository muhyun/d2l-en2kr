# Documentation

Due to the length of this book, it is impossible for us to introduce all  MXNet functions and classes. The API documentation and additional tutorials and examples provide plenty of documentation beyond the book.

이 책에서 MXNet 함수와 클래스를 모두 설명하기는 불가능하니, API 문서나 추가적인 튜토리얼과 예제를 참고하면 이 책에서 다루지 못한 많은 내용을 찾아볼 수 있습니다.

## Finding all the functions and classes in the module

In order to know which functions and classes can be called in a module, we use the `dir` function. For instance we can query all the members or properties in the `nd.random` module.

모듈에서 어떤 함수와 클래스가 제공되는지 알기 위해서 `dir` 함수를 이용합니다. 예를 들어, `nd.random` 모듈의 모든 맴버와 속성을 다음과 같이 조회할 수 있습니다.

```{.python .input  n=1}
from mxnet import nd
print(dir(nd.random))
```

Generally speaking, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). According to the remaining member names, we can then hazard a  guess that this module offers a generation method for various random numbers, including uniform distribution sampling (`uniform`), normal distribution sampling (`normal`), and Poisson sampling  (`poisson`).

일번적으로는 이름이 `__` 로 시작하는 함수(Python에서 특별한 객체를 나타냄)나 `_` 로 시작하는 함수(보통은 내부 함수들)는 무시해도 됩니다. 나머지 맴버들에 대해서는 이름을 통해서 추측해보면, 다양한 난수를 생성하는 메소드들로 추측할 수 있습니다. 즉, 균일한 분포에서 난수를 생성하는 `uniform`, 표준 분산에서 난수를 생성하는 `normal` 그리고 Poisson 샘플링인 `poisson` 등의 기능을 제공함을 알 수 있습니다.

## Finding the usage of specific functions and classes

For specific function or class usage, we can use the  `help` function. Let's take a look at the usage of the `ones_like` function of an NDArray as an example.

`help` 함수를 이용하면 특정 함수나 클래스의 사용법 확인할 수 있습니다. NDArray의 `ones_like` 함수를 예로 살펴봅니다.

```{.python .input}
help(nd.ones_like)
```

From the documentation, we learned that the `ones_like` function creates a new one with the same shape as the NDArray and an element of 1. Let's verify it:

문서를 보면, `ones_like` 함수는 NDArray 객체와 모두 1로 설정된 같은 shape의 새로운 객체를 만들어 줍니다. 확인해보겠습니다.

```{.python .input}
x = nd.array([[0, 0, 0], [2, 2, 2]])
y = x.ones_like()
y
```

In the Jupyter notebook, we can use `?` to display the document in another window. For example, `nd.random.uniform?` will create content that is almost identical to `help(nd.random.uniform)`, but will be displayed in an extra window. In addition, if we use two `nd.random.uniform??`, the function implementation code will also be displayed.

Jupyter 노트북에서는 `?` 를 이용해서 다른 윈도우에 문서를 표시할 수 있습니다. 예를 들어 `nd.random.uniform?` 를 수행하면 `help(nd.random.uniform)` 과 거의 동일한 내용이 다른 윈도우에 나옵니다. 그리고, `nd.random.uniform??` 와 같이 `?` 를 두개 사용하면, 함수를 구현하는 코드도 함께 출력됩니다.

## API Documentation

For further details on the API details check the MXNet website at  [http://mxnet.apache.org/](http://mxnet.apache.org/). You can find the details under the appropriate headings (also for programming languages other than Python).

API에 대한 더 자세한 내용은 MXNet 웹사이트 [http://mxnet.apache.org/](http://mxnet.apache.org/) 를 확인하세요. Python 및 이외의 다른 프로그램 언어에 대한 내용들을 웹 사이트에서 찾을 수 있습니다.

## Problem

Look up `ones_like` and `autograd` in the API documentation.

API 문서에서 `ones_like` 와 `autograd` 를 찾아보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2322)

![](../img/qr_lookup-api.svg)
