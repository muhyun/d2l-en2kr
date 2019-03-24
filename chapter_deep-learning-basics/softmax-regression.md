# Softmax Regression

Over the last two sections we worked through how to implement a linear regression model,
both [*from scratch*](linear-regression-scratch.ipynb)
and [using Gluon](linear-regression-gluon.ipynb) to automate most of the repetitive work
like allocating and initializing parameters, defining loss functions, and implementing optimizers.

앞 두 절, [*from scratch*](linear-regression-scratch.ipynb) 와 [using Gluon](linear-regression-gluon.ipynb)을 통해서 선형 회귀 모델을 직접 구현해보기도 했고, Gluon을 이용해서 구현해보았습니다. Gluon을 이용하면 파라메터 정의나 초기화, loss 함수 정의, optimizer 구현과 같은 반복된 일을 자동화할 수 있었습니다.

Regression is the hammer we reach for when we want to answer *how much?* or *how many?* questions.
If you want to predict the number of dollars (the *price*) at which a house will be sold,
or the number of wins a baseball team might have,
or the number of days that a patient will remain hospitalized before being discharged,
then you're probably looking for a regression model.

회귀(regression)는 몇 개인지, 얼마인지 등에 대한 답을 구할 때 사용하는 도구로, 예를 들면 집 가격이 얼마인지, 어떤 야구팀이 몇 번 승리를 할 것인지 등을 예측하는데 사용할 수 있는 방법입니다. 다른 예로는, 환자가 몇 일 만에 퇴원할 것인지 예측하는 것도 회귀(regression) 문제입니다. 

In reality, we're more often interested in making categorical assignments.

현실에서는 어떤 카테고리에 해당하는지를 예측하는 문제를 더 많이 접하게 됩니다.

* Does this email belong in the spam folder or the inbox*?
* How likely is this customer to sign up for subscription service?*
* What is the object in the image (donkey, dog, cat, rooster, etc.)?
* Which object is a customer most likely to purchase?
* 메일이 스팸인지 아닌지
* 고객이 구독 서비스에 가입할지 아닐지
* 이미지에 있는 객체가 무엇인지 (원숭이, 강아지, 고양이, 닭 등)
* 고객이 어떤 물건을 구매할 것인지

When we're interested in either assigning datapoints to categories
or assessing the *probability* that a category applies,
we call this task *classification*. The issue with the models that we studied so far is that they cannot be applied to problems of probability estimation.

카테고리별로 값을 할당한다든지, 어떤 카테고리에 속할 확률이 얼마나 되는지를 예측하는 것은 분류(classification) 라고 부릅니다. 앞 절들에서 살펴본 모델은 확률을 예측하는 문제에 적용하기 어렵습니다.

## Classification Problems

Let's start with an admittedly somewhat contrived image problem where the input image has a height and width of 2 pixels and the color is grayscale. Thus, each pixel value can be represented by a scalar. We record the four pixels in the image as $x_1, x_2, x_3, x_4$. We assume that the actual labels of the images in the training data set are "cat", "chicken" or "dog" (assuming that the three animals can be represented by 4 pixels).

입력 이미지의 높이와 넓이가 2 픽셀이고, 색은 회색인 이미지를 입력으로 다루는 간단한 문제부터 시작해보겠습니다. 이미지의 4개 픽셀의 값은  $x_1, x_2, x_3, x_4$ 으로 표현하고, 각 이미지의 실제 label는 "고양이", "닭", "강아지" 중에 하나로 정의되어 있다고 하겠습니다. (4 픽셀로 구성된 이미지가 3개 동물 중에 어떤 것인지를 구별할 수 있다고 가정합니다.)

To represent these labels we have two choices. Either we set $y \in \{1, 2, 3\}$, where the integers represent {dog, cat, chicken} respectively. This is a great way of *storing* such information on a computer. It would also lend itself rather neatly to regression, but the ordering of outcomes imposes some quite unnatural ordering. In our toy case, this would presume that cats are more similar to chickens than to dogs, at least mathematically. It doesn't work so well in practice either, which is why statisticians invented an alternative approach: one hot encoding via

이 label들을 표현하는데 두가지 방법이 있습니다. 첫번째 방법은  {강아지, 고양이, 닭}을 각각  $y \in \{1, 2, 3\}$ 으로 정의합니다. 이 방법은 컴퓨터에 정보를 저장하는 좋은 방법이지만, 이 방법은 회귀 문제에 적합합니다. 더구나 이 숫자들의 순서가 분류의 문제에서는 의미가 없습니다. 우리의 간단한 예제에서는 적어도 수학적으로는 고양이가 강아지보다는 닭과 더 비슷하다는 것을 의미할 수도 있게됩니다. 하지만, 실제 문제들에서 이런 비교가 잘되지 않습니다. 그렇기 때문에, 통계학자들은 one hot encoding 을 통해서 표현하는 방법을 만들었습니다.

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}$$

That is, $y$ is viewed as a three-dimensional vector, where $(1,0,0)$ corresponds to "cat", $(0,1,0)$ to "chicken" and $(0,0,1)$ to "dog".

즉, $y$ 는 3차원 백터로 (1,0,0)은 고양이를, (0,1,0)은 닭은, (0,0,1)은 강아지를 의미합니다.

### Network Architecture

If we want to estimate multiple classes, we need multiple outputs, matching the number of categories. This is one of the main differences to regression. Because there are 4 features and 3 output animal categories, the weight contains 12 scalars ($w$ with subscripts) and the bias contains 3 scalars ($b$ with subscripts). We compute these three outputs, $o_1, o_2$, and $o_3$, for each input:

여러 클래스들에 대한 분류를 예측할 때는 카테고리 개수와 같은 수의 output들이 필요합니다. 이점이 회귀 문제와 가장 다른 점입니다. 4개 feature들과 3개의 동물 카테고리 output들이 있으니, weight($w$)는 12개의 scalar들로 구성되고 bias ($b$)는 3개의 scalar로 정의됩니다. 각 입력에 대해서 3개의 output ($o1, o2, o3$)는 다음과 같이 계산됩니다.
$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{21} + x_3 w_{31} + x_4 w_{41} + b_1,\\
o_2 &= x_1 w_{12} + x_2 w_{22} + x_3 w_{32} + x_4 w_{42} + b_2,\\
o_3 &= x_1 w_{13} + x_2 w_{23} + x_3 w_{33} + x_4 w_{43} + b_3.
\end{aligned}
$$

The neural network diagram below depicts the calculation above.  Like linear regression, softmax regression is also a single-layer neural network.  Since the calculation of each output, $o_1, o_2$, and $o_3$, depends on all inputs, $x_1$, $x_2$, $x_3$, and $x_4$, the output layer of the softmax regression is also a fully connected layer.

아래 neural network 다이어그램은 위 연산을 표현하고 있습니다. 선형 회귀처럼, softmax regression은 단일 계층의 뉴럴 네트워크로 구성됩니다. output ($o1, o2, o3$) 는 모든 input ($x1, x2, x3, x4$) 값들과 연관되서 계산되기 때문에, softmax regression은 output 래이어는 fully connected 래이어입니다.

![Softmax regression is a single-layer neural network.  ](../img/softmaxreg.svg)


### Softmax Operation

The chosen notation is somewhat verbose. In vector form we arrive at $\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$, which is much more compact to write and code. Since the classification problem requires discrete prediction output, we can use a simple approach to treat the output value $o_i$ as the confidence level of the prediction category $i$. We can choose the class with the largest output value as the predicted output, which is output $\operatorname*{argmax}_i o_i$. For example, if $o_1$, $o_2$, and $o_3$ are 0.1, 10, and 0.1, respectively, then the prediction category is 2, which represents "chicken".

위 표기법은 다소 장황해보입니다. 이를 백터 표현으로 하면  $\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$ 와 같이 쓰기도 간단하고 코딩하기도 간단합니다. 하지만, 분류 문제는 discrete 예측 결과가 필요하기 때문에, $i$ 번째 카테고리에 대한 confidence 레벨을 표현하기 위해서 output 을 $o_i$ 로 표현하는 간단한 방법을 사용합니다. 이렇게 구성하면, 어떤 카테고리에 속하는지를 결과 값들 중에 가장 큰 값의 클래스로 선택하면 되고,  $\operatorname*{argmax}_i o_i$ 로 간단히 계산할 수 있습니다. 예를 들면, 결과 $o1, o2, o3$ 가 각 각 0.1, 10, 0.1 이라면, 예측된 카테고리는 2, 즉 "닭"이 됩니다.

However, there are two problems with using the output from the output layer directly. On the one hand, because the range of output values from the output layer is uncertain, it is difficult for us to visually judge the meaning of these values. For instance, the output value 10 from the previous example indicates a level of "very confident" that the image category is "chicken". That is because its output value is 100 times that of the other two categories.  However, if $o_1=o_3=10^3$, then an output value of 10 means that the chance for the image category to be chicken is very low.  On the other hand, since the actual label has discrete values, the error between these discrete values and the output values from an uncertain range is difficult to measure.

하지만, output 래이어의 값을 직접 사용하기에는 두 가지 문제가 있습니다. 첫번째는 output 값의 범위가 불확실해서, 시각적으로 이 값들의 의미를 판단하기 어렵다는 것입니다. 예를 들어, 이전 예에서 결과 10은 주어진 이미지가 "닭" 카테고리에 속할 것이라고 "매우 확신"한다는 것을 의미합니다. 왜냐하면, 다른 두 카테고리들의 값보다 100배 크기 때문입니다. 만약에 $o_1=o_3=10^3$ 이라면, 10이라는 output 값은 이미지가 "닭" 카테고리에 속할 가능성이 매우 낮다는 것의 의미하게 됩니다. 두번째 문제는 실제 label은 discrete 값을 갖기 때문에, 불특정 범위을 갖는 output 값과 label 값의 오류를 측정하는 것이 매우 어렵다는 것입니다.

We could try forcing the outputs to correspond to probabilities, but there's no guarantee that on new (unseen) data the probabilities would be nonnegative, let alone sum up to 1. For this kind of discrete value prediction problem, statisticians have invented classification models such as (softmax) logistic regression. Unlike linear regression, the output of softmax regression is subjected to a nonlinearity which ensures that the sum over all outcomes always adds up to 1 and that none of the terms is ever negative. The nonlinear transformation works as follows:

output 값들이 확률값으로 나오도록 해볼 수 있겠지만, 새로운 데이터가 주어졌을 때 확률값이 0 또는 양수이고, 전체 합이 1이 된다는 것을 보장할 수는 없습니다. 이런 discrete value 예측 문제를 다루기 위해서 통계학자들은 (softmax) logistic regression이라는 분류 모델을 만들었습니다. 선형 회귀(linear regression)과는 다르게, softmax regression의 결과는 모든 결과값들의 합이 1이 되도록 하는 비선형성에 영향을 받고, 각 결과 값는 0 또는 양수값을 갖습니다. 비선형 변환은 다음 공식으로 이뤄집니다.
$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \text{ where }
\hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}
$$

It is easy to see $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$ with $0 \leq \hat{y}_i \leq 1$ for all $i$. Thus, $\hat{y}$ is a proper probability distribution and the values of $o$ now assume an easily quantifiable meaning. Note that we can still find the most likely class by

모든 $i$ 에 대해서  $0 \leq \hat{y}_i \leq 1$  이고  $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$ 를 만족하는 것을 쉽게 확인할 수 있습니다. 따라서, $\hat{y}$ 은 적절한 확률 분포이고, $o$ 값은 쉽게 측정할 수 있는 값으로 간주할 수 있습니다. 아래 공식은 가장 가능성 있는 클래스를 찾아줍니다.
$$
\hat{\imath}(\mathbf{o}) = \operatorname*{argmax}_i o_i = \operatorname*{argmax}_i \hat y_i
$$

So, the softmax operation does not change the prediction category output but rather it gives the outputs $\mathbf{o}$ proper meaning. Summarizing it all in vector notation we get ${\mathbf{o}}^{(i)} = \mathbf{W} {\mathbf{x}}^{(i)} + {\mathbf{b}}$ where ${\hat{\mathbf{y}}}^{(i)} = \mathrm{softmax}({\mathbf{o}}^{(i)})$.

즉, softmax 연산은 예측하는 카테고리의 결과를 바꾸지 않으면서, 결과 $o$ 에 대한 적절한 의미를 부여해줍니다. 이것을 백터 표현법으로 요약해보면, get ${\mathbf{o}}^{(i)} = \mathbf{W} {\mathbf{x}}^{(i)} + {\mathbf{b}}$,  ${\hat{\mathbf{y}}}^{(i)} = \mathrm{softmax}({\mathbf{o}}^{(i)})$ 이 됩니다.


### Vectorization for Minibatches

To improve computational efficiency further, we usually carry out vector calculations for mini-batches of data. Assume that we are given a mini-batch $\mathbf{X}$ of examples with dimensionality $d$ and batch size $n$. Moreover, assume that we have $q$ categories (outputs). Then the minibatch features $\mathbf{X}$ are in $\mathbb{R}^{n \times d}$, weights $\mathbf{W} \in \mathbb{R}^{d \times q}$ and the bias satisfies $\mathbf{b} \in \mathbb{R}^q$.

연산 효율을 더 높이기 위해서, 데이터의 미니 배치에 대한 연산을 백터화합니다. 차원이 $d$ 이고 배치 크기가 $n$ 인 데이터들의 미니 배치  $\mathbf{X}$ 가 있고, 결과로 $q$ 개의 카테고리가 있다고 가정하겠습니다. 그러면, 미니 배치 feature  $\mathbf{X}$ 는  $\mathbb{R}^{n \times d}$ 에 속하고, weight들 $\mathbf{W}$ 는 $\mathbb{R}^{d \times q}$ 에, bias  $\mathbf{b}$ 는 $\mathbb{R}^q$ 에 속합니다.
$$
\begin{aligned}
\mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b} \\
\hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O})
\end{aligned}
$$

This accelerates the dominant operation: $\mathbf{W} \mathbf{X}$ from a matrix-vector to a matrix-matrix product. The softmax itself can be computed by exponentiating all entries in $\mathbf{O}$ and then normalizing them by the sum appropriately.

이렇게 정의하면 가장 많이 차지하는 연산을 가속화할 수 있습니다. 즉, : $\mathbf{W} \mathbf{X}$ 이 형렬-백터의 곱에서 행렬-행렬의 곱으로 변환됩니다. softmax는 결과  $\mathbf{O}$ 의 모든 항목에 지수 함수를 적용하고, 지수 함수들의 값의 합으로 normalize 하는 것으로 계산됩니다.

## Loss Function

Now that we have some mechanism for outputting probabilities, we need to transform this into a measure of how accurate things are, i.e. we need a *loss function*. For this we use the same concept that we already encountered in linear regression, namely likelihood maximization.

확률 결과를 출력하는 방법을 정의했으니, 이 값이 얼마나 정확한지를 측정하는 값으로 변환하는 것이 필요합니다. 즉, loss 함수가 필요합니다. 선형 회귀에서 사용했던 것과 동일한 개념을 사용하는데, 이는 likelihood maxmization이라고 합니다.

### Log-Likelihood

The softmax function maps $\mathbf{o}$ into a vector of probabilities corresponding to various outcomes, such as $p(y=\mathrm{cat}|\mathbf{x})$. This allows us to compare the estimates with reality, simply by checking how well it predicted what we observe.

softmax 함수는 결과  $\mathbf{o}$ 를 여러 결과들에 대한 확률, $p(y=\mathrm{cat}|\mathbf{x})$, 들의 백터로 변환합니다. 이는, 예측된 값이 얼마나 잘 예측하고 있는지를 확인하는 것으로 실제 값과 예측 결과에 대한 비교를 할 수 있습니다.
$$
p(Y|X) = \prod_{i=1}^n p(y^{(i)}|x^{(i)})
\text{ and thus }
-\log p(Y|X) = \sum_{i=1}^n -\log p(y^{(i)}|x^{(i)})
$$

Minimizing $-\log p(Y|X)$ corresponds to predicting things well. This yields the loss function (we dropped the superscript $(i)$ to avoid notation clutter):

잘 예측하는 것은 $-\log p(Y|X)$ 를 최소화하는 것을 의미합니다. 이를 통해서 loss 함수를 다음과 같이 정의할 수 있습니다. (표기를 간단하게 하기 위해서 $i$ 는 제외했습니다.)
$$
l = -\log p(y|x) = - \sum_j y_j \log \hat{y}_j
$$

Here we used that by construction $\hat{y} = \mathrm{softmax}(\mathbf{o})$ and moreover, that the vector $\mathbf{y}$ consists of all zeroes but for the correct label, such as $(1, 0, 0)$. Hence the the sum over all coordinates $j$ vanishes for all but one term. Since all $\hat{y}_j$ are probabilities, their logarithm is never larger than $0$. Consequently, the loss function is minimized if we correctly predict $y$ with *certainty*, i.e. if $p(y|x) = 1$ for the correct label.

여기서  $\hat{y} = \mathrm{softmax}(\mathbf{o})$ 이고, 백터 $\mathbf{y}$ 는 해당하는 label이 아닌 위치에는 모두 0을 갖습니다. (예를 들면 (1,0,0)). 따라서, 모든 $j$ 에 대한 합을 하면, 하나의 항목만 남게됩니다. 모든 $\hat{y}_j$ 는 확률값이기 때문에, 이에 대한 logarithm 값은 0보다 커질 수 없습니다. 그 결과, 주어진 x에 대해서 y를 잘 예측하는 경우라면 (즉,  $p(y|x) = 1$), loss 함수는 최소화될 것입니다.

### Softmax and Derivatives

Since the Softmax and the corresponding loss are so common, it is worth while understanding a bit better how it is computed. Plugging $o$ into the definition of the loss $l$ and using the definition of the softmax we obtain:

Softmax와 이에 대한 loss는 많이 사용되기 때문에, 어떻게 계산되는지 자세히 살펴볼 필요가 있습니다.  $o$ 를 loss $l$ 의 정의에 대입하고, softmax의 정의를 이용하면, 다음과 같이 표현을 얻습니다.
$$
l = -\sum_j y_j \log \hat{y}_j = \sum_j y_j \log \sum_k \exp(o_k) - \sum_j y_j o_j
= \log \sum_k \exp(o_k) - \sum_j y_j o_j
$$

To understand a bit better what is going on, consider the derivative with respect to $o$. We get

어떤 일이 일어나는지 더 살펴보기 위해서, loss 함수를 $o$ 에 대해서 미분을 해보면 아래 공식을 유도할 수 있습니다.
$$
\partial_{o_j} l = \frac{\exp(o_j)}{\sum_k \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j = \Pr(y = j|x) - y_j
$$

In other words, the gradient is the difference between what the model thinks should happen, as expressed by the probability $p(y|x)​$, and what actually happened, as expressed by $y​$. In this sense, it is very similar to what we saw in regression, where the gradient was the difference between the observation $y​$ and estimate $\hat{y}​$. This seems too much of a coincidence, and indeed, it isn't. In any [exponential family](https://en.wikipedia.org/wiki/Exponential_family) model the gradients of the log-likelihood are given by precisely this term. This fact makes computing gradients a lot easier in practice.

다르게 설명해보면, gradient 는 모델이  $p(y|x)$ 확률 표현식으로 예측한 것과 실제 값  $y_j$ 의 차이입니다. 이는 회귀 문제에서 보았던 것과 아주 비슷합니다. 회귀 문제에서 gradient가 관찰된 실제 값 $y$ 와 예측된 값 $\hat{y}$ 의 차이로 계산되었습니다. 이는 너무 우연으로 보이는데, 사실은 그렇지 않습니다. [exponential 계열](https://en.wikipedia.org/wiki/Exponential_family)의 모델의 경우에는, log-likelihood 의 gradient는 정확하게 이 항목으로 주어집니다. 이로 인해서 gradient를 구하는 것이 실제 적용할 때 매우 간단해집니다.

### Cross-Entropy Loss

Now consider the case where we don't just observe a single outcome but maybe, an entire distribution over outcomes. We can use the same representation as before for $y$. The only difference is that rather than a vector containing only binary entries, say $(0, 0, 1)$, we now have a generic probability vector, say $(0.1, 0.2, 0.7)$. The math that we used previously to define the loss $l$ still works out fine, just that the interpretation is slightly more general. It is the expected value of the loss for a distribution over labels.

자 이제는 하나의 결과에 대한 관찰을 하는 경우가 아니라, 결과들에 대한 전체 분포를 다루는 경우를 생각해봅시다.  $y$ 에 대한 표기를 이전과 동일하게 사용할 수 있습니다. 오직 다른 점은 (0,0,1) 과 같이 binary 값을 갖는 것이 아니라 (0.1, 0.2, 0.7)과 같이 일반적인 확률 백터를 사용한다는 것입니다. loss $l$ 의 정의도 동일한 수학을 사용하지만, 이에 대한 해석은 조금 더 일반적입니다. label들의 분포에 대한 loss의 기대값을 의미합니다.
$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_j y_j \log \hat{y}_j
$$

This loss is called the cross-entropy loss. It is one of the most commonly used ones for multiclass classification. To demystify its name we need some information theory. The following section can be skipped if needed.

이렇게 정의된 loss는 cross-entropy loss라고 부릅니다. 이것은 다중 클래스 분류에 가장 흔히 사용되는 loss 입니다. 이 이름에 대해서 알아보기 위해서는 information theory에 대한 설명이 필요하며, 지금부터 설명하겠습니다. 다음 내용은 넘어가도 됩니다.

## Information Theory Basics

Information theory deals with the problem of encoding, decoding, transmitting and manipulating information (aka data), preferentially in as concise form as possible.

Information theory는 정보 (또는 데이터)를 가능한 한 간결한 형식으로 인코딩, 디코딩, 전송, 및 변조하는 문제를 다룹니다.

### Entropy

A key concept is how many bits of information (or randomness) are contained in data. It can be measured as the [entropy](https://en.wikipedia.org/wiki/Entropy) of a distribution $p$ via

데이터 (또는 난수)에 몇개의 정보 비트들이 담겨있는지가 중요한 개념입니다. 이는 분표 $p$ 의 [entropy](https://en.wikipedia.org/wiki/Entropy)로 다음과 같이 수치화할 수 있습니다.
$$
H[p] = \sum_j - p(j) \log p(j)
$$

One of the fundamental theorems of information theory states that in order to encode data drawn randomly from the distribution $p$ we need at least $H[p]$ 'nats' to encode it. If you wonder what a 'nat' is, it is the equivalent of bit but when using a code with base $e$ rather than one with base 2. One nat is $\frac{1}{\log(2)} \approx 1.44$ bit. $H[p] / 2$ is often also called the binary entropy.

정보 이론의 근본적인 이론중에 하나로 분포 $p$ 로부터 임의로 추출된 데이터를 인코드하기 위해서는 최소  $H[p]$ 개의 'nat'이 필요하다는 것이 있습니다. 여기서 'nat'은 비트와 동일하나, base 2가 아니라 base $e$ 를 이용합니다. 즉, 1 nat은 $\frac{1}{\log(2)} \approx 1.44$  비트이고,  $H[p] / 2$ 는 종종 binary entropy라고 불립니다.

To make this all a bit more theoretical consider the following: $p(1) = \frac{1}{2}$ whereas $p(2) = p(3) = \frac{1}{4}$. In this case we can easily design an optimal code for data drawn from this distribution, by using `0` to encode 1, `10` for 2 and `11` for 3. The expected number of bit is $1.5 = 0.5 * 1 + 0.25 * 2 + 0.25 * 2$. It is easy to check that this is the same as the binary entropy $H[p] / \log 2$.

조금 더 이론적으로 들어가보겠습니다. $p(1) = \frac{1}{2}$ 이고,  $p(2) = p(3) = \frac{1}{4}$ 인 분포를 가정하겠습니다. 이 경우, 이 분포에서 추출한 데이터에 대한 최적의 코드를 굉장히 쉽게 설계할 수 있습니다. 즉, 1의 인코딩은 `0`, 2와 3에 대한 인코딩은 각 각 `10`, `11` 로 정의하면 됩니다. 예상되는 비트 개수는  $1.5 = 0.5 * 1 + 0.25 * 2 + 0.25 * 2$ 이고, 이 숫자는 binary entropy $H[p] / \log 2$ 와 같다는 것을 쉽게 확인할 수 있습니다.

### Kullback Leibler Divergence

One way of measuring the difference between two distributions arises directly from the entropy. Since $H[p]$ is the minimum number of bits that we need to encode data drawn from $p$, we could ask how well it is encoded if we pick the 'wrong' distribution $q$. The amount of extra bits that we need to encode $q$ gives us some idea of how different these two distributions are. Let us compute this directly - recall that to encode $j$ using an optimal code for $q$ would cost $-\log q(j)$ nats, and we need to use this in $p(j)$ of all cases. Hence we have

두 분포간에 차이를 측정하는 방법 중에 하나로 entropy를 이용하는 방법이 있습니다. $H[p]$ 는 분포 $p$를 따르는 데이터를 인코드하는데 필요한 최소 비트 수를 의미하기 때문에, 틀린 분포 $q$ 에서 뽑았을 때 얼마나 잘 인코딩이 되었는지를 물어볼 수 있습니다. $q$ 를 인코딩하는데 추가로 필요한 비트 수는 두 분표가 얼마나 다른지에 대한 아이디어를 제공합니다. 직접 계산해보겠습니다. 분포 $q$ 에 대해 최적인 코드를 이용해서 $j$ 를 인코딩하기 위해서는 $-\log q(j)$ nat이 필요하고,  $p(j)$ 인 모든 경우에서 이를 사용하면, 다음 식을 얻습니다.
$$
D(p\|q) = -\sum_j p(j) \log q(j) - H[p] = \sum_j p(j) \log \frac{p(j)}{q(j)}
$$

Note that minimizing $D(p\|q)$ with respect to $q$ is equivalent to minimizing the cross-entropy loss. This can be seen directly by dropping $H[p]$ which doesn't depend on $q$. We thus showed that softmax regression tries the minimize the surprise (and thus the number of bits) we experience when seeing the true label $y$ rather than our prediction $\hat{y}$.

$q$ 에 대해서  $D(p\|q)$ 를 최소화하는 것은 cross-entropy loss를 최소화하는 것과 같습니다. 이는 $q$ 에 의존하지 않는 $H[p]$ 를 빼버리면 바로 얻을 수 있습니다. 이를 통해서 우리는 softmax regression은 예측된 값  $\hat{y}$ 이 아니라 실제 label $y$ 를 봤을 때 얻는 놀라움(비트 수)을 최소화하려는 것임을 증명했습니다.

## Model Prediction and Evaluation

After training the softmax regression model, given any example features, we can predict the probability of each output category. Normally, we use the category with the highest predicted probability as the output category. The prediction is correct if it is consistent with the actual category (label). In the next part of the experiment, we will use accuracy to evaluate the model’s performance. This is equal to the ratio between the number of correct predictions and the total number of predictions.

학습된 softmax regression 모델을 사용하면, 새로운  feature가 주어졌을 때, 각 output 카테고리에 속할 확률값을 예측할 수 있습니다. 일반적으로는 가장 크게 예측된 확률값을 갖는 카테고리를 결과 카테고리라고 정의합니다. 실제 카테고리 (label)와 일치하는 경우에 예측이 정확하다고 합니다. 다음에는 모델의 성능을 평가하는 방법으로 accuracy 정확도를 사용할 예정입니다. 이는 정확하게 예측한 개수와 전체 예측의 개수의 비율과 같습니다. 

## Summary

* We introduced the softmax operation which takes a vector maps it into probabilities.
* Softmax regression applies to classification problems. It uses the probability distribution of the output category in the softmax operation.
* Cross entropy is a good measure of the difference between two probability distributions. It measures the number of bits needed to encode the data given our model.
* 벡터를 확률로 변환하는 softmax 연산을 알아봤습니다.
* softmax regression은 분류의 문제에 적용할 수 있습니다. softmax 연산을 이용해서 얻은 결과 카테고리의 확률 분포를 이용합니다.
* cross entropy는 두 확률 분포의 차이를 측정하는 좋은 방법입니다. 이는 주어진 모델이 데이터를 인코드하는데 필요한 비트 수를 나타냅니다.

## Problems

1. Show that the Kullback-Leibler divergence $D(p\|q)$ is nonnegative for all distributions $p$ and $q$. Hint - use Jensen's inequality, i.e. use the fact that $-\log x$ is a convex function.
1. Show that $\log \sum_j \exp(o_j)$ is a convex function in $o$.
1. We can explore the connection between exponential families and the softmax in some more depth
    * Compute the second derivative of the cross entropy loss $l(y,\hat{y})$ for the softmax.
    * Compute the variance of the distribution given by $\mathrm{softmax}(o)$ and show that it matches the second derivative computed above.
1. Assume that we three classes which occur with equal probability, i.e. the probability vector is $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    * What is the problem if we try to design a binary code for it? Can we match the entropy lower bound on the number of bits?
    * Can you design a better code. Hint - what happens if we try to encode two independent observations? What if we encode $n$ observations jointly?
1. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as $\mathrm{RealSoftMax}(a,b) = \log (\exp(a) + \exp(b))$.
    * Prove that $\mathrm{RealSoftMax}(a,b) > \mathrm{max}(a,b)$.
    * Prove that this holds for $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, provided that $\lambda > 0$.
    * Show that for $\lambda \to \infty$ we have $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a,b)$.
    * What does the soft-min look like?
    * Extend this to more than two numbers.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2334)

![](../img/qr_softmax-regression.svg)
