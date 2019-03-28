# Probability and Statistics

In some form or another, machine learning is all about making predictions.
We might want to predict the *probability* of a patient suffering a heart attack in the next year,
given their clinical history.
In anomaly detection, we might want to assess how *likely* a set of readings from an airplane's jet engine would be,
were it operating normally.
In reinforcement learning, we want an agent to act intelligently in an environment.
This means we need to think about the probability of getting a high reward under each of the available action.
And when we build recommender systems we also need to think about probability.
For example, if we *hypothetically* worked for a large online bookseller,
we might want to estimate the probability that a particular user would buy a particular book, if prompted.
For this we need to use the language of probability and statistics.
Entire courses, majors, theses, careers, and even departments, are devoted to probability.
So our goal here isn't to teach the whole subject.
Instead we hope to get you off the ground,
to teach you just enough that you know everything necessary to start building your first machine learning models
and to have enough of a flavor for the subject that you can begin to explore it on your own if you wish.

머신러닝은 어떤 방식이든지 결국 예측을 수행하는 것입니다. 어떤 환자의 의료 기록을 바탕으로 내년에 심장 마비를 격을 확률 예측하기를 예로 들어볼 수 있습니다. 비정상 탐지를 위해서, 비행기 제트 엔진의 센서 데이터가 정상적으로 동작할 때 어떤 값을 같게될지 예측을 할 수도 있습니다. 강화학습에서는 에이전트가 주어진 환경에서 똑똑하게 동작하게 만드는 것이 목표입니다. 이 경우에는 주어진 행동들 중에 가장 높은 보상을 받는 확률을 고려해야합니다. 추천 시스템을 만드는 경우에도 확률을 고려해야합니다. 예를 들어 여러분이 대형 온라인 서점에서 일은 한다면, 어떤 책을 홍보했을 때 특정 사용자가 그 책을 구매할 확률을 추정하는 것을 하고 싶어할 것입니다. **For this we need to use the language of probability and statistics.
Entire courses, majors, theses, careers, and even departments, are devoted to probability.** 이 책의 목표는 이 모든 주제들에 대해서 배워보는 것은 아니고, 여러분이 스스로 머신러닝 모델을 만들 수 있을 정도의 내용을 알려주고, 이후에 스스로 공부해 볼 수 있는 주제들을 선택할 수 있도록 하는 것입니다.

We've talked a lot about probabilities so far without articulating what precisely they are or giving a concrete example. Let's get more serious by considering the problem of distinguishing cats and dogs based on photographs. This might sound simpler but it's actually a formidable challenge. To start with, the difficulty of the problem may depend on the resolution of the image.

지금까지 확률에 대해서 많이 이야기를 해왔지만, 확률에 정확하게 무엇인지를 설명하지 않았고 구체적인 예제를 들지는 않았습니다. 동물의 사진이 주어졌을 때, 고양이인지 개인지를 구분하는 문제를 조금 자세하게 보겠습니다. 이 문제는 간단해 보이지만, 사실 쉽지 않은 문제가 있습니다. 우선은 문제의 난의도가 이미지의 해상도에 따라 차이가 있을 수 있습니다.

| 10px | 20px | 40px | 80px | 160px |
|:----:|:----:|:----:|:----:|:-----:|
|![](../img/whitecat10.jpg)|![](../img/whitecat20.jpg)|![](../img/whitecat40.jpg)|![](../img/whitecat80.jpg)|![](../img/whitecat160.jpg)|
|![](../img/whitedog10.jpg)|![](../img/whitedog20.jpg)|![](../img/whitedog40.jpg)|![](../img/whitedog80.jpg)|![](../img/whitedog160.jpg)|

While it's easy for humans to recognize cats and dogs at 320 pixel resolution,
it becomes challenging at 40 pixels
and next to impossible at 10 pixels.
In other words, our ability to tell cats and dogs apart at a large distance (and thus low resolution)
might approach uninformed guessing.
Probability gives us a formal way of reasoning about our level of certainty.
If we are completely sure that the image depicts a cat,
we say that the *probability* that the corresponding label $l​$ is $\mathrm{cat}​$,
denoted $P(l=\mathrm{cat})​$ equals 1.0.
If we had no evidence to suggest that $l =\mathrm{cat}​$ or that $l = \mathrm{dog}​$,
then we might say that the two possibilities were equally $likely​$
expressing this as $P(l=\mathrm{cat}) = 0.5​$.
If we were reasonably confident, but not sure that the image depicted a cat,
we might assign a probability $.5  < P(l=\mathrm{cat}) < 1.0​$.

사람이 320 픽셀 해상도의 이미지에서 개와 고양이를 구분하는 것은 쉽습니다. 하지만, 40 픽셀이되면 그 분류가 어렵고, 10픽셀로 줄어들면 거의 불가능합니다. 즉, 개와 고양이를 먼 거리에서 판별하는 것은 (또는 낮은 해상도의 이미지에서) 동전 던지기를 해서 추측하는 것과 동일해집니다. 확률은 확실성에 대한 추론에 대한 공식적인 방법을 제공합니다. 만약, 이미지에 고양이가 있다는 것을 완벽하게 확신한다면, 해당 레이블 $l$ 이 고양이일 확률,  $P(l=\mathrm{cat})$ 는 1.0이라고 말합니다. 만약  $l =\mathrm{cat}$ 인지 $l = \mathrm{dog}$ 에 대한 아무런 판단을 못한다면, 두 확률은 동일하다고 하다고 말하며, $P(l=\mathrm{cat}) = 0.5$ 이 됩니다. 만약 이미지에 고양이가 있다는 것을 확실하지는 안지만 어느 정도 확신한다면, 확률은 $.5  < P(l=\mathrm{cat}) < 1.0$ 로 주어질 것입니다.

Now consider a second case:
given some weather monitoring data,
we want to predict the probability that it will rain in Taipei tomorrow.
If it's summertime, the rain might come with probability $.5​$
In both cases, we have some value of interest.
And in both cases we are uncertain about the outcome.
But there's a key difference between the two cases.
In this first case, the image is in fact either a dog or a cat,
we just don't know which.
In the second case, the outcome may actually be a random event,
if you believe in such things (and most physicists do).
So probability is a flexible language for reasoning about our level of certainty,
and it can be applied effectively in a broad set of contexts.

이제 두번째 예를 들어보겠습니다. 대만 날씨에 대한 데이터를 관찰한 데이터가 있을 때, 내일 비가 내릴 확률을 예측하고자 합니다. 여름인 경우에는 비가 내릴 확률이 $0.5$ 정도가 될 것입니다.  위 두가지 예제 모두 살펴볼 가치가 있습니다. 두 경우 모두 결과에 대한 불확실성이 있지만, 주요 차이점이 있습니다. 첫번째 예제는 이미지가 고양이인지 개이지만, 우리가 어떤 것인지 모르는 경우이고, 두번째 예제는 결과가 실제로 임의로 일어나는 이벤트일 수도 있습니다. 즉, 확률이란 우리의 활실성에 대한 유론을 하기 위해 유연한 언어리며, 다양한 경우에 효과적으로 적용될 수 있습니다.

## Basic probability theory

Say that we cast a die and want to know
what the chance is of seeing a $1​$
rather than another digit.
If the die is fair, all six outcomes $\mathcal{X} = \{1, \ldots, 6\}​$
are equally likely to occur,
hence we would see a $1​$ in $1​$ out of $6​$ cases.
Formally we state that $1​$ occurs with probability $\frac{1}{6}​$.

주사위를 던저서 다른 숫자가 아닌 1일 나오는 확률이 얼마나 되는지 찾는 경우를 생각해보겠습니다. 주사위가 공정하다면, 모든 6개 숫자들,  $\mathcal{X} = \{1, \ldots, 6\}$, 은 일어날 가능성이 동일합니다. 학술 용어로는 "1은 확률 $\frac{1}{6}$ 로 일어난다"라고 말합니다.

For a real die that we receive from a factory,
we might not know those proportions
and we would need to check whether it is tainted.
The only way to investigate the die is by casting it many times
and recording the outcomes.
For each cast of the die,
we'll observe a value $\{1, 2, \ldots, 6\}​$.
Given these outcomes, we want to investigate the probability of observing each outcome.

공장에서 막 만들어진 주사위에 대해서 우리는 이 비율을 알지 못할 수 있고, 주사위가 공정한지 확인해야할 필요가 있습니다. 주사위를 조사하는 유일한 방법은 여러번 던져보면서 결과를 기록하는 것입니다. 주사위를 던질 때마다, 우리는 $\{1, 2, \ldots, 6\}$에 하나의 숫자를 얻게되고, 이 결과들에 주어지면, 각 숫자들이 일어날 수 있는 확률을 조사할 수 있습니다.

One natural approach for each value is to take the individual count for that value
and to divide it by the total number of tosses.
This gives us an *estimate* of the probability of a given event.
The law of large numbers tell us that as the number of tosses grows this estimate will draw closer and closer to the true underlying probability.
Before going into the details of what's going here, let's try it out.

가장 자연스러운 방법은 각 숫자들이 나온 횟수를 전체 던진 횟수로 나누는 것입니다. 이를 통해서 우리는 특정 이벤트에 대한 확률을 *추정* 합니다. 큰 수의 법칙(the law of large numbers)에 따라, 던지는 횟수가 늘어날 수록 이 추정은 실제 확률과 계속 가까워집니다. 더 자세한 논의를 하기전에, 실제로 실험을 해보겠습니다.

To start, let's import the necessary packages:

우선 필요한 패키지들을 import 합니다.

```{.python .input}
import mxnet as mx
from mxnet import nd
```

Next, we'll want to be able to cast the die.
In statistics we call this process of drawing examples from probability distributions *sampling*.
The distribution which assigns probabilities to a number of discrete choices is called
the *multinomial* distribution.
We'll give a more formal definition of *distribution* later,
but at a high level, think of it as just an assignment of probabilities to events.
In MXNet, we can sample from the multinomial distribution via the aptly named `nd.random.multinomial` function.
The function can be called in many ways, but we'll focus on the simplest.
To draw a single sample, we simply pass in a vector of probabilities.

다음으로는 주사위를 던지는 것을 해야합니다. 통계에서는 확률 분포에서 샘플을 뽑는 것을 *샘플링* 이라고 합니다. 연속되지 않은 선택들에 확률이 부여된 분포를 우리는 *다항(multinomial)* 분포라고 합니다. *분포(distribution)* 에 대한 공식적인 정의는 다음에 다루겠고, 지금은 분포를 이벤트들에 확률을 할당하는 것 정도로 생각하겠습니다. MXNet에서 `nd.random.multinomial` 함수를 이용하면 다항 분포에서 샘플을 추출할 수 있습니다.

```{.python .input}
probabilities = nd.ones(6) / 6
nd.random.multinomial(probabilities)
```

If you run the sampler a bunch of times,
you'll find that you get out random values each time.
As with estimating the fairness of a die,
we often want to generate many samples from the same distribution.
It would be really slow to do this with a Python `for` loop,
so `random.multinomial` supports drawing multiple samples at once,
returning an array of independent samples in any shape we might desire.

여러 샘플을 뽑아보면, 매번 임의의 숫자를 얻는 것을 확인할 수 있습니다. 주사위의 공정성을 추정하는 예제에서 우리는 같은 분포에서 많은 샘플을 추출하기를 원합니다. Python의 `for` loop을 이용하면 너무 느리기 때문에, `random.multinomial` 이 여러 샘플을 한번째 뽑아주는 기능을 이용해서 우리가 원하는 shape의 서로 연관이 없는 샘플들의 배열을 얻겠습니다.

```{.python .input}
print(nd.random.multinomial(probabilities, shape=(10)))
print(nd.random.multinomial(probabilities, shape=(5,10)))
```

Now that we know how to sample rolls of a die,
we can simulate 1000 rolls. We can then go through and count, after each of the 1000 rolls,
how many times each number was rolled.

이제 주사위를 던지는 샘플을 구하는 방법을 알았으니, 100번 주사위를 던지는 시뮬레이션을 해서, 각 숫자들이 나온 횟수를 카운팅합니다.

```{.python .input}
rolls = nd.random.multinomial(probabilities, shape=(1000))
counts = nd.zeros((6,1000))
totals = nd.zeros(6)
for i, roll in enumerate(rolls):
    totals[int(roll.asscalar())] += 1
    counts[:, i] = totals
```

To start, we can inspect the final tally at the end of $1000$ rolls.

1000번을 던져본 후에 최종 합계를 확인합니다.

```{.python .input}
totals / 1000
```

As you can see, the lowest estimated probability for any of the numbers is about $.15$
and the highest estimated probability is $0.188$.
Because we generated the data from a fair die,
we know that each number actually has probability of $1/6$, roughly $.167$,
so these estimates are pretty good.
We can also visualize how these probabilities converge over time
towards reasonable estimates.

결과에 따르면, 모든 숫자 중에 가장 낮계 추정된 확률은 약 $0.15$ 이고, 가장 높은 추정 확률은 $0.188$ 입니다. 공정한 주사위를 사용해서 데이터를 생성했기 때문에, 각 숫자들은  $1/6$ 즉 $0.167$ 의 확률을 갖는다는 것을 알고 있고, 예측도 매우 좋게 나왔습니다. 이 확률이 시간이 지나면서 의미있는 추정치로 어떻게 수렴하는지를 시각해볼 수도 있습니다.

To start let's take a look at the `counts`
array which has shape `(6, 1000)`.
For each time step (out of 1000),
`counts` says how many times each of the numbers has shown up.
So we can normalize each $j​$-th column of the counts vector by the number of tosses
to give the `current` estimated probabilities at that time.
The counts object looks like this:

이를 위해서 우선은  `(6, 1000)` 의 shape을 갖는 `counts` 배열을 살펴봅시다. 1000번을 수행하는 각 단계마다, `counts` 는 각 숫자가 몇번 나왔는지를 알려줍니다. 그렇다면, `counts` 배열의 $j$ 번째 열의 그때까지 던진 총 횟수로 표준화해서, 그 시점에서의 추정 확률 `current` 를 계산합니다. `counts` 객체는 다음과 같습니다.

```{.python .input}
counts
```

Normalizing by the number of tosses, we get:

던진 총 횟수로 표준화면,

```{.python .input}
x = nd.arange(1000).reshape((1,1000)) + 1
estimates = counts / x
print(estimates[:,0])
print(estimates[:,1])
print(estimates[:,100])
```

As you can see, after the first toss of the die, we get the extreme estimate that one of the numbers will be rolled with probability $1.0​$ and that the others have probability $0​$. After $100​$ rolls, things already look a bit more reasonable.
We can visualize this convergence by using the plotting package `matplotlib`. If you don't have it installed, now would be a good time to [install it](https://matplotlib.org/).

결과에서 보이듯이, 주사위를 처음 던진 경우 하나의 숫자에 대한 확률이 $1.0$ 이고 나머지 숫자들에 대한 확률이 $0$ 인 심한 에측을 하지만, 100번을 넘어서면 결과가 상당히 맞아보입니다. 플롯을 그리는 패키지 `matplotlib` 을 이용해서 이 수렴 과정을 시각화해봅니다. 이 패키지를 아직 설치하지 않았다면, [install it](https://matplotlib.org/) 를 참고해서 지금하세요.

```{.python .input}
%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

plt.figure(figsize=(8, 6))
for i in range(6):
    plt.plot(estimates[i, :].asnumpy(), label=("P(die=" + str(i) +")"))

plt.axhline(y=0.16666, color='black', linestyle='dashed')
plt.legend()
plt.show()
```

Each solid curve corresponds to one of the six values of the die
and gives our estimated probability that the die turns up that value
as assessed after each of the 1000 turns.
The dashed black line gives the true underlying probability.
As we get more data, the solid curves converge towards the true answer.

각 선은 주사위의 숫자 중에 하나를 의미하고, 1000번 주사위 던지기를 수행하면서 각 횟수마다 각 숫자가 나올 확률의 추정값을 나타내는 그리프입니다. 검은 점선은 진짜 확률(true probability, $1/6$)을 표시합니다. 회수가 늘어가면 선들이 진짜 확률에 수렴하고 있습니다.

In our example of casting a die, we introduced the notion of a **random variable**.
A random variable, which we denote here as $X​$ can be pretty much any quantity and is not deterministic.
Random variables could take one value among a set of possibilities.
We denote sets with brackets, e.g., $\{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit}\}​$.
The items contained in the set are called *elements*,
and we can say that an element $x​$ is *in* the set S, by writing $x \in S​$.
The symbol $\in​$ is read as "in" and denotes membership.
For instance, we could truthfully say $\mathrm{dog} \in \{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit}\}​$.
When dealing with the rolls of die, we are concerned with a variable $X \in \{1, 2, 3, 4, 5, 6\}​$.

주사위 던지기 예를 통해서 **확률 변수(random variable)** 이라는 개념을 소개했습니다. 여기서  $X$ 로 표현할 확률 변수는 어떤 양이 될 수 있고, 결정적이지 않을 수 있습니다. 확률 변수는 여러 가능성들의 집합에서 하나의 값을 나타낼 수도 있습니다. 집합은 괄호를 이용해서 표현합니다. 예를 들면,  $\{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit}\}$ 입니다. 집합에 속한 아이템들은 *원소(element)* 라고 하고, 어떤 원소 $x$ 가 집합 $S$ 에 *속한다* 라고 하면 표기는 $x \in S$ 로 합니다. 기호 $\in$ 는 "속한다"라고 읽고, 포함 관계를 표현합니다. 예를 들어,   $\mathrm{dog} \in \{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit}\}$ 입니다. 주사위 던지는 것의 경우, 확률 변수 $X \in \{1, 2, 3, 4, 5, 6\}$ 입니다

Note that there is a subtle difference between discrete random variables, like the sides of a dice,
and continuous ones, like the weight and the height of a person.
There's little point in asking whether two people have exactly the same height.
If we take precise enough measurements you'll find that no two people on the planet have the exact same height.
In fact, if we take a fine enough measurement,
you will not have the same height when you wake up and when you go to sleep.
So there's no purpose in asking about the probability
that someone is $2.00139278291028719210196740527486202$ meters tall.
Given the world population of humans the probability is virtually 0.
It makes more sense in this case to ask whether someone's height falls into a given interval,
say between 1.99 and 2.01 meters.
In these cases we quantify the likelihood that we see a value as a *density*.
The height of exactly 2.0 meters has no probability, but nonzero density.
In the interval between any two different heights we have nonzero probability.

연속적이지 않은 확률변수(예를 들어 주사위의 6면)과 연속적인 확률변수(예를 들어 사람의 몸무게나 키) 사이에는 미묘한 차이점이 있다는 것을 기억하세요. 두 사람의 키가 정확하게 같은지를 묻는 경우는 드물 것입니다. 아주 정확한 측정 방법이 있어서 이를 적용한다면, 이 세상에 키가 완전하게 같은 사람 두사람이 없습니다. 사실, 적당히 정교한 측정을 하는 경우에도 아침에 일어났을 때의 키와 밤에 잠자기 전에 잰 키는 다르게 나옵니다. 즉, 어떤 사람의 키가  $2.00139278291028719210196740527486202$ 미터일 확률을 물어보는 것은 의미 없습니다. 전체 인구에 대해서도 이 확률은 거의 $0$ 입니다. 따라서, 어떤 사람의 키가 어느 구간(예를 들면 1.99 와 2.01 미터 사이)에 속하는지를 묻는 것이 더 의미가 있습니다. 이런 경우들에는 우리는 이떤 값을 *밀도(density)*로 볼 가능성을 정량화합니다. 정확하게 2.0미터인 키에 대한 확률은 없습지다만, 밀도는 0이 아닙니다. 서로 다른 두 키의 구간에 대해서는 확률값이 0이 아닌 수가 됩니다.

There are a few important axioms of probability that you'll want to remember:

기억해 두어야할 몇가지 중요한 확률에 대한 공리(axiom)들이 있습니다.

* For any event $z​$, the probability is never negative, i.e. $\Pr(Z=z) \geq 0​$.
* For any two events $Z=z​$ and $X=x​$ the union is no more likely than the sum of the individual events, i.e. $\Pr(Z=z \cup X=x) \leq \Pr(Z=z) + \Pr(X=x)​$.
* For any random variable, the probabilities of all the values it can take must sum to 1, i.e. $\sum_{i=1}^n \Pr(Z=z_i) = 1​$.
* For any two mutually exclusive events $Z=z​$ and $X=x​$, the probability that either happens is equal to the sum of their individual probabilities, that is $\Pr(Z=z \cup X=x) = \Pr(Z=z) + \Pr(X=x)​$.
* 어떤 이벤트 $z$ 에 대해서, 확률은 절대로 음수가 아닙니다. 즉, $\Pr(Z=z) \geq 0$
* 두 이벤트 $Z=z$ 과 $X=x$ 에 대해서, 두 이벤트의 합집합(union)에 대한 확률은 각 이벤트의 확률의 합보다 클 수 없습니다. 즉,  $\Pr(Z=z \cup X=x) \leq \Pr(Z=z) + \Pr(X=x)​$.
* 어떤 확률 변수에 대해서, 모든 값들에 대한 확률의 합는 항상 1입니다. 즉, $\sum_{i=1}^n \Pr(Z=z_i) = 1$.
* 서로 겹치지 않는 두 사건, $Z=z$ 과 $X=x$, t,에 대해서, 둘 중에 한 사건이 일어날 확률은 각 사건의 확률의 합과 같습니다. 즉, $\Pr(Z=z \cup X=x) = \Pr(Z=z) + \Pr(X=x)$.

## Dealing with multiple random variables

Very often, we'll want to consider more than one random variable at a time.
For instance, we may want to model the relationship between diseases and symptoms.
Given a disease and symptom, say 'flu' and 'cough',
either may or may not occur in a patient with some probability.
While we hope that the probability of both would be close to zero,
we may want to estimate these probabilities and their relationships to each other
so that we may apply our inferences to effect better medical care.

종종 하나 이상의 확률 변수를 동시에 다룰 필요가 생깁니다. 질병과 증상의 관계를 모델링하는 경우를 들 수 있습니다. 질병과 증상이 주어졌을 때, 예를 들면 '독감'과 '기침', 두개는 어떤 확률로 환자에게 일어날 수도 일어나지 않을 수 있습니다. 이 둘에 대한 확률이 작기를 기대하지만, 더 좋은 의료 처방을 할 수 있도록 확률과 둘 사이의 관계를 예측하고자 합니다.

As a more complicated example, images contain millions of pixels, thus millions of random variables.
And in many cases images will come with a label, identifying objects in the image.
We can also think of the label as a random variable.
We can even get crazy and think of all the metadata as random variables
such as location, time, aperture, focal length, ISO, focus distance, camera type, etc.
All of these are random variables that occur jointly.
When we deal with multiple random variables,
there are several quantities of interest.
The first is called the joint distribution $\Pr(A, B)​$.
Given any elements $a​$ and $b​$,
the joint distribution lets us answer,
what is the probability that $A=a​$ and $B=b​$ simultaneously?
It might be clear that for any values $a​$ and $b​$, $\Pr(A,B) \leq \Pr(A=a)​$.

더 복잡한 예로, 수백만 픽셀로 이뤄진 이미지를 들어보겠습니다. 즉, 수백만 확률 변수가 존재합니다. 많은 경우에 이미지들은 이미지에 있는 객체를 지칭하는 레이블을 갖습니다. 이 레이블도 확률 변수로 생각할 수 있습니다. 더 나아가서는, 위치, 시간, 구경(apeture), 초점 거리, ISO, 초점, 카메라 종류 등 과같은 모든 메타 데이터를 확률 변수로 생각할수도 있습니다. 이 모든 것은 연관되서 발생하는 확률 변수들입니다. 여러 확률 변수를 다룰 때 몇가지 중요한 것들이 있습니다. 첫번째는 교차 확률 분포 $\Pr(A, B)$ 입니다.  두 원소 $a$ 와 $b$ 가 주어졌을 때, 교차 확률 분포는 동시에 $A=a$ 이고 $B=b$ 일 확률이 얼마인지에 대한 답을 줍니다. 임의의 값 $a$ 와 $b$ 에 대해서, $\Pr(A,B) \leq \Pr(A=a)$ 이라는 사실은 쉽게 알 수 있습니다.

This has to be the case, since for $A​$ and $B​$ to happen,
$A​$ has to happen *and* $B​$ also has to happen (and vice versa).
Thus $A,B​$ cannot be more likely than $A​$ or $B​$ individually.
This brings us to an interesting ratio: $0 \leq \frac{\Pr(A,B)}{\Pr(A)} \leq 1​$.
We call this a **conditional probability** and denote it by $\Pr(B
 | A)​$,
the probability that $B​$ happens, provided that $A​$ has happened.

$A$ 와 $B$ 가 일어났기 때문에, $A$ 가 발생하고, $B$ 또한 발생해야합니다. (또는 반대로).  즉, $A$ 와 $B$ 가 동시에 일어나는 것은 $A$ 와 $B$ 가 별도로 일어나는 것보다는 가능성이 낮습니다. 이 사실로 흥미로운 비율을 정의할 수 있습니다. 즉,  $0 \leq \frac{\Pr(A,B)}{\Pr(A)} \leq 1$. 우리는 이것을 **조건부 확률(conditional probability)** 이라고 부르며,  $\Pr(B
 | A)$ 로 표현합니다. 다시 말하면, $A$ 가 일어났을 때 $B$ 가 일어날 확률입니다.

Using the definition of conditional probabilities,
we can derive one of the most useful and celebrated equations in statistics - Bayes' theorem.
It goes as follows: By construction, we have that $\Pr(A, B) = \Pr(B
 | A) \Pr(A)​$.
By symmetry, this also holds for $\Pr(A,B) = \Pr(A | B) \Pr(B)​$.
Solving for one of the conditional variables we get:

조건부 확률의 정의를 이용하면, 확률에서 가장 유용하고 유명한 방적식을 도출할 수 있는데, 이것이 바로 베이즈 이론(Bayes' theorem) 입니다. 이를 도출하는 방법으로  $\Pr(A, B) = \Pr(B
 | A) \Pr(A)$ 로부터 출발합니다. 대칭성을 적용하면, $\Pr(A,B) = \Pr(A | B) \Pr(B)$ 이 돕니다. 조건 변수들 중 하나에 대해서 풀어보면 다음 공식을 얻게됩니다.

$$\Pr(A | B) = \frac{\Pr(B | A) \Pr(A)}{\Pr(B)}$$

This is very useful if we want to infer one thing from another,
say cause and effect but we only know the properties in the reverse direction.
One important operation that we need, to make this work, is **marginalization**, i.e.,
the operation of determining $\Pr(A)$ and $\Pr(B)$ from $\Pr(A,B)$.
We can see that the probability of seeing $A$ amounts to accounting
for all possible choices of $B$ and aggregating the joint probabilities over all of them, i.e.

**어떤 것으로부터 다른 어떤 것을 추론하고자 하는데, 즉 원인과 효과, 반대 방향에 대한 것만 알고 있을 경우에  아주 유용합니다. **marginalization 은 이것이 작동하게 만드는데 아주 중요한 연산입니다. 이 연산은  $\Pr(A,B)$ 로 부터 $\Pr(A)$ 와 $\Pr(B)$ 를 알아내는 연산입니다. $A$ 가 일어날 확률은 모든 $B$에 대한 교차 확률(joint probability)의 값으로 계산됩니다. 즉,

$$\Pr(A) = \sum_{B'} \Pr(A,B') \text{ and } \Pr(B) = \sum_{A'} \Pr(A',B)$$

A really useful property to check is for **dependence** and **independence**.
Independence is when the occurrence of one event does not influence the occurrence of the other.
In this case $\Pr(B | A) = \Pr(B)​$. Statisticians typically use $A \perp\!\!\!\perp B​$ to express this.
From Bayes' Theorem it follows immediately that also $\Pr(A | B) = \Pr(A)​$.
In all other cases we call $A​$ and $B​$ dependent.
For instance, two successive rolls of a die are independent.
On the other hand, the position of a light switch and the brightness in the room are not
(they are not perfectly deterministic, though,
since we could always have a broken lightbulb, power failure, or a broken switch).

점검해야할 아주 유용한 특성은 **종속**과 **독립** 입니다. 독립은 하나의 사건의 발생이 다른 사건의 발생에 영향을 주지 않는 것을 의미합니다. 위 경우에는 $\Pr(B | A) = \Pr(B)$ 를 의미합니다. 그 외의 경우들은 $A$ 와 $B$가 종속적이라고 합니다. 주사위를 두번 연속으로 던지는 것은 독립적이나, 방의 전등 스위치의 위치와 방의 밝기는 그렇지 않습니다. (이 둘이 완전히 결정적이지는 않습니다. 왜냐하면, 전구가 망가질 수도 있고, 전원이 나갈 수도 있고, 스위치가 망가질 경우 등이 있기 때문입니다.) 

Let's put our skills to the test.
Assume that a doctor administers an AIDS test to a patient.
This test is fairly accurate and it fails only with 1% probability
if the patient is healthy by reporting him as diseased. Moreover,
it never fails to detect HIV if the patient actually has it.
We use $D$ to indicate the diagnosis and $H$ to denote the HIV status.
Written as a table the outcome $\Pr(D | H)$ looks as follows:

배운 것을 테스트해보겠습니다. 의사가 환자에게 AIDS 테스트를 하는 것을 가정하겠습니다. 이 테스트는 상당히 정확해서, 환자가 음성일 경우 이를 틀리게 예측하는 확률이 1%이고, 환자가 양성일 경우 HIV 검출을 실패하디 않습니다. $D$ 는 진단 결과를 $H$ 는 HIV 상태를 표기합니다. $\Pr(D | H)$ 결과를 표로 만들어보면 다음과 같습니다.

|      outcome| HIV positive | HIV negative |
|:------------|-------------:|-------------:|
|Test positive|            1 |         0.01 |
|Test negative|            0 |         0.99 |

Note that the column sums are all one (but the row sums aren't),
since the conditional probability needs to sum up to $1​$, just like the probability.
Let us work out the probability of the patient having AIDS if the test comes back positive.
Obviously this is going to depend on how common the disease is, since it affects the number of false alarms.
Assume that the population is quite healthy, e.g. $\Pr(\text{HIV positive}) = 0.0015​$.
To apply Bayes' Theorem we need to determine

같은 열의 값을 더하면 1이나, 행으로 더하면 그렇지 않습니다.  그 이유는 조건부 확률도 합이 확률처럼 1이여야하기 때문입니다. 테스트 결과가 양성일 경우 환자가 AIDS에 결렸을 확률을 계산해보겠습니다. 당연하게 도 이는 질병이 얼마나 일반적인가에 따라 달라집니다. 인구의 대부분이 건강하다고 가정하겠습니다. 즉 $\Pr(\text{HIV positive}) = 0.0015$. 베이즈 이론(Bayes' Theorem)을 적용하기 위해서 우리는 다음을 결정해야합니다.

$$\begin{aligned}
\Pr(\text{Test positive}) =& \Pr(D=1 | H=0) \Pr(H=0) + \Pr(D=1 | H=1) \Pr(H=1) \\
=& 0.01 \cdot 0.9985 + 1 \cdot 0.0015 \\
=& 0.011485
\end{aligned}$$

따라서, 우리가 얻는 것은 다음과 같습니다.

$$\begin{aligned}
\Pr(H = 1 | D = 1) =& \frac{\Pr(D=1 | H=1) \Pr(H=1)}{\Pr(D=1)} \\
=& \frac{1 \cdot 0.0015}{0.011485} \\
=& 0.131
\end{aligned}
$$

In other words, there's only a 13.1% chance that the patient actually has AIDS, despite using a test that is 99% accurate! As we can see, statistics can be quite counterintuitive.

이 결과는 99% 정확도로 테스트 결과가 양성으로 나올지라도 환자가 실제로 AIDS에 걸렸을 확률은 13.1% 밖에 되지 않는 다는 것을 의미입니다. 이 결과에서 보듯이, 통계는 매우 직관적이지 않을 수 있습니다.

## Conditional independence

What should a patient do upon receiving such terrifying news?
Likely, he/she would ask the physician to administer another test to get clarity.
The second test has different characteristics (it isn't as good as the first one).

그렇다면, 환자가 이렇게 무서운 결과를 받았을 때 어떻게 해야할까요? 아마도 환자는 의사에게 테스트를 다시 해봐달라고 요청할 것입니다. 두번째 테스트는 다른게 나왔다고 하겠습니다. (즉, 첫번째 만큰 좋지 않습니다.)

|     outcome |  HIV positive |  HIV negative |
|:------------|--------------:|--------------:|
|Test positive|          0.98 |          0.03 |
|Test negative|          0.02 |          0.97 |

Unfortunately, the second test comes back positive, too.
Let us work out the requisite probabilities to invoke Bayes' Theorem.

안타깝게도 두번째 테스트 역시 양성으로 나오고 있습니다. 베이즈 이론(Bayes' Theorom)을 적용하기 위한 필요한 확률값들을 계산해봅니다.

* $\Pr(D_1 = 1 \text{ and } D_2 = 1 | H = 0) = 0.01 \cdot 0.03 = 0.0003$
* $\Pr(D_1 = 1 \text{ and } D_2 = 1 | H = 1) = 1 \cdot 0.98 = 0.98$
* $\Pr(D_1 = 1 \text{ and } D_2 = 1) = 0.0003 \cdot 0.9985 + 0.98 \cdot 0.0015 = 0.00176955$
* $\Pr(H = 1 | D_1 = 1 \text{ and } D_2 = 1) = \frac{0.98 \cdot 0.0015}{0.00176955} = 0.831$

That is, the second test allowed us to gain much higher confidence that not all is well.
Despite the second test being considerably less accurate than the first one,
it still improved our estimate quite a bit.
*Why couldn't we just run the first test a second time?*
After all, the first test was more accurate.
The reason is that we needed a second test that confirmed *independently* of the first test that things were dire, indeed. In other words, we made the tacit assumption that $\Pr(D_1, D_2 | H) = \Pr(D_1 | H) \Pr(D_2 | H)​$. Statisticians call such random variables **conditionally independent**. This is expressed as $D_1 \perp\!\!\!\perp D_2  | H​$.

즉, 두번째 테스트 결과는 좋지 않다는 것에 더 확신하게 만듭니다. 두번째 결과는 첫번째 보다 덜 정확함에도 불구하고, 예측 결과를 더 향상시켰습니다. *그렇다면, 첫번째 테스트를 두번하지 않을까요?* 결국, 첫번째 테스트가 더 정확했습니다. 두번째 테스트가 필요한 이유는 첫번째 테스트는 독립적으로 확인하기 위함입니다. 즉, $\Pr(D_1, D_2 | H) = \Pr(D_1 | H) \Pr(D_2 | H)$ 이라는 암묵적인 가정을 했습니다. 통계학에서는 이런 확률 변수를 **조건에 독립적**이라고 하며, $D_1 \perp\!\!\!\perp D_2  | H$ 라고 표현합니다.

## Summary

So far we covered probabilities, independence, conditional independence, and how to use this to draw some basic conclusions. This is already quite powerful. In the next section we will see how this can be used to perform some basic estimation using a Naive Bayes classifier.

이 절에서 우리는 확률, 독립, 조건 독립, 그리고 기본적인 결론을 도출하는데 이것들을 어떻게 사용하는지를 알아봤습니다. 이 개념들은 아주 유용합니다. 다음 절에서는 나이브 베이즈 분류기(Naive Nayes)를 사용한기본적인 예측을 하는데 이 개념들이 어떻게 사용되는지 살펴보겠습니다.

## Problems

1. Given two events with probability $\Pr(A)$ and $\Pr(B)$, compute upper and lower bounds on $\Pr(A \cup B)$ and $\Pr(A \cap B)​$. Hint - display the situation using a [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram).
1. Assume that we have a sequence of events, say $A​$, $B​$ and $C​$, where $B​$ only depends on $A​$ and $C​$ only on $B​$, can you simplify the joint probability? Hint - this is a [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain).
1. $\Pr(A)$ 과  $\Pr(B)$ 확률로 두 사건이 주어졌을 때, $\Pr(A \cup B)$ 와  $\Pr(A \cap B)$ 의 상한과 하한을 구하세요. 힌트 -  [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram)으ㄹ 사용하는 상황을 그려보세요.
1. 연속적인 사건, 즉 $A$, $B$, $C$, 들이 있는데, $B$ 는 $A$에만 의존하고, $C$ 는 $B$에만 의존한다고 가정합니다. 이 경우 교차 확률(joint probability)를 간단하게 할 수 있을까요? 힌트 - 이는  [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain) 입니다.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2319)

![](../img/qr_probability.svg)
