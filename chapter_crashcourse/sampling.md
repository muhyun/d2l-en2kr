# Sampling

Random numbers are just one form of random variables, and since computers are particularly good with numbers, pretty much everything else in code ultimately gets converted to numbers anyway. One of the basic tools needed to generate random numbers is to sample from a distribution. Let's start with what happens when we use a random number generator (after our usual import ritual).

난수는 랜덤 변수의 한 형태인데, 컴퓨터는 숫자를 다루는 것을 아주 잘하기 때문에 코드를 제외한 거의 모든 것은 결국 숫자로 바뀝니다. 난수를 만드는데 필요한 기본 도구중에 하나는 어떤 분포에서 샘플을 추출하는 것입니다. 그럼 난수 생성기를 사용했을 때 어떤일이 벌어지는지 보겠습니다. 우선 필요한 모듈을 import 하겠습니다.

```{.python .input}
%matplotlib inline
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import nd
import numpy as np
```

```{.python .input}
import random
for i in range(10):
    print(random.random())
```

## Uniform Distribution

These are some pretty random numbers. As we can see, their range is between 0 and 1, and they are evenly distributed. That means there is (actually, should be, since this is not a *real* random number generator) no interval in which numbers are more likely than in any other. In other words, the chances of any of these numbers to fall into the interval, say $[0.2,0.3)$ are as high as in the interval $[.593264, .693264)$. The way they are generated internally is by producing a random integer first, and then dividing it by its maximum range. If we want to have integers directly, try the following instead. It generates random numbers between 0 and 100.

위 결과의 숫자들은 굉장히 임의로 선택되언 것들입니다. 이 숫자들은 0과 1사이의 범위에서 잘 분포되어 있습니다. 즉, 어떤 특정 구간에 숫자들이 몰려있지 않습니다. (사실 *진짜* 난수 발생기가 아니기 때문에 그럴 수 도 있습니다.) 어떤 숫자가 $[0.2,0.3)$ 구간에서 뽑일 가능성과 $[.593264, .693264)$ 구간에서 뽑일 가능성이 비슷하다는 의미 입니다. 이 난수 발생기는 내부적으로 임의의 정수를 먼저 만들고, 그 다음에 이 숫자를 최대 구간의 숫자로 나누는 방식으로 동작합니다. 정수를 얻고 싶은 경우에는, 다음과 같이하면 됩니다. 아래 코드는 0과 100사이의 임의의 정수를 생성합니다.

```{.python .input}
for i in range(10):
    print(random.randint(1, 100))
```

What if we wanted to check that ``randint`` is actually really uniform. Intuitively the best strategy would be to run it, say 1 million times, count how many times it generates each one of the values and to ensure that the result is uniform.

`randint` 가 정말로 균일하다는 것을 확인을 어떻게 할 수 있을까요? 직관적으로 가장 좋은 방법은 100만개의 난수를 생성해서 각 숫자가 몇번 나오는지 계산하고, 이 분포가 균일한지를 확인하는 것입니다.

```{.python .input}
import math

counts = np.zeros(100)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.reshape(6)
# Mangle subplots such that we can index them in a linear fashion rather than
# a 2d grid

for i in range(1, 1000001):
    counts[random.randint(0, 99)] += 1
    if i in [10, 100, 1000, 10000, 100000, 1000000]:
        axes[int(math.log10(i))-1].bar(np.arange(1, 101), counts)
plt.show()
```

What we can see from the above figures is that the initial number of counts looks *very* uneven. If we sample fewer than 100 draws from a distribution over 100 outcomes this is pretty much expected. But even for 1000 samples there is a significant variability between the draws. What we are really aiming for is a situation where the probability of drawing a number $x$ is given by $p(x)$.

위 결과로 나온 그림으로 확인하면, 초기의 숫자는 *균등해 보이지 않습니다.* 100개의 정수에 대한 분포에서 100개보다 적은 개수를 뽑는 경우는 당연한 결과입니다. 하지만, 1000 샘플을 뽑아도 여전히 변동이 있습니다. 우리가 원하는 결과는 숫자를 뽑았을 때 그 확률이  $p(x)$ 가 되는 것입니다.

## The categorical distribution

Quite obviously, drawing from a uniform distribution over a set of 100 outcomes is quite simple. But what if we have nonuniform probabilities? Let's start with a simple case, a biased coin which comes up heads with probability 0.35 and tails with probability 0.65. A simple way to sample from that is to generate a uniform random variable over $[0,1]$ and if the number is less than $0.35$, we output heads and otherwise we generate tails. Let's try this out.

사실, 100개 중에서 균일한 분포로 뽑기를 하는 것은 아주 간단합니다. 만약에 불균일한 확률을 사용해야한다면 어떻게 해야할까요? 동전을 던졌을 때 앞면이 나올 확률이 0.35이고 뒷면이 나올 확률이 0.65이 나오는 편향된 (biased) 동전을 간단한 예로 들어보겠습니다. 이를 구현하는 간단한 방법은  $[0,1]$ 구간에서 균일하게 숫자를 선택하고 그 수가 0.35 보다 작으면 앞면으로, 크면 뒷면으로 하는 것입니다. 그럼 코드를 보겠습니다.

```{.python .input}
# Number of samples
n = 1000000
y = np.random.uniform(0, 1, n)
x = np.arange(1, n+1)
# Count number of occurrences and divide by the number of total draws
p0 = np.cumsum(y < 0.35) / x
p1 = np.cumsum(y >= 0.35) / x

plt.figure(figsize=(15, 8))
plt.semilogx(x, p0)
plt.semilogx(x, p1)
plt.show()
```

As we can see, on average this sampler will generate 35% zeros and 65% ones. Now what if we have more than two possible outcomes? We can simply generalize this idea as follows. Given any probability distribution, e.g.
$p = [0.1, 0.2, 0.05, 0.3, 0.25, 0.1]$ we can compute its cumulative distribution (python's ``cumsum`` will do this for you) $F = [0.1, 0.3, 0.35, 0.65, 0.9, 1]$. Once we have this we draw a random variable $x$ from the uniform distribution $U[0,1]$ and then find the interval where $F[i-1] \leq x < F[i]$. We then return $i$ as the sample. By construction, the chances of hitting interval $[F[i-1], F[i])$ has probability $p(i)$.

결과에서 보이듯이, 평균적으로 보면 35%가 0이고, 65%가 1입니다. 두개 이상의 결과가 있다면 어떻게할까요? 위 아이디어를 다음과 같이 일반화하면 됩니다. 예를 들면 $p = [0.1, 0.2, 0.05, 0.3, 0.25, 0.1]$ 와 같은 분포가 있다고 하면, 누적된 분포를 계산해서 ) $F = [0.1, 0.3, 0.35, 0.65, 0.9, 1]$ 를 얻습니다. (이는 Python의 `cumsum` 함수를 이용하면 간단히 할 수 있습니다.) 이전과 동일하게 $U[0,1]$ 범위의 균일 분포에서 난수 $x$ 를 뽑고,  $F[i-1] \leq x < F[i]$ 를 만족시키는 구간을 찾아서  $i$ 를 샘플 결과로 리턴합니다. 이렇게 하면 난수가 $[F[i-1], F[i])$ 구간에 속할 확률이 $p(i)$이 됩니다.

Note that there are many more efficient algorithms for sampling than the one above. For instance, binary search over $F​$ will run in $O(\log n)​$ time for $n​$ random variables. There are even more clever algorithms, such as the [Alias Method](https://en.wikipedia.org/wiki/Alias_method) to sample in constant time, after $O(n)​$ preprocessing.

위 방법보다 훨씬 더 효율적인 알고리즘이 많이 있습니다. 예를 들면,  $n$ 개의 랜덤 변수에 대해서 $F$ 에 대한 이진 검색을 수행하면  $O(\log n)$ 시간이 걸립니다.  $O(n)$ 만큼의 전처리를 하면 샘플링에 상수의 시간이 걸리는 [Alias Method](https://en.wikipedia.org/wiki/Alias_method) 와 같은 더 좋은 알고리즘들이 있습니다.

## The Normal distribution

The Normal distribution (aka the Gaussian distribution) is given by $p(x) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{1}{2} x^2\right)$. Let's plot it to get a feel for it.

표준 분포 (또는 가우시안 분포)는 $p(x) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{1}{2} x^2\right)$ 로 정의됩니다. 그림으로 확인해보겠습니다.

```{.python .input}
x = np.arange(-10, 10, 0.01)
p = (1/math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)
plt.figure(figsize=(10, 5))
plt.plot(x, p)
plt.show()
```

Sampling from this distribution is a lot less trivial. First off, the support is infinite, that is, for any $x​$ the density $p(x)​$ is positive. Secondly, the density is nonuniform. There are many tricks for sampling from it - the key idea in all algorithms is to stratify $p(x)​$ in such a way as to map it to the uniform distribution $U[0,1]​$. One way to do this is with the probability integral transform.

이 분포에서 샘플링을 하는 것은 그리 간단하지 않습니다. 우선 **서포트는 무한합니다.** 즉, 어떤  $x$ 값이 주어지든지 확률 밀도 $p(x)$ 값이 양수입니다. 두번째 특징은 확률 밀도는 균일하지 않습니다. 이 분포에서 샘플링을 수행하는 많은 기법이 있는데, 모든 알고리즘에 사용되는 주요 아이디어는  $p(x)$ 를 계층화해서 균일한 분포 $U[0,1]$ 로 매핑시키는 것입니다. 확률 적분 변환이 그 방법중에 하나입니다.

Denote by $F(x) = \int_{-\infty}^x p(z) dz​$ the cumulative distribution function (CDF) of $p​$. This is in a way the continuous version of the cumulative sum that we used previously. In the same way we can now define the inverse map $F^{-1}(\xi)​$, where $\xi​$ is drawn uniformly. Unlike previously where we needed to find the correct interval for the vector $F​$ (i.e. for the piecewise constant function), we now invert the function $F(x)​$.

$p$ 의 누적 분포 함수(cumulative distribuion function, CDF)를 $F(x) = \int_{-\infty}^x p(z) dz$ 로 표기합니다. 이 방법은 앞에서 누적합의 연속된 버전이라고 할 수 있습니다. 이전과 같은 방법으로 균일하게 뽑은 $\xi$ 에 대해서 역으로 매핑하는 $F^{-1}(\xi)$ 를 정의할 수 있습니다. 백터 $F$ 에 대한 정확한 구간을 찾는 이전의 문제와 다르게, 우리는 $F(x)$ 의 역을 구해야합니다.

In practice, this is slightly more tricky since inverting the CDF is hard in the case of a Gaussian. It turns out that the *twodimensional* integral is much easier to deal with, thus yielding two normal random variables than one, albeit at the price of two uniformly distributed ones. For now, suffice it to say that there are built-in algorithms to address this.

실제로 가우시인의 경우 CDF의 역을 구하는 것은 다소 까다롭습니다. 두개의 균등한 분포들을 다뤄야하지만 *이차원* 적분이 더 다루기 쉽기 떄문에 두개의 표준 랜텀 변수로 만듭니다. 지금은 이 문제를 해결해주는 알고리즘이 있다고 해두겠습니다.

The normal distribution has yet another desirable property. In a way all distributions converge to it, if we only average over a sufficiently large number of draws from any other distribution. To understand this in a bit more detail, we need to introduce three important things: expected values, means and variances.

표준 분포는 또다른 중요한 특징이 있습니다. 만약 어떤 분포에서 충분히 많은 뽑기를 해서 평균을 구한다면, 모든 분포는 표준 분포에 수렴합니다. 이를 더 자세히 이해가기 위해서, 세가지 중요한 개념들, 기대값(expected value), 평균과 분산을 소개하겠습니다. 

* The expected value $\mathbf{E}_{x \sim p(x)}[f(x)]​$ of a function $f​$ under a distribution $p​$ is given by the integral $\int_x p(x) f(x) dx​$. That is, we average over all possible outcomes, as given by $p​$.
* A particularly important expected value is that for the function $f(x) = x​$, i.e. $\mu := \mathbf{E}_{x \sim p(x)}[x]​$. It provides us with some idea about the typical values of $x​$.
* Another important quantity is the variance, i.e. the typical deviation from the mean
  $\sigma^2 := \mathbf{E}_{x \sim p(x)}[(x-\mu)^2]​$. Simple math shows (check it as an exercise) that
  $\sigma^2 = \mathbf{E}_{x \sim p(x)}[x^2] - \mathbf{E}^2_{x \sim p(x)}[x]​$.
* 분포 $p$ 를 따르는 함수 $f$ 에 대한 기대값 $\mathbf{E}_{x \sim p(x)}[f(x)]$  은 적분 $\int_x p(x) f(x) dx$ 으로 계산됩니다. 즉, 이는 $p$ 에 따라 주어지는 모든 결과에 대한 평균값입니다.
* 함수 $f(x) = x$ 에 대한 기대값은 특별하게 중요합니다. 이 함수의 기대값은 $\mu := \mathbf{E}_{x \sim p(x)}[x]$ 입니다. 이는 전형적인 $x$ 에 대한 아이디어를 제공해주기 때문입니다. 
* 중요한 다른 개념으로는 분산이 있습니다. 이는 $\sigma^2 := \mathbf{E}_{x \sim p(x)}[(x-\mu)^2]$  으로 표현디며, 평균으로 부터 얼마나 떨어져 있는지를 알려줍니다. 간단한 계산을 하면 분산은 $\sigma^2 = \mathbf{E}_{x \sim p(x)}[x^2] - \mathbf{E}^2_{x \sim p(x)}[x]$ 로 표현되기도 합니다.

The above allows us to change both mean and variance of random variables. Quite obviously for some random variable $x​$ with mean $\mu​$, the random variable $x + c​$ has mean $\mu + c​$. Moreover, $\gamma x​$ has the variance $\gamma^2 \sigma^2​$. Applying this to the normal distribution we see that one with mean $\mu​$ and variance $\sigma^2​$ has the form $p(x) = \frac{1}{\sqrt{2 \sigma^2 \pi}} \exp\left(-\frac{1}{2 \sigma^2} (x-\mu)^2\right)​$. Note the scaling factor $\frac{1}{\sigma}​$ - it arises from the fact that if we stretch the distribution by $\sigma​$, we need to lower it by $\frac{1}{\sigma}​$ to retain the same probability mass (i.e. the weight under the distribution always needs to integrate out to 1).

위 개념은 랜덤 변수의 평균과 분산을 바꿀 수 있게 해줍니다. 예를 들면 랜덤 변수 $x$ 의 평균이 $\mu$ 일 경우 랜덤 변수 $x + c$ 의 평균은  $\mu + c$ 이 됩니다. 또한, 랜덤 변수가 $\gamma x$ 일 경우에는 분산은 $\gamma^2 \sigma^2$ 이 됩니다. 평균 $\mu$ 이고 분산이  $\sigma^2$ 인 랜덤 변수에 표준 분포(normal distribution)을 적용하면 $p(x) = \frac{1}{\sqrt{2 \sigma^2 \pi}} \exp\left(-\frac{1}{2 \sigma^2} (x-\mu)^2\right)$ 의 형태가 됩니다. 스캐일 팩터 $\frac{1}{\sigma}$ 가 적용된 것을 주의하세요. 이렇게 한 이유는 이 분포를 $\sigma$ 만큼 늘릴 경우, 같은 확률 값을 갖게하기 위해서  $\frac{1}{\sigma}$ 만큼 줄여야할 필요가 있디 때문입니다. (즉, 분포의 weight들의 합은 항상 1이어야하기 때문입니다.)

Now we are ready to state one of the most fundamental theorems in statistics, the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem). It states that for sufficiently well-behaved random variables, in particular random variables with well-defined mean and variance, the sum tends toward a normal distribution. To get some idea, let's repeat the experiment described in the beginning, but now using random variables with integer values of $\{0, 1, 2\}​$.

자 이제 통계학에서 가장 기본적인 이론 중에 하나에 대해서 알아볼 준비가 되었습니다. 이는 [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) 입니다. 이 이론은 충분히 잘 행동하는 확률 변수에 대해서, 특히 잘 정의된 평균과 분산을 가지고 있는 확률 변수, 전체 합은 표준 분포로 근접합니다. 이해를 돕기 위해서 시작할 때 사용했던 실험을 정수값 \{0, 1, 2\} 을 갖는 확률변수를 사용해서 해봅니다.

```{.python .input}
# Generate 10 random sequences of 10,000 uniformly distributed random variables
tmp = np.random.uniform(size=(10000,10))
x = 1.0 * (tmp > 0.3) + 1.0 * (tmp > 0.8)
mean = 1 * 0.5 + 2 * 0.2
variance = 1 * 0.5 + 4 * 0.2 - mean**2
print('mean {}, variance {}'.format(mean, variance))

# Cumulative sum and normalization
y = np.arange(1,10001).reshape(10000,1)
z = np.cumsum(x,axis=0) / y

plt.figure(figsize=(10,5))
for i in range(10):
    plt.semilogx(y,z[:,i])

plt.semilogx(y,(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.semilogx(y,-(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.show()
```

This looks very similar to the initial example, at least in the limit of averages of large numbers of variables. This is confirmed by theory. Denote by mean and variance of a random variable the quantities

위 결과를 보면 많은 변수들의 평균만을 보면 처음 예제와 아주 비슷하게 보입니다. 즉, 이론이 맞다는 것을 보여줍니다. 확률변수의 평균과 분산은 다음과 같이 표현됩니다.

$$\mu[p] := \mathbf{E}_{x \sim p(x)}[x] \text{ 와 } \sigma^2[p] := \mathbf{E}_{x \sim p(x)}[(x - \mu[p])^2]$$

Then we have that $\lim_{n\to \infty} \frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{x_i - \mu}{\sigma} \to \mathcal{N}(0, 1)$. In other words, regardless of what we started out with, we will always converge to a Gaussian. This is one of the reasons why Gaussians are so popular in statistics.

그러면, $\lim_{n\to \infty} \frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{x_i - \mu}{\sigma} \to \mathcal{N}(0, 1)$ 이 됩니다. 즉, 어떤 값으로 부터 시작했는지 상관없이, 가우시안 분포에 항상 수렴하게 됩니다. 이것이 통계에서 가우시안이 유명한 이유들 중에 하나입니다.


## More distributions

Many more useful distributions exist. We recommend consulting a statistics book or looking some of them up on Wikipedia for further detail.

그외에도 유용한 분산들이 많이 있습니다. 더 자세한 내용은 통계책이나 위키피디아를 참조하세요.

* **Binomial Distribution** It is used to describe the distribution over multiple draws from the same distribution, e.g. the number of heads when tossing a biased coin (i.e. a coin with probability $\pi \in [0, 1]​$ of returning heads) 10 times. The binomial probability is given by $p(x) = {n \choose x} \pi^x (1-\pi)^{n-x}​$.
* **Multinomial Distribution** Obviously we can have more than two outcomes, e.g. when rolling a dice multiple times. In this case the distribution is given by $p(x) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k \pi_i^{x_i}​$.
* **Poisson Distribution** It is used to model the occurrence of point events that happen with a given rate, e.g. the number of raindrops arriving within a given amount of time in an area (weird fact - the number of Prussian soldiers being killed by horses kicking them followed that distribution). Given a rate $\lambda$, the number of occurrences is given by $p(x) = \frac{1}{x!} \lambda^x e^{-\lambda}$.
* **Beta, Dirichlet, Gamma, and Wishart Distributions** They are what statisticians call *conjugate* to the Binomial, Multinomial, Poisson and Gaussian respectively. Without going into detail, these distributions are often used as priors for coefficients of the latter set of distributions, e.g. a Beta distribution as a prior for modeling the probability for binomial outcomes.
* **이항 분포(Binomial Distribution)** 같은 분포에서 여러번 뽄을 때의 분포를 설명하는데 사용됩니다. 즉, 편향된 동전(동전앞면이 나올 확률이 $\pi \in [0, 1]$ 인 동전을 사용할 때)을 10번 던져서 앞면이 나오는 회수. 이산 분포는  $p(x) = {n \choose x} \pi^x (1-\pi)^{n-x}$ 입니다.
* **다항 분포(Multinomial Distribution)** 두개보다 많은 결과가 있을 경우에 해당합니다. 즉, 주사위를 여러번 던지는 경우를 예로 들 수 있습니다. 이 경우 분포는 $p(x) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k \pi_i^{x_i}​$ 로 주어집니다.
* **포아송 분포(Poisson Distribution)** 주어진 속도(rate)에 따라서 일어나는 이벤트를 모델링할 때 사용됩니다. 예를 들면, 어느 공간에 일정 시간동안 떨어지는 빗방울의 수가 됩니다. (특이한 사실은, 프러시안 군인들이 말의 발길에 치여서 죽은 수가 이 분포를 따르고 있습니다.) 속도 $\lambda$ 에 대해서, 일이 일어날 확율은 $p(x) = \frac{1}{x!} \lambda^x e^{-\lambda}$ 로 표현됩니다.
* **배타, 디리치(Dirichlet), 감마, 위샤트(Wishart) 분포** 통계학자들은 이것들을 각각 이산, 다항, 포아송, 그리고 가우시안 분포의 변종이라고 설명하고 있습니다. 이 분포들은 분포들의 집합에 대한 계수를 위한 사전 순위로 사용되는데, 자세한 설명은 생략하겠습니다. 이산 결과들의 활률을 모델링하는데 사전 순위로서의 베타 분포 같은 것입니다.


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2321)

![](../img/qr_sampling.svg)
