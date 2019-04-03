# Environment

# 환경

So far we did not worry very much about where the data came from and how the models that we build get deployed. Not caring about it can be problematic. Many failed machine learning deployments can be traced back to this situation. This chapter is meant to help with detecting such situations early and points out how to mitigate them. Depending on the case this might be rather simple (ask for the 'right' data) or really difficult (implement a reinforcement learning system).

지금까지 우리는 데이터가 어디서 왔는지 모델이 어떻게 배포되는지에 대해서는 걱정하지 않았습니다. 하지만, 이것들을 고려하지 않는 것은 문제가 됩니다. 실패한 많은 머신 러닝 배포들의 원인을 추적해보면 이런 상황이 원인이 됩니다. 이 절에서는 이 상황을 초기에 발견하고, 완화하는 방법을 알아봅니다. 상황에 따라서, 정확한 데이터를 사용하면되는 다소 간단한 문제일 수 있기도 하지만, 강화학습 시스템을 만드는 것과 같이 어려운 문제이기도 합니다.

## Covariate Shift

At its heart is a problem that is easy to understand but also equally easy to miss. Consider being given the challenge of distinguishing cats and dogs. Our training data consists of images of the following kind:

이해하는 것은 쉽지만, 놓치기 쉬운 문제가 있습니다. 강아지와 고양을 구분하는 문제를 생각해봅시다. 학습 데이터는 다음과 같이 주어졌습니다.

|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|

At test time we are asked to classify the following images:

테스트에서는 다음 그림을 분류하도록 요청 받습니다.

|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat-cartoon1.png)|![](../img/cat-cartoon2.png)|![](../img/dog-cartoon1.png)|![](../img/dog-cartoon2.png)|

Obviously this is unlikely to work well. The training set consists of photos, while the test set contains only cartoons. The colors aren't even accurate. Training on a dataset that looks substantially different from the test set without some plan for how to adapt to the new domain is a bad idea. Unfortunately, this is a very common pitfall. Statisticians call this **Covariate Shift**, i.e. the situation where the distribution over the covariates (aka training data) is shifted on test data relative to the training case. Mathematically speaking, we are referring the case where $p(x)$ changes but $p(y|x)$ remains unchanged.

당연하게 이것은 잘 작동하지 않습니다. 학습 데이터는 실제 사진으로 구성되어 있지만, 테스트 셋은 만화 그림으로 되어있습니다. 색상도 정확하지 않습니다. 새로운 도메인에 어떻게 적용할지 계획이 없이 테스트 셋과 다른 데이터로 학습을 시키는 것은 나쁜 아이디어 입니다. 불행하게도 이것은 흔한 함정입니다. 통계학자들은 이것을  **Covariate Shift** 라고 합니다. 즉, covariates (학습 데이터)의 분포가 테스트 데이터의 분포가 다른 상황을 의미합니다. 수학적으로 말하자면,  $p(x)$ 는 변화하는데,  $p(y|x)$ 는 그대로 있는 경우를 의미합니다.

## Concept Shift

A related problem is that of concept shift. This is the situation where the labels change. This sounds weird - after all, a cat is a cat. Well, cats maybe but not soft drinks. There is considerable concept shift throughout the USA, even for such a simple term:

**……..**

![](../img/popvssoda.png)

If we were to build a machine translation system, the distribution $p(y|x)$ would be different, e.g. depending on our location. This problem can be quite tricky to spot. A saving grace is that quite often the $p(y|x)$ only shifts gradually (e.g. the click-through rate for NOKIA phone ads). Before we go into further details, let us discuss a number of situations where covariate and concept shift are not quite as blatantly obvious.

머신 번역 시스템을 만든다면, 분포 $p(y|x)$ 는 지역에 따라서 다를 수 있습니다. 이 문제를 집어 내기에는 상당이 까다롭습니다. 다행한 것은 많은 경우에  $p(y|x)$ 는 조금씩만 변화한다는 것입니다. 더 자세히 살펴보기 전에, covariate shift와 concept shift가 명백하게 드러나지 않는 많은 상황에 대해서 살펴보겠습니다.

## Examples

## 예제

### Medical Diagnostics

### 의학 분석

Imagine you want to design some algorithm to detect cancer. You get data of healthy and sick people; you train your algorithm; it works fine, giving you high accuracy and you conclude that you’re ready for a successful career in medical diagnostics. Not so fast ...

암을 진단하는 알고리즘을 설계하는 것을 상상해보세요. 건강한 사람과 아픈 사람의 데이터를 얻은 후, 알고리즘을 학습시킵니다. 학습된 모델이 높은 정확도를 보여주면서 잘 동작합니다. 당신은 이제 의료 분석 분야에서 성공적인 경력을 시작할 수 있다고 판단합니다. 하지만 너무 이릅니다.

Many things could go wrong. In particular, the distributions that you work with for training and those in the wild might differ considerably. This happened to an unfortunate startup I had the opportunity to consult for many years ago. They were developing a blood test for a disease that affects mainly older men and they’d managed to obtain a fair amount of blood samples from patients. It is considerably more difficult, though, to obtain blood samples from healthy men (mainly for ethical reasons). To compensate for that, they asked a large number of students on campus to donate blood and they performed their test. Then they asked me whether I could help them build a classifier to detect the disease. I told them that it would be very easy to distinguish between both datasets with probably near perfect accuracy. After all, the test subjects differed in age, hormone level, physical activity, diet, alcohol consumption, and many more factors unrelated to the disease. This was unlikely to be the case with real patients: Their sampling procedure had caused an extreme case of covariate shift that couldn’t be corrected by conventional means. In other words, training and test data were so different that nothing useful could be done and they had wasted significant amounts of money.

많은 것들이 잘못될 수 있습니다. 특히, 학습에 사용한 분포와 실제 분포는 상당히 다를 수 있습니다. 실제로 수년 전에 스타트업 회사를 컨설팅하면서 겪었던 일입니다. 이 회사는 주로 나이 많은 남성에서 발견되는 질병에 대한 혈액 테스트를 개발하고 있었습니다. 이를 위해서 환자들로부터 상당히 많은 샘플을 수집할 수 있었습니다. 하지만, 윤리적인 이유로 건강한 남자의 혈액 샘플을 구하는 것은 상당히 어려웠습니다. 이를 해결하기 위해서, 캠퍼스의 학생들에게 혈액을 기증 받아서 테스트를 수행했습니다. 그리고, 그 회사는 나에게 질병을 분류하는 모델을 만드는 것에 대한 도움을 요청했습니다. 거의 완벽한 정확도의 확률로 두 데이터 셋을 분류하는 것은 아주 쉽다고 알려줬습니다. 결국, 모든 테스트 대상은 나이, 호르몬 레벨, 신체 활동, 식이 상태, 알콜 섭취, 그리고 질병과 연관이 없는 아주 많은 요소들이 달랐습니다. 하지만, 이는 실제 환자의 경우와 차이가 있습니다. 이들이 사용한 샘플링 절차는 아주 심한 covariante shift를 가지고 와서, 어떤 전통적인 방법으로 고쳐질 수가 없었습니다. 달리 말하면, 학습 데이터와 테스트 데이터가 너무나 달라서 어떤 유용한 일도 할 수 없었고, 결국 상당히 많은 돈을 낭비만 했습니다.

### Self Driving Cars

### 자율 주행 자동차

A company wanted to build a machine learning system for self-driving cars. One of the key components is a roadside detector. Since real annotated data is expensive to get, they had the (smart and questionable) idea to use synthetic data from a game rendering engine as additional training data. This worked really well on 'test data' drawn from the rendering engine. Alas, inside a real car it was a disaster. As it turned out, the roadside had been rendered with a very simplistic texture. More importantly, *all* the roadside had been rendered with the *same* texture and the roadside detector learned about this 'feature' very quickly.

자율 주행차를 위한 머신러닝 시스템을 만들고자 하는 한 회사가 있습니다. 도로를 탐지하는 것이 중요한 컴포넌트 중에 하나입니다. 실제 답을 다는 것이 너무 비싸기 때문에, 게임 렌더링 엔진을 사용해서 생성한 데이터를 추가 학습 데이터로 사용하기로 했습니다. 이렇게 학습된 모델은 렌더링 엔진으로 만들어진 '테스트 데이터'에는 잘 동작했습니다. 하지만, 실제 차에서는 재앙이었습니다. 이유는 렌더링 된 도로가 너무 단순한 텍스처를 사용했기 때문 였습니다. 더 중요한 것은 모든 도로 경계가 같은 텍스터로 렌더되었기에, 도로 탐지기는 이 '특징'을 너무 빨리 배워버렸습니다.

A similar thing happened to the US Army when they first tried to detect tanks in the forest. They took aerial photographs of the forest without tanks, then drove the tanks into the forest and took another set of pictures. The so-trained classifier worked 'perfectly'. Unfortunately, all it had learned was to distinguish trees with shadows from trees without shadows - the first set of pictures was taken in the early morning, the second one at noon.

미국 군대에서 숲 속에서 있는 탱크를 탐지하는 것을 하려고 했을 때도 비슷한 문제가 발생했습니다. 탱크가 없는 숲의 항공 사진을 찍고, 탱크를 숲으로 몰고 가서 다른 사진을 찍었습니다. 이렇게 학습된 분류기는 아주 완벽하게 동작했습니다. 하지만 불행히도 이 모델은 그늘이 있는 나무들과 그늘이 없는 나무를 구분하고 있었습니다. 이유는 첫번째 사진은 이른 아침에 찍었고, 두번째 사진은 정오에 찍었기 때문이었습니다.

### Nonstationary distributions

### 정적이지 않은 분포(nonstationary distribution)

A much more subtle situation is where the distribution changes slowly and the model is not updated adequately. Here are a number of typical cases:

더 알아내기 힘든 상황은 분포가 천천히 변화하는 상황에서 모델을 적절하게 업데이트를 하지 않는 경우입니다. 전형적인 사례로는 다음과 같은 경우가 있습니다.

* We train a computational advertising model and then fail to update it frequently (e.g. we forget to incorporate that an obscure new device called an iPad was just launched).
* We build a spam filter. It works well at detecting all spam that we've seen so far. But then the spammers wisen up and craft new messages that look quite unlike anything we've seen before.
* We build a product recommendation system. It works well for the winter. But then it keeps on recommending Santa hats after Christmas.
* 광고 모델을 학습시킨 후, 자주 업데이트하는 것을 실패한 경우. (예를 들면, iPad 라는 새로운 디바이스가 막 출시된 것을 반영하는 것을 잊은 경우)
* 스팸 필더를 만들었습니다. 이 스팸 필터는 우리가 봤던 모든 스팸을 모두 잘 탐지합니다. 하지만, 스팸을 보내는 사람들이 이를 알고 이전에 봐왔던 것과는 아무 다른 새로운 메시지를 만듭니다.
* 상품 추천 시스템을 만들었습니다. 겨울에는 잘 동작합니다. 하지만, 크리스마스가 지난 후에도 산타 모자를 계속 추천하고 있습니다.

### More Anecdotes

### 더 많은 예제들

* We build a classifier for "Not suitable/safe for work" (NSFW) images. To make our life easy, we scrape a few seedy Subreddits. Unfortunately the accuracy on real life data is lacking (the pictures posted on Reddit are mostly 'remarkable' in some way, e.g. being taken by skilled photographers, whereas most real NSFW images are fairly unremarkable ...). Quite unsurprisingly the accuracy is not very high on real data.
* We build a face detector. It works well on all benchmarks. Unfortunately it fails on test data - the offending examples are close-ups where the face fills the entire image (no such data was in the training set).
* We build a web search engine for the USA market and want to deploy it in the UK.
* "업무에 부적합 또는 안전한 (Not suitable/safe for work (NSFW))" 이미지 판별기를 만들고 있습니다. 쉽게하기 위해서, Subreddit에서 이미지를 수집합니다. 불행하게도 실제 생활 데이터에 대한 정확도는 낮게 나옵니다. (Reddit에 올라와 있는 사진은 전문 사진가가 찍은 품질이 좋은 사진들인 반면에 실제 NSFW 이미지는 품질이 좋지 않습니다.)
* 얼굴 인식기를 만듭니다. 모든 밴치마크에서 잘 동작합니다. 하지만, 테스트 데이터에서는 그렇지 못합니다. 실패한 이미지를 보니 이미지 전체를 얼굴이 차지하는 클로즈업 사진들입니다.
* 미국 마켓을 위한 웹 검색 엔진을 만들어서 영국에 배포하고 싶습니다.

In short, there are many cases where training and test distribution $p(x)$ are different. In some cases, we get lucky and the models work despite the covariate shift. We now discuss principled solution strategies. Warning - this will require some math and statistics.

요약하면, 학습 데이터의 분포와 테스트 데이터의 분포가 다른 다양한 사례가 있습니다. 어떤 경우에는 운이 좋아서 covariate shift가 있음에도 불구하고 모델이 잘 동작할 수 있습니다. 자 지금부터 원칙적인 해결 전략에 대해서 이야기하겠습니다. 경고 - 약간의 수학과 통계가 필요합니다.

## Covariate Shift Correction

## Covarate shift 교정

Assume that we want to estimate some dependency $p(y|x)$ for which we have labeled data $(x_i,y_i)$. Alas, the observations $x_i$ are drawn from some distribution $q(x)$ rather than the ‘proper’ distribution $p(x)$. To make progress, we need to reflect about what exactly is happening during training: we iterate over training data and associated labels $\{(x_1, y_1), \ldots (y_n, y_n)\}$ and update the weight vectors of the model after every minibatch.

Label을 달아놓은 데이터  $(x_i,y_i)$ 에 대한 의존도  $p(y|x)$ 를 추정하는 것을 한다고 가정합니다. 그런데,  $x_i$ 가 올바른 분포인  $p(x)$ 가 아닌 다른 분포 $q(x)$ 를 갖는 곳에서 추출됩니다. 먼저, 우리는 학습 과정에 정확하게 어떤 일이 일어나는지에 대해서 잘 생각해볼 필요가 있습니다. 즉, 학습 데이터와 연관된 label을 반복하면서, 매 mini-batch 이후에 모델의 weight vector들을 업데이트합니다.

Depending on the situation we also apply some penalty to the parameters, such as weight decay, dropout, zoneout, or anything similar. This means that we largely minimize the loss on the training.

경우에 따라서 우리는 파라메터에 weight decay, dropout, zoneout 또는 유사한 패널티를 적용합니다. 즉, 학습은 대부분 loss를 최초화하는 것을 의미합니다.
$$
\mathop{\mathrm{minimize}}_w \frac{1}{n} \sum_{i=1}^n l(x_i, y_i, f(x_i)) + \mathrm{some~penalty}(w)
$$

Statisticians call the first term an *empirical average*, that is an average computed over the data drawn from $p(x) p(y|x)$. If the data is drawn from the 'wrong' distribution $q$, we can correct for that by using the following simple identity:

통계학자들인 첫번째 항을 경험적인 평균 (empirical average)이라고 합니다. 즉, 이것은  $p(x) p(y|x)$ 확률로 선택된 데이터에 구해진 평균을 의미합니다. 만약 데이터가 잘못된 분포 $q$ 에서 선택된다면, 다음과 같이 간단한 identity를 사용해서 수정할 수 있습니다.
$$
\begin{aligned}
\int p(x) f(x) dx & = \int p(x) f(x) \frac{q(x)}{p(x)} dx \\
& = \int q(x) f(x) \frac{p(x)}{q(x)} dx
\end{aligned}
$$

In other words, we need to re-weight each instance by the ratio of probabilities that it would have been drawn from the correct distribution $\beta(x) := p(x)/q(x)​$. Alas, we do not know that ratio, so before we can do anything useful we need to estimate it. Many methods are available, e.g. some rather fancy operator theoretic ones which try to recalibrate the expectation operator directly using a minimum-norm or a maximum entropy principle. Note that for any such approach, we need samples drawn from both distributions - the 'true' $p​$, e.g. by access to training data, and the one used for generating the training set $q​$ (the latter is trivially available).

다르게 설명해보면, 데이터가 추출 되어야하는 올바른 분포에 대한 확률의 비율을 곱($\beta(x) := p(x)/q(x)​$ )해서 각 샘플의 weight를 조절하면 됩니다. 하지만 안타깝게도 이 비율을 알지 못 합니다. 따라서, 우선 해야하는 일은 이 값을 추정하는 것입니다. 이를 추정하는 다양한 방법이 존재합니다. 예로는 다소 멋진 이론적인 연산 방법이 있습니다. 이는 예상치를 계산하는 연산을 재조정하는 것으로, 이는 minimum-norm 이나 maximum entropy 원칙을 직접 이용하는 방법입니다. 이런 방법들은 두 분포에서 샘플들을 수집해야하는 것을 염두해 두세요. 즉, 학습 데이터를 이용해서 진짜 $p​$ , 그리고 학습 데이터셋을 $q​$ 를 만드는데 사용한 분포를 의미합니다.

In this case there exists a very effective approach that will give almost as good results: logistic regression. This is all that is needed to compute estimate probability ratios. We learn a classifier to distinguish between data drawn from $p(x)$ and data drawn from $q(x)$. If it is impossible to distinguish between the two distributions then it means that the associated instances are equally likely to come from either one of the two distributions. On the other hand, any instances that can be well discriminated should be significantly over/underweighted accordingly. For simplicity’s sake assume that we have an equal number of instances from both distributions, denoted by $x_i \sim p(x)$ and $x_i′ \sim q(x)$ respectively. Now denote by $z_i$ labels which are 1 for data drawn from $p$ and -1 for data drawn from $q$. Then the probability in a mixed dataset is given by

이 경우 좋은 결과를 주는 효과적인 방법이 있는데, 그것은 바로 선형 회귀(logistic regression)입니다. 선형 회귀를 이용하면 확률 비율을 계산해낼 수 있습니다.  $p(x)$ 로 부터 추출된 데이터와 $q(x)$ 로 부터 추출된 데이터를 구분하기 위한 분리 모델을 학습 시키실 수 있습니다. 두 분포를 구별하는 것이 불가능하다면, 샘플들은 두 분포 중에 하나에서 나왔다는 것을 의미합니다. 반면에 분류가 잘 되는 샘플들은 overweighted 되었거나 underweight 되어 있을 것입니다. 간단하게 설명하기 위해서, 두 분포로부터 같은 개수만큼 샘플을 추출했다고 가정하겠습니다. 이를 각각 $x_i \sim p(x)$ 와 $x_i′ \sim q(x)$ 로 표기합니다. $p$ 로부터 추출된 경우 $z_i$ 를 1로, $q$ 로 부터 추출된 경우에는 -1로 값을 할당합니다. 그러면, 섞인 데이터셋의 확률은 다음과 같이 표현됩니다.

$$p(z=1|x) = \frac{p(x)}{p(x)+q(x)} \text{ and hence } \frac{p(z=1|x)}{p(z=-1|x)} = \frac{p(x)}{q(x)}$$

Hence, if we use a logistic regression approach where $p(z=1|x)=\frac{1}{1+\exp(−f(x))}$ it follows that

따라서,  $p(z=1|x)=\frac{1}{1+\exp(−f(x)}$ 를 만족시키는 선형 회귀(logistic regression) 방법을 사용하면, 이 비율은 아래와 같은 수식으로 계산됩니다.
$$
\beta(x) = \frac{1/(1 + \exp(-f(x)))}{\exp(-f(x)/(1 + \exp(-f(x)))} = \exp(f(x))
$$

As a result, we need to solve two problems: first one to distinguish between data drawn from both distributions, and then a reweighted minimization problem where we weigh terms by $\beta$, e.g. via the head gradients. Here's a prototypical algorithm for that purpose which uses an unlabeled training set $X$ and test set $Z$:

결론적으로 우리는 두 문제를 풀어야합니다. 첫번째 문제는 두 분포에서 추출된 데이터를 구분하는 것이고, 두번째는 가중치를 다시 적용한 최소화 문제입니다. 가중치 조정은  $\beta$ 를 이용하는데, 이는 head gradient를 이용합니다. Label이 없는 학습 셋 $X$ 와 테스트  셋 $Z$ 을 사용하는 프로토타입의 알고리즘은 아래와 같습니다.

1. Generate training set with $\{(x_i, -1) ... (z_j, 1)\}$
1. Train binary classifier using logistic regression to get function $f$
1. Weigh training data using $\beta_i = \exp(f(x_i))$ or better $\beta_i = \min(\exp(f(x_i)), c)$
1. Use weights $\beta_i$ for training on $X$ with labels $Y$
1. 학습 셋  $\{(x_i, -1) ... (z_j, 1)\}$ 을 생성합니다.
1. Logistic regression을 이용해서 binary 분류기를 학습시킵니다. 이를 함수 $f$ 라고 하겠습니다.
1. $\beta_i = \exp(f(x_i))$ 또는 $\beta_i = \min(\exp(f(x_i)), c)$ 를 이용해서 학습 데이터에 가중치를 적용합니다.
1. 데이터 $X$ 와 이에 대한 label $Y$ 에 대한 학습을 수행할 때, weight  $\beta_i$ 를 이용합니다.

**Generative Adversarial Networks** use the very idea described above to engineer a *data generator* such that it cannot be distinguished from a reference dataset. For this, we use one network, say $f$ to distinguish real and fake data and a second network $g$ that tries to fool the discriminator $f$ into accepting fake data as real. We will discuss this in much more detail later.

**Generative Adversarial Networks** 는 위에서 설명한 아이디어를 이용해서, 참조 데이터 셋과 구분이 어려운 데이터를 만드는 *데이터 생성기(data generator)*를 만듭니다. 네트워크 $f$ 는 진짜와 가짜 데이터는 구분하고, 다른 네트워크 $g$ 는 판정하는 역할을 하는 $f$ 를 속이는 역할, 즉 가짜 데이터를 진짜라고 판별하도록하는 역할을 수행합니다. 이에 대한 자세한 내용은 다시 다루겠습니다.

## Concept Shift Correction

## Concept Shift 교정

Concept shift is much harder to fix in a principled manner. For instance, in a situation where suddenly the problem changes from distinguishing cats from dogs to one of distinguishing white from black animals, it will be unreasonable to assume that we can do much better than just training from scratch using the new labels. Fortunately, in practice, such extreme shifts almost never happen. Instead, what usually happens is that the task keeps on changing slowly. To make things more concrete, here are some examples:

Concept shift는 개념적으로 해결하기 훨씬 어렵습니다. 예를 들면, 고양이와 강아지를 구분하는 문제에서 흰색과 검은색 동물을 구분하는 문제로 갑자기 바뀌었다고 하면, 새로운 label을 이용해서 새로 학습을 시키는 것보다 더 잘 동작시키는 것을 기대하는 것은 무리일 것입니다. 다행히, 실제 상황에서는 이렇게 심한 변화는 발생하지 않습니다. 대신, 변화가 천천히 일어나는 것이 보통의 경우입니다. 더 정확하게 하기 위해서, 몇가지 예를 들어보겠습니다.

* In computational advertising, new products are launched, old products become less popular. This means that the distribution over ads and their popularity changes gradually and any click-through rate predictor needs to change gradually with it.
* Traffic cameras lenses degrade gradually due to environmental wear, affecting image quality progressively.
* News content changes gradually (i.e. most of the news remains unchanged but new stories appear).
* 광고에서 새로운 상품이 출시되고, 이전 상품의 인기는 떨어집니다. 즉, 광고의 분포와 인기도는 서서히 변화되기 때문에, click-through rate 예측 모델은 그에 따라서 서서히 바뀌어야 합니다.
* 교통 카메라 렌즈는 환경의 영향으로 서서히 성능이 떨어지게 되고, 그 결과 이미지 품질에 영향을 미칩니다.
* 뉴스 내용이 서서히 바뀝니다. (즉, 대부분의 뉴스는 바뀌지 않지만, 새로운 이야기가 추가됩니다.)

In such cases, we can use the same approach that we used for training networks to make them adapt to the change in the data. In other words, we use the existing network weights and simply perform a few update steps with the new data rather than training from scratch.

이런 경우에 네트워크 학습에 사용한 것과 같은 방법을 데이터의 변화에 적응시키는 데 사용할 수 있습니다. 즉, 네트워크를 처음부터 다시 학습시키는 것이 아니라, 현재 weight 값을 갖는 네트워크에 새로이 추가된 데이터를 이용해서 학습시키는 것입니다. 

## A Taxonomy of Learning Problems

## 학습 문제의 분류

Armed with knowledge about how to deal with changes in $p(x)​$ and in $p(y|x)​$, let us consider a number of problems that we can solve using machine learning.

 $p(x)$ 과  $p(y|x)$ 이 바뀔 때 어떻게 다뤄야하는지에 대해서 알아봤으니, 머신러닝을 이용해서 풀 수 있는 여러가지 문제들에 대해서 알아보겠습니다.

* **Batch Learning.** Here we have access to training data and labels $\{(x_1, y_1), \ldots (x_n, y_n)\}$, which we use to train a network $f(x,w)$. Later on, we deploy this network to score new data $(x,y)$ drawn from the same distribution. This is the default assumption for any of the problems that we discuss here. For instance, we might train a cat detector based on lots of pictures of cats and dogs. Once we trained it, we ship it as part of a smart catdoor computer vision system that lets only cats in. This is then installed in a customer's home and is never updated again (barring extreme circumstances).
* **배치 러닝**. 학습 데이터와 레이블 쌍  $\{(x_1, y_1), \ldots (x_n, y_n)\}​$ 을 사용해서 네트워크 $f(x,w)​$ 를 학습시킨다고 생각해봅니다. 모델을 학습시킨 후, 학습 데이터와 같은 분포에서 새로운 데이터 $(x,y)​$ 를  뽑아서 이 모델에 적용합니다. 우리가 여기서 논의하는 대부분의 문제는 이 기본적인 가정을 포함하고 있습니다. 예를 들면, 고양이와 강아지 사진을 사용해서 고양이 탐지 모델을 학습시킵니다. 모델을 학습시킨 후, 고양이만 들어올 수 있도록 하는 컴퓨터 비전을 이용한 고양이 전용 문 시스템에 이 모델을 사용합니다. 이 시스템을 고객의 가정에 설치한 후에 모델을 다시 업데이트하지 않습니다.
* **Online Learning.** Now imagine that the data $(x_i, y_i)$ arrives one sample at a time. More specifically, assume that we first observe $x_i$, then we need to come up with an estimate $f(x_i,w)$ and only once we've done this, we observe $y_i$ and with it, we receive a reward (or incur a loss), given our decision. Many real problems fall into this category. E.g. we need to predict tomorrow's stock price, this allows us to trade based on that estimate and at the end of the day we find out whether our estimate allowed us to make a profit. In other words, we have the following cycle where we are continuously improving our model given new observations.
* **온라인 러닝**. 데이터 $(x_i, y_i)​$ 가 한번에 하나씩 들어오는 것을 가정합니다. 조금 더 명확하게 말하자면, 우선 $x_i​$ 가 관찰되면, $f(x_i,w)​$ 를 통해서 추측을 수행 한 이후에만  $y_i​$ 를 알 수 있는 경우를 가정합니다. 이 후, 추측 결과에 대한 보상 또는 loss 를 계산합니다. 많은 실제 문제가 이러한 분류에 속합니다. 예를 들면, 다음 날의 주식 가격을 예측하는 경우를 생각해보면, 예측된 주가에 근거해서 거래를 하고, 그날의 주식시장이 끝나면 예측이 수익을 가져다 줬는지 알 수 있습니다. 달리 말하면, 새로운 관찰을 통해서 모델을 지속적으로 발전시키는 다음과 같은 사이클을 만들 수 있습니다.

$$
\mathrm{model} ~ f_t \longrightarrow
\mathrm{data} ~ x_t \longrightarrow
\mathrm{estimate} ~ f_t(x_t) \longrightarrow
\mathrm{observation} ~ y_t \longrightarrow
\mathrm{loss} ~ l(y_t, f_t(x_t)) \longrightarrow
\mathrm{model} ~ f_{t+1}
$$

* **Bandits.** They are a *special case* of the problem above. While in most learning problems we have a continuously parametrized function $f$ where we want to learn its parameters (e.g. a deep network), in a bandit problem we only have a finite number of arms that we can pull (i.e. a finite number of actions that we can take). It is not very surprising that for this simpler problem stronger theoretical guarantees in terms of optimality can be obtained. We list it mainly since this problem is often (confusingly) treated as if it were a distinct learning setting.
* **반딧**. 반딧은 위 문제의 *특별한 경우*입니다. 대부분의 학습 문제는 연속된 값을 출력하는 함수  $f$ 의 파라메터(예를 들면 딥 네트워크)를 학습하는 경우이지만, 반딧 문제는 선택할 수 있는 종류가 유한한 경우 (즉, 취할 수 있는 행동이 유한한 경우)입니다. 이 간단한 문제의 경우, 최적화의 측면에서 강력한 이론적인 보증을 얻을 수 있다는 것이 당연합니다. 이 문제를 별도로 분류한 이유는 이 문제를 종종 distinct learning과 혼동하기 때문입니다.
* **Control (and nonadversarial Reinforcement Learning).** In many cases the environment remembers what we did. Not necessarily in an adversarial manner but it'll just remember and the response will depend on what happened before. E.g. a coffee boiler controller will observe different temperatures depending on whether it was heating the boiler previously. PID (proportional integral derivative) controller algorithms are a [popular choice](http://pidkits.com/alexiakit.html) there. Likewise, a user's behavior on a news site will depend on what we showed him previously (e.g. he will read most news only once). Many such algorithms form a model of the environment in which they act such as to make their decisions appear less random (i.e. to reduce variance).
* **Control (and nonadversarial Reinforcement Learning).** 많은 경우에 환경은 우리가 취한 행동을 기억합니다. 적의적인 의도가 아닌 경우에도, 단순히 기억하고, 이전에 일어난 일에 근거해서 반응하는 환경들이 있습니다. 즉, 커피 포트 제어기의 경우 이전에 데웠는지 여부에 따라서 다른 온도를 감지하기도 합니다. PID (Propotional Integral Derivative) 제어 알고리즘도 [유명한 예](http://pidkits.com/alexiakit.html)입니다. 비슷한 예로, 뉴스 사이트에 대한 사용자의 행동은 이전에 무엇을 무엇이었는지 영향을 받습니다. 이런 종류의 많은 알고리즘들은 그 결정들이 임의의 선택으로 보이지 않도록 모델을 만들어냅니다. (즉, 분산을 줄이는 방향으로)
* **Reinforcement Learning.** In the more general case of an environment with memory, we may encounter situations where the environment is trying to *cooperate* with us (cooperative games, in particular for non-zero-sum games), or others where the environment will try to *win*. Chess, Go, Backgammon or StarCraft are some of the cases. Likewise, we might want to build a good controller for autonomous cars. The other cars are likely to respond to the autonomous car's driving style in nontrivial ways, e.g. trying to avoid it, trying to cause an accident, trying to cooperate with it, etc.
* **강화 학습**. 기억을 하는 환경의 더 일반적인 예로 우리와 협력을 시도하는 환경(non-zero-sum 게임과 같이 협력적인 게임)이나 이기려고 하는 환경이 있습니다. 체스나, 바둑, 서양주사위놀이(Backgammon) 또는 스타크래프트가 경쟁하는 환경의 예들입니다. 마찬가지로, 자율주행차를 위한 좋은 제어기를 만드는 것도 생각해볼 수 있습니다. 이 경우 다른 차량들은 자율주행차의 운전 스타일에 여러가지로 반응을 합니다. 때로는 피하려고 하거나, 사고를 내려고 하거나, 같이 잘 주행하려고 하는 등 여러 반응을 보일 것입니다.

One key distinction between the different situations above is that the same strategy that might have worked throughout in the case of a stationary environment, might not work throughout when the environment can adapt. For instance, an arbitrage opportunity discovered by a trader is likely to disappear once he starts exploiting it. The speed and manner at which the environment changes determines to a large extent the type of algorithms that we can bring to bear. For instance, if we *know* that things may only change slowly, we can force any estimate to change only slowly, too. If we know that the environment might change instantaneously, but only very infrequently, we can make allowances for that. These types of knowledge are crucial for the aspiring data scientist to deal with concept shift, i.e. when the problem that he is trying to solve changes over time.

위에 설명한 다양한 상황들 간의 주요 차이점은 안정적인 환경에서 잘 작동하는 전략이 환경이 변화는 상황에서는 잘 작동하지 않을 수 있다는 것입니다. 예를 들면, 거래자가 발견한 차익 거래 기회는 한번 실행되면 사라질 가능성이 높습니다. 환경이 변화하는 속도나 형태는 계속해서 사용할 수 있는 알고리즘의 형태를 많이 제약합니다. 예를 들면, 어떤 것이 천천히 변화할 것이라고 알고 있을 경우, 예측 모델 또한 천천히 바뀌도록 할 수 있습니다. 만약, 환경이 불규적으로 순간적으로 바뀐다고 알고 있는 경우에는, 이에 대응 하도록 만들 수 있습니다. 이런 종류의 지식은 풀고자 하는 문제가 시간에 따라서 바뀌는 상황, 즉 concerpt shit를 다루는 야심 찬 데이터 사이언티스트에게 아주 중요합니다.

## Summary

## 요약

* In many cases training and test set do not come from the same distribution. This is called covariate shift.
* Covariate shift can be detected and corrected if the shift isn't too severe. Failure to do so leads to nasty surprises at test time.
* In some cases the environment *remembers* what we did and will respond in unexpected ways. We need to account for that when building models.
* 많은 경우에 학습 세트와 테스트 세트는 같은 분포로부터 얻어지지 않습니다. 이런 상황을 우리는 covariate shift 라고 합니다.
* Covariate shift는 shift 가 아주 심하지 않을 경우에 탐지하고 이를 교정할 수 있습니다. 만약 그렇게 하지 못하면, 테스트 시점에 좋지 않은 결과가 나옵니다.
* 어떤 경우에는 환경이 우리가 취한 것을 기억하고, 예상하지 못한 방법으로 결과를 줄 수도 있습니다. 모델을 만들에 이점을 유의해야합니다.

## Problems

## 문제

1. What could happen when we change the behavior of a search engine? What might the users do? What about the advertisers?
1. Implement a covariate shift detector. Hint - build a classifier.
1. Implement a covariate shift corrector.
1. What could go wrong if training and test set are very different? What would happen to the sample weights?
1. 검색 엔진의 행동을 바꾸면 어떤 일이 일어날까요? 사용자는 어떻게 반응할까요? 광고주는 어떨까요?
1. Covariate shift 탐기지를 구현해보세요. 힌트 - 분류기를 만들어 봅니다.
1. Covariate shift 교정기를 구현해보세요.
1. 학습 세트와 테스트 세트가 많이 다를 경우 무엇이 잘못될 수 있을까요? 샘플 weight들에는 어떤 일이 일어날까요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2347)

![](../img/qr_environment.svg)
