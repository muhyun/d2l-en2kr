# Introduction to Deep Learning 

# 딥러닝 소개

In 2016 Joel Grus, a well-known data scientist went for a [job interview](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/) at a major internet company. As is common, one of the interviewers had the task to assess his programming skills. The goal was to implement a simple children's game - FizzBuzz. In it, the player counts up from 1, replacing numbers divisible by 3 by 'fizz' and those divisible by 5 by 'buzz'. Numbers divisible by 15 result in 'FizzBuzz'. That is, the player generates the sequence

2016년 Joel Grus, 잘 알려진 데이터 과학자가 주요 인터넷 회사의 [job interview](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/) 에 갔습니다. 일반적으로 하듯이, 인터뷰 담당자는 그의 프로그래밍 기술을 평가하는 문제를 냈습니다. 간단한 어린이 게임인 FizzzBuzz를 구현하는 것이 과제였습니다. 그 안에서 플레이어는 1부터 카운트하면서 3으로 나눌 수 있는 숫자를 'fizz' 로, 5로 나눌 수있는 숫자를 'buzz' 로 바꿉니다. 15로 나눌 수 있는 숫자는 'FizzBuzz' 가 됩니다. 즉, 플레이어는 시퀀스를 생성합니다.

```
1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 ...
```

What happened was quite unexpected. Rather than solving the problem with a few lines of Python code *algorithmically*, he decided to solve it by programming with data. He used pairs of the form (3, fizz), (5, buzz), (7, 7), (2, 2), (15, fizzbuzz) as examples to train a classifier for what to do. Then he designed a small neural network and trained it using this data, achieving pretty high accuracy (the interviewer was nonplussed and he did not get the job).

전혀 예상하지 못한 일이 벌어졌습니다. 겨의 몇 줄의 Python 코드로 *알고리즘* 을 구현해서 문제를 해결하는 대신, 그는 데이터를 활용한 프로그램으로 문제를 풀기로 했습니다. 그는 (3, fizz), (5, buzz), (7, 7), (2, 2), (15, fizzbuzz) 의 쌍을 활용하여 분류기를 학습시켰습니다. 그가 작은 뉴럴 네트워크(neural network)를 만들고, 그것을 이 데이터를 활용하여 학습시켰고 그 결과 꽤 높은 정확도를 달성하였습니다. (면접관은 좋은 점수를 주지 않아서, 그는 그 잡을 얻지 못했습니다).

Situations such as this interview are arguably watershed moments in computer science when program design is supplemented (and occasionally replaced) by programming with data. They are significant since they illustrate the ease with which it is now possible to accomplish these goals (arguably not in the context of a job interview). While nobody would seriously solve FizzBuzz in the way described above, it is an entirely different story when it comes to recognizing faces, to classify the sentiment in a human's voice or text, or to recognize speech. Due to good algorithms, plenty of computation and data, and due to good software tooling it is now within the reach of most software engineers to build sophisticated models, solving problems that only a decade ago were considered too challenging even for the best scientists.

이 인터뷰와 같은 상황은 프로그램 설계가 데이터로 프로그래밍하여 보완 (때로는 대체) 되는 컴퓨터 과학의 획기적인 순간입니다. 그것들은 이러한 목표를 달성 할 수 있는 용이성을 설명하기 때문에 중요합니다 (틀림없이 면접의 맥락에서는 아니고). 누구든 위에서 설명한 방식으로 FizzBuzz를 심각하게 해결하지는 않겠지만, 얼굴을 인식하거나, 사람의 목소리 또는 텍스트로 감정을 분류하거나, 음성을 인식할 때는 완전히 다른 이야기입니다. 좋은 알고리즘, 많은 연산 장치 및 데이터, 그리고 좋은 소프트웨어 도구들로 인해 이제는 대부분의 소프트웨어 엔지니어가 단지 10년전 최고의 과학자들에게도 너무 도전적이라고 여겨졌던 문제를 해결하는 정교한 모델을 만들 수게 되었습니다.

This book aims to help engineers on this journey. We aim to make machine learning practical by combining mathematics, code and examples in a readily available package. The Jupyter notebooks are available online, they can be executed on laptops or on servers in the cloud. We hope that they will allow a new generation of programmers, entrepreneurs, statisticians, biologists, and anyone else who is interested to deploy advanced machine learning algorithms to solve their problems.

이 책은 이 여정에 엔지니어를 돕는 것을 목표로 합니다. 우리는 수학, 코드 및 예제를 쉽게 사용할 수 있는 패키지로 결합하여 머신러닝을 실용적으로 만드는 것을 목표로 합니다. Jupyter 노트북은 온라인으로 제공되고, 예제들을 랩톱 또는 클라우드의 서버에서 실행할 수 있습니다. 우리는 이를 통해서 새로운 세대의 프로그래머, 기업가, 통계학자, 생물 학자 및 고급 머신러닝 알고리즘을 배포하는 데 관심이 있는 사람들이 문제를 해결할 수 있기를 바랍니다.

## Programming with Data

## 데이터를 활용하는 프로그래밍

Let us delve into the distinction between programing with code and programming with data into a bit more detail, since it is more profound than it might seem. Most conventional programs do not require machine learning. For example, if we want to write a user interface for a microwave oven, it is possible to design a few buttons with little effort. Add some logic and rules that accurately describe the behavior of the microwave oven under various conditions and we're done. Likewise, a program checking the validity of a social security number needs to test whether a number of rules apply. For instance, such numbers must contain 9 digits and not start with 000.

코드를 이용하는 프로그래밍과 데이터를 활용하는 프로그래밍의 차이점을 좀 더 자세히 살펴 보겠습니다. 이 둘은 보이는 것보다 더 심오하기 때문입니다. 대부분의 전통적인 프로그램은 머신러닝을 필요로 하지 않습니다. 예를 들어 전자 레인지용 사용자 인터페이스를 작성하려는 경우 약간의 노력으로 몇 가지 버튼을 설계 할 수 있습니다. 다양한 조건에서 전자 레인지의 동작을 정확하게 설명하는 몇 가지 논리와 규칙을 추가하면 완료됩니다. 마찬가지로 사회 보장 번호의 유효성을 확인하는 프로그램은 여러 규칙이 적용되는지 여부를 테스트하면 됩니다. 예를 들어, 이러한 숫자는 9 자리 숫자를 포함해야 하며 000으로 시작하지 않아야 한다와 같은 규칙입니다.

It is worth noting that in the above two examples, we do not need to collect data in the real world to understand the logic of the program, nor do we need to extract the features of such data. As long as there is plenty of time, our common sense and algorithmic  skills are enough for us to complete the tasks.

위의 두 가지 예에서 프로그램의 논리를 이해하기 위해 현실 세계에서 데이터를 수집할 필요가 없으며, 그 데이터의 특징을 추출 할 필요가 없다는 점은 주목할 가치가 있습니다. 많은 시간이 있다면, 우리의 상식과 알고리즘 기술은 우리가 작업을 완료하기에 충분합니다.

As we observed before, there are plenty of examples that are beyond the abilities of even the best programmers, yet many children, or even many animals are able to solve them with great ease. Consider the problem of detecting whether an image contains a cat. Where should we start? Let us further simplify this problem: if all  images are assumed to have the same size (e.g. 400x400 pixels) and each pixel consists of values for red, green and blue, then an image is represented by 480,000 numbers. It is next to impossible to decide where the relevant information for our cat detector resides.  Is it the average of all the values, the values of the four corners, or is it a particular point in the image? In fact, to interpret the content in an image, you need to look for features that only appear when you combine thousands of values, such as edges, textures, shapes, eyes, noses. Only then will one be able to determine whether the image contains a cat.

우리가 전에 관찰 한 바와 같이, 심지어 최고의 프로그래머의 능력을 넘는 많은 예가 있지만, 많은 아이들, 심지어 많은 동물들이 쉽게 그들을 해결할 수 있습니다. 이미지에 고양이가 포함되어 있는지 여부를 감지하는 문제를 고려해보겠습니다. 어디서부터 시작해야 할까요? 이 문제를 더욱 단순화해 보겠습니다. 모든 이미지가 동일한 크기 (예, 400x400 픽셀)이라고 가정하고, 각 픽셀이 빨강, 녹색 및 파랑 값으로 구성된 경우 이미지는 480,000 개의 숫자로 표시됩니다. 우리의 고양이 탐지기가 관련된 정보가 어디에 있는지 결정하는 것은 불가능합니다. 그것은 모든 값의 평균일까요? 네 모서리의 값일까요? 아니면 이미지의 특정 지점일까요? 실제로 이미지의 내용을 해석하려면 가장자리, 질감, 모양, 눈, 코와 같은 수천 개의 값을 결합 할 때만 나타나는 특징을 찾아야합니다. 그래야만 이미지에 고양이가 포함되어 있는지 여부를 판단 할 수 있습니다.

An alternative strategy is to start by looking for a solution based on the final need, i.e. by *programming with data*, using examples of images and desired responses (cat, no cat) as a starting point.  We can collect real images of cats (a popular motif on the internet) and beyond. Now our goal translates into finding a function that can *learn* whether the image contains a cat. Typically the form of the function, e.g. a polynomial, is chosen by the engineer, its parameters are *learned* from data.

다른 전략은 최종 필요성에 기반한 솔루션을 찾는 것입니다. 즉, 이미지 예제 및 원하는 응답 (cat, cat 없음) 을 출발점으로 사용하여 *데이터로 프로그래밍하는 것*입니다. 우리는 고양이의 실제 이미지 (인터넷에서 인기있는 주제)들과 다른 것들을 수집 할 수 있습니다. 이제 우리의 목표는 이미지에 고양이가 포함되어 있는지 여부를 *배울 수 있는* 함수를 찾는 것입니다. 일반적으로 함수의 형태 (예, 다항식)는 엔지니어에 의해 선택되며 그 함수의 파라메터들은 데이터에서 *학습 됨*니다.

In general, machine learning deals with a wide class of functions that can be used in solving problems such as that of cat recognition. Deep learning, in particular, refers to a specific class of functions that are inspired by neural networks, and a specific way of training them (i.e. computing the parameters of such functions). In recent years, due to big data and powerful hardware,  deep learning has gradually become the de facto choice for processing complex high-dimensional data such as images, text and audio signals.

일반적으로 머신러닝은 고양이 인식과 같은 문제를 해결하는 데 사용할 수 있는 다양한 종류의 함수를 다룹니다. 딥 러닝은 특히 신경망에서 영감을 얻은 특정 함수의 클래스를 사용해서, 이것을 특별한 방법으로 학습(함수의 파라메터를 계산하는 것)시키는 방법입니다. 최근에는 빅 데이터와 강력한 하드웨어로 인해 이미지, 텍스트, 오디오 신호 등 복잡한 고차원 데이터를 처리하는 데 있어 딥 러닝이 사실상의 표준으로(de facto choice) 자리잡았습니다.

## Roots

## 기원

Although deep learning is a recent invention, humans have held the desire to analyze data and to predict future outcomes for centuries. In fact, much of natural science has its roots in this. For instance, the Bernoulli distribution is named after [Jacob Bernoulli (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli), and the Gaussian distribution was discovered by [Carl Friedrich Gauss (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss). He invented for instance the least mean squares algorithm, which is still used today for a range of problems from insurance calculations to medical diagnostics. These tools gave rise to an experimental approach in natural sciences - for instance, Ohm's law relating current and voltage in a resistor is perfectly described by a linear model.

딥 러닝은 최근 발명품이지만, 인간은 데이터를 분석하고 미래의 결과를 예측하려는 욕구를 수세기 동안 가지고 있어왔습니다. 사실, 자연 과학의 대부분은 이것에 뿌리를 두고 있습니다. 예를 들어, 베르누이 분포는 [야곱 베르누이 (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli) 의 이름을 따서 명명되었으며, 가우시안 분포는 [칼 프리드리히 가우스 (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) 에 의해 발견되었습니다. 예를 들어, 그는 최소 평균 제곱 알고리즘을 발명했는데, 이것은 보험 계산부터 의료 진단까지 다양한 문제들에서 오늘날에도 사용되고 있습니다. 이러한 도구는 자연 과학에서 실험적인 접근법을 일으켰습니다. 예를 들어 저항기의 전류 및 전압에 관한 옴의 법칙은 선형 모델로 완벽하게 설명됩니다.

Even in the middle ages mathematicians had a keen intuition of estimates. For instance, the geometry book of [Jacob Köbel (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry) illustrates averaging the length of 16 adult men's feet to obtain the average foot length.

중세 시대에도 수학자들은 예측에 대한 예리한 직관을 가지고있었습니다. 예를 들어, [야곱 쾨벨 (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry) 의 기하학 책에서는 발의 평균 길이를 얻기 위해 성인 남성 발 16개의 평균을 보여줍니다.

![Estimating the length of a foot](../img/koebel.jpg)

Figure 1.1 illustrates how this estimator works. 16 adult men were asked to line up in a row, when leaving church. Their aggregate length was then divided by 16 to obtain an estimate for what now amounts to 1 foot. This 'algorithm' was later improved to deal with misshapen feet - the 2 men with the shortest and longest feet respectively were sent away, averaging only over the remainder. This is one of the earliest examples of the trimmed mean estimate.

그림 1.1은 이 추정기가 어떻게 작동하는지 보여줍니다. 16 명의 성인 남성은 교회를 떠날 때 연속으로 정렬하도록 요구 받았습니다. 그런 다음 총 길이를 16으로 나누어 현재 1피트 금액에 대한 추정치를 얻습니다. 이 '알고리즘' 은 나중에 잘못된 모양의 발을 다루기 위해 개선되었습니다 - 각각 가장 짧고 긴 발을 가진 2 명의 남성은 제외하고 나머지 발들에 대해서만 평균값을 계산합니다. 이것은 절사 평균 추정치의 초기 예 중 하나입니다.

Statistics really took off with the collection and availability of data. One of its titans, [Ronald Fisher (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher), contributed significantly to its theory and also its applications in genetics. Many of his algorithms (such as Linear Discriminant Analysis) and formulae (such as the Fisher Information Matrix) are still in frequent use today (even the Iris dataset that he released in 1936 is still used sometimes to illustrate machine learning algorithms).

통계는 실제로 데이터의 수집 및 가용성으로 시작되었습니다. 거장 중 한명인 [로널드 피셔 (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher) 는 이론과 유전학의 응용에 크게 기여했습니다. 그의 알고리즘들 (예, 선형 판별 분석)과 수식들(예, Fisher 정보 매트릭스)은 오늘날에도 여전히 자주 사용되고 있습니다 (1936년에 발표한 난초(Iris) 데이터셋도 머신러닝 알고리즘을 설명하는 데 사용되기도 합니다).

A second influence for machine learning came from Information Theory [(Claude Shannon, 1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) and the Theory of computation via [Alan Turing (1912-1954)](https://en.wikipedia.org/wiki/Alan_Turing). Turing posed the question "can machines think?” in his famous paper [Computing machinery and intelligence](https://www.jstor.org/stable/2251299) (Mind, October 1950). In what he described as the Turing test, a machine can be considered intelligent if it is difficult for a human evaluator to distinguish between the replies from a machine and a human being through textual interactions. To this day, the development of intelligent machines is changing rapidly and continuously.

머신러닝에 대한 두 번째 영향은 정보 이론 [(클로드 섀넌, 1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) 과 [앨런 튜링 (1912-1954)](https://en.wikipedia.org/wiki/Alan_Turing)의 계산 이론에서 나왔습니다. 튜링은 그의 유명한 논문, [기계 및 지능 컴퓨팅, Computing machinery and intelligence](https://www.jstor.org/stable/2251299) (Mind, 1950년10월)에서, 그는  “기계가 생각할 수 있습니까?” 라는 질문을 제기했습니다. 그는 튜링 테스트로 설명 된 것에서, 인간 평가자가 텍스트 상호 작용을 통해 기계와 인간의 응답을 구별하기 어려운 경우 ''기계는 지능적이다''라고 간주될 수 있습니다. 오늘날까지 지능형 기계의 개발은 신속하고 지속적으로 변화하고 있습니다.

Another influence can be found in neuroscience and psychology. After all, humans clearly exhibit intelligent behavior. It is thus only reasonable to ask whether one could explain and possibly reverse engineer these insights. One of the oldest algorithms to accomplish this was formulated by [Donald Hebb (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb).

또 다른 영향은 신경 과학 및 심리학에서 발견 될 수 있습니다. 결국, 인간은 분명히 지적인 행동을 나타냅니다. 따라서 이러한 통찰력을 설명하고 아마도 리버스 엔지니어링 할 수 있는지 여부를 묻는 것은 합리적입니다. 이를 달성하기위한 가장 오래된 알고리즘 중 하나는 [도널드 헤브 (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb) 에 의해 공식화 되었습니다.

In his groundbreaking book [The Organization of Behavior](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf) (John Wiley & Sons, 1949) he posited that neurons learn by positive reinforcement. This became known as the Hebbian learning rule. It is the prototype of Rosenblatt's perceptron learning algorithm and it laid the foundations of many stochastic gradient descent algorithms that underpin deep learning today: reinforce desirable behavior and diminish undesirable behavior to obtain good weights in a neural network.

그의 획기적인 책 [행동의 조직, The Organization of Behavior](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf) (John Wiley & Sons, 1949) 에서 그는 뉴런이 긍정적인 강화로 배울 것이라고 가정했습니다. 이것은 Hebbian 학습 규칙으로 알려지게 되었습니다. 그것은 Rosenblatt의 퍼셉트론 학습 알고리즘의 원형이며 오늘날 딥 러닝을 뒷받침하는 많은 stochastic gradient descent 알고리즘의 기초를 마련했습니다: 뉴럴 네트워크의 좋은 가중치를 얻기 위해 바람직한 행동은 강화하고 바람직하지 않은 행동을 감소시킵니다.

Biological inspiration is what gave Neural Networks its name. For over a century (dating back to the models of Alexander Bain, 1873 and James Sherrington, 1890) researchers have tried to assemble computational circuits that resemble networks of interacting neurons. Over time the interpretation of biology became more loose but the name stuck. At its heart lie a few key principles that can be found in most networks today:

생물학적 영감이 Neural Networ에 이 이름을 부여한 것입니다. 한 세기 이상 (알렉산더 베인, 1873 및 제임스 셰링턴의 모델로 거슬러 올라가는 1890) 연구자들은 상호 작용하는 뉴런들의  네트워크들과 유사한 계산 회로를 조립하려고 시도했습니다. 시간이 지남에 따라 생물학의 해석은 더 느슨해 졌지만 이름은 여전히 그대로 있었다. 오늘날 대부분의 네트워크에서 찾을 수 있는 몇 가지 주요 원칙이 핵심에 있습니다:

* The alternation of linear and nonlinear processing units, often referred to as 'layers'.
* The use of the chain rule (aka backpropagation) for adjusting parameters in the entire network at once.
* 선형 및 비선형 처리 유닛들의 교차, 종종 '레이어' 라고도 함. 
* 체인 규칙 (일명 역 전파) 을 사용하여 한 번에 전체 네트워크의 매개 변수를 조정합니다.

After initial rapid progress, research in Neural Networks languished from around 1995 until 2005. This was due to a number of reasons. Training a network is computationally very expensive. While RAM was plentiful at the end of the past century, computational power was scarce. Secondly, datasets were relatively small. In fact, Fisher's 'Iris dataset' from 1932 was a popular tool for testing the efficacy of algorithms. MNIST with its 60,000 handwritten digits was considered huge.

초기 급속한 진행 이후, 뉴럴 네트워크의 연구는 1995년경부터 2005년까지 쇠퇴했습니다. 여러 가지 이유들이 있습니다. 네트워크 학습은 매우 많은 계산량이 필요합니다. 지난 세기 말에 RAM이 풍부했지만 계산 능력이 부족했습니다. 둘째, 데이터셋이 상대적으로 작았습니다. 사실, 1932년의 Fisher의 '난초(Iris) 데이터셋'은 알고리즘의 효능을 테스트하는 데 널리 사용되는 도구였습니다. 60,000 개의 손으로 쓴 숫자들로 구성된 MNIST는 거대한 것으로 간주되었습니다.

Given the scarcity of data and computation, strong statistical tools such as Kernel Methods, Decision Trees and Graphical Models proved empirically superior. Unlike Neural Networks they did not require weeks to train and provided predictable results with strong theoretical guarantees.

데이터 및 계산의 부족한 경우, 커널 방법, 의사 결정 트리 및 그래픽 모델과 같은 강력한 통계 도구는 경험적으로 우수합니다. 뉴럴 네트워크들과는 달리 이것들은 훈련하는 데 몇 주가 걸리지 않고 강력한 이론적 보장으로 예측 가능한 결과를 제공했습니다.

## The Road to Deep Learning

## 딥 러닝으로의 길

Much of this changed with the ready availability of large amounts of data, due to the World Wide Web, the advent of companies serving hundreds of millions of users online, a dissemination of cheap, high quality sensors, cheap data storage (Kryder's law), and cheap computation (Moore's law), in particular in the form of GPUs, originally engineered for computer gaming. Suddenly algorithms and models that seemed computationally infeasible became relevant (and vice versa). This is best illustrated in the table below:

이 중 대부분은 때문에 월드 와이드 웹, 온라인 사용자의 수억 서비스를 제공하는 회사의 출현, 저렴한 고품질의 센서, 저렴한 데이터 저장 (Kryder의 법칙), 특히  원래 컴퓨터 게임을 위해 설계된 GPU의 형태의 저렴한 계산 (무어의 법칙)가 바로 사용할 수 있게 되자 많은 것들이 바뀌었습니다. 갑자기 계산이 불가능한 것처럼 보이는 알고리즘과 모델이 의미가 있게 되었습니다.(반대의 경우도 마찬가지입니다). 이것은 아래 표에 가장 잘 설명되어 있습니다.

|Decade연대|Dataset데이터셋|Memory메모리|Floating Point Calculations per Second초당 floating point 연산수|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (social network)|100 GB|1 PF (NVIDIA DGX-2)|

It is quite evident that RAM has not kept pace with the growth in data. At the same time, the increase in computational power has outpaced that of the data available. This means that statistical models needed to become more memory efficient (this is typically achieved by adding nonlinearities) while simultaneously being able to spend more time on optimizing these parameters, due to an increased compute budget. Consequently the sweet spot in machine learning and statistics moved from (generalized) linear models and kernel methods to deep networks. This is also one of the reasons why many of the mainstays of deep learning, such as Multilayer Perceptrons (e.g. McCulloch & Pitts, 1943), Convolutional Neural Networks (Le Cun, 1992), Long Short Term Memory (Hochreiter & Schmidhuber, 1997), Q-Learning (Watkins, 1989), were essentially 'rediscovered' in the past decade, after laying dormant for considerable time.

RAM이 데이터의 증가와 보조를 유지하지 않은 것은 매우 분명합니다. 동시에 계산 능력의 증가는 사용 가능한 데이터의 증가보다 앞서고 있습니다. 즉, 통계 모델은 메모리 효율성이 향상 되어야하고 (일반적으로 비선형을 추가하여 달성됨) 컴퓨팅 예산 증가로 인해 이러한 매개 변수를 최적화하는 데 더 많은 시간을 할애해햐야 했습니다. 결과적으로 머신러닝 및 통계의 스위트 스폿은 (일반화 된) 선형 모델 및 커널 방법에서 딥 네트워크로 이동했습니다. 이것은 다층 퍼셉트론(Multilayer Perceptron) (예, 맥컬록 & 피츠, 1943), 컨볼루션 뉴럴 네트워크(Convolutional Neural Network) (Le Cun, 1992), Long Short Tem Memory (Hochreiter & Schmidhuber, 1997), Q-러닝 (왓킨스, 1989) 과 같은 딥 러닝의 메인 스테이의 많은 것들이 상당 시간 동안 휴면기에 있다가 과서 10년간 다시 발견된 이유 중에 하나입니다.

The recent progress in statistical models, applications, and algorithms, has sometimes been likened to the Cambrian Explosion: a moment of rapid progress in the evolution of species. Indeed, the state of the art is not just a mere consequence of available resources, applied to decades old algorithms. Note that the list below barely scratches the surface of the ideas that have helped researchers achieve tremendous progress over the past decade.

통계 모델, 응용 프로그램 및 알고리즘의 최근 발전은 때때로 캄브리아 폭발 (Cambrian Dexplosion) 에 비유되었습니다: 종의 진화가 급속히 진행되는 순간입니다. 실제로, 가장 좋은 성과들(state of art)은 수십 년의 오래된 알고리즘에 적용된 사용 가능한 리소스의 단순한 결과가 아닙니다. 아래 목록은 연구자들이 지난 10년간 엄청난 진전을 달성하는 데 도움이 된 아이디어의 표면을 간신히 긁어낸 것들입니다.

* Novel methods for capacity control, such as Dropout [3] allowed for training of relatively large networks without the danger of overfitting, i.e. without the danger of merely memorizing large parts of the training data. This was achieved by applying noise injection [4] throughout the network, replacing weights by random variables for training purposes.
* Attention mechanisms solved a second problem that had plagued statistics for over a century: how to increase the memory and complexity of a system without increasing the number of learnable parameters. [5] found an elegant solution by using what can only be viewed as a learnable pointer structure. That is, rather than having to remember an entire sentence, e.g. for machine translation in a fixed-dimensional representation, all that needed to be stored was a pointer to the intermediate state of the translation process. This allowed for significantly increased accuracy for long sentences, since the model no longer needed to remember the entire sentence before beginning to generate sentences.
* Multi-stage designs, e.g. via the Memory Networks [6] and the Neural Programmer-Interpreter [7] allowed statistical modelers to describe iterative approaches to reasoning. These tools allow for an internal state of the deep network to be modified repeatedly, thus carrying out subsequent steps in a chain of reasoning, similar to how a processor can modify memory for a computation.
* Another key development was the invention of Generative Adversarial Networks [8]. Traditionally statistical methods for density estimation and generative models focused on finding proper probability distributions and (often approximate) algorithms for sampling from them. As a result, these algorithms were largely limited by the lack of flexibility inherent in the statistical models. The crucial innovation in GANs was to replace the sampler by an arbitrary algorithm with differentiable parameters. These are then adjusted in such a way that the discriminator (effectively a two-sample test) cannot distinguish fake from real data. Through the ability to use arbitrary algorithms to generate data it opened up density estimation to a wide variety of techniques. Examples of galloping Zebras [9] and of fake celebrity faces [10] are both testimony to this progress.
* In many cases a single GPU is insufficient to process the large amounts of data available for training. Over the past decade the ability to build parallel distributed training algorithms has improved significantly. One of the key challenges in designing scalable algorithms is that the workhorse of deep learning optimization, stochastic gradient descent, relies on relatively small minibatches of data to be processed. At the same time, small batches limit the efficiency of GPUs. Hence, training on 1024 GPUs with a minibatch size of, say 32 images per batch amounts to an aggregate minibatch of 32k images. Recent work, first by Li [11],  and subsequently by You et al. [12] and Jia et al. [13] pushed the size up to 64k observations, reducing training time for ResNet50 on ImageNet to less than 7 minutes. For comparison - initially training times were measured in the order of days.
* The ability to parallelize computation has also contributed quite crucially to progress in reinforcement learning, at least whenever simulation is an option. This has led to significant progress in computers achieving superhuman performance in Go, Atari games, Starcraft, and in physics simulations (e.g. using MuJoCo). See e.g. Silver et al. [18] for a description of how to achieve this in AlphaGo. In a nutshell, reinforcement learning works best if plenty of (state, action, reward) triples are available, i.e. whenever it is possible to try out lots of things to learn how they relate to each other. Simulation provides such an avenue.
* Deep Learning frameworks have played a crucial role in disseminating ideas. The first generation of frameworks allowing for easy modeling encompassed [Caffe](https://github.com/BVLC/caffe), [Torch](https://github.com/torch), and [Theano](https://github.com/Theano/Theano). Many seminal papers were written using these tools. By now they have been superseded by [TensorFlow](https://github.com/tensorflow/tensorflow), often used via its high level API [Keras](https://github.com/keras-team/keras), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2), and [Apache MxNet](https://github.com/apache/incubator-mxnet). The third generation of tools, namely imperative tools for deep learning, was arguably spearheaded by [Chainer](https://github.com/chainer/chainer), which used a syntax similar to Python NumPy to describe models. This idea was adopted by [PyTorch](https://github.com/pytorch/pytorch) and the [Gluon API](https://github.com/apache/incubator-mxnet) of MxNet. It is the latter that this course uses to teach Deep Learning.
* 드롭아웃 [3] 과 같은 새로운 용량 제어 방법, 즉 학습 데이터의 큰 부분을 암기하는 위험 없이 비교적 큰 네트워크의 학습이 허용됩니다. 이것은 학습 목적을 위해 무작위 변수로 가중치를 대체하여 네트워크 전체에 노이즈 주입 [4] 을 적용하여 달성되었습니다. 
* 주의 메커니즘은 1 세기 이상 통계를 괴롭히는 두 번째 문제를 해결했습니다: 수를 늘리지 않고 시스템의 메모리와 복잡성을 증가시키는 방법 학습 가능한 매개 변수. [5] 는 학습 가능한 포인터 구조로만 볼 수 있는 것을 사용하여 우아한 해결책을 찾았습니다. 즉, 전체 문장을 기억할 필요없이 (예: 고정 차원 표현의 기계 번역의 경우) 저장해야하는 모든 것은 번역 프로세스의 중간 상태에 대한 포인터였습니다. 이것은 문장을 생성하기 전에 모델이 더 이상 전체 문장을 기억할 필요가 없기 때문에 긴 문장의 정확도를 크게 높일 수 있었습니다. 
* 다단계 디자인 (예: 메모리 네트워크 [6] 및 신경 프로그래머 인터프리터 [7] 를 통해 통계 모델러가 추론에 대한 반복적 인 접근법. 이러한 도구는 딥 네트워크의 내부 상태가 따라서 프로세서가 계산을 위해 메모리를 수정하는 방법과 유사한 추론 체인의 후속 단계를 수행, 반복적으로 수정 될 수 있습니다. 
* 또 다른 주요 개발은 생성 적 대적 네트워크의 발명이었다 [8]. 밀도 추정 및 생성 모델에 대한 전통적으로 통계 방법은 적절한 확률 분포와 그들로부터 샘플링에 대한 (종종 근사) 알고리즘을 찾는 데 초점을 맞추었다. 결과적으로 이러한 알고리즘은 통계 모델에 내재 된 유연성이 부족하여 크게 제한되었습니다. GAN의 중요한 혁신은 샘플러를 차별화 가능한 매개 변수로 임의의 알고리즘으로 대체하는 것이었습니다. 그런 다음 판별 자 (사실상 두 표본 검정) 가 가짜와 실제 데이터를 구분할 수 없도록 조정됩니다. 임의의 알고리즘을 사용하여 데이터를 생성하는 기능을 통해 다양한 기술로 밀도 추정을 열었습니다. 질주하는 Zebras [9] 가짜 유명 인사 얼굴 [10] 의 예는 이 진척에 대한 증언입니다. 
* 대부분의 경우 단일 GPU는 교육에 사용할 수 있는 많은 양의 데이터를 처리하기에 충분하지 않습니다. 지난 10년간 병렬 분산 교육 알고리즘을 구축하는 능력이 크게 향상되었습니다. 확장 가능한 알고리즘을 설계할 때 가장 큰 과제 중 하나는 딥 러닝 최적화, 확률 그래디언트 하강의 주제가 처리해야 할 데이터의 상대적으로 작은 미니배치에 의존한다는 것입니다. 동시에 작은 배치는 GPU의 효율성을 제한합니다. 따라서 미니 배치 크기가 인 1024 GPU에 대한 교육은 배치 당 32 개의 이미지가 32k 이미지의 집계 미니 배치에 달한다고 말합니다. 최근 작업은 Li [11] 에 의해, 그리고 그 다음에 Yo etal. [12] 와 Jia 등에 의해. [13] 는 크기를 64K 관측으로 밀어 넣었으며, ImageNet에서 ResNet50의 교육 시간을 7 분 미만으로 단축했습니다. 비교를 위해 - 처음에는 훈련 시간은 일 단위의 측정이었습니다. 
* 계산을 병렬화하는 능력은 적어도 시뮬레이션이 옵션 일 때마다 보강 학습의 발전에 상당히 결정적으로 기여했습니다. 이로 인해 이동, 아타리 게임, 스타크래프트 및 물리 시뮬레이션 (예: MuJoco 사용) 에서 초인적 성능을 달성하는 컴퓨터가 크게 발전했습니다. AlphaGo에서이를 달성하는 방법에 대한 설명은 예를 들어 Silver et al. [18] 을 참조하십시오. 간단히 말해서, 강화 학습은 많은 (주, 행동, 보상) 트리플을 사용할 수있는 경우, 즉 서로 어떻게 관련되는지 배우기 위해 많은 것들을 시도 할 수있을 때마다 가장 잘 작동합니다. 시뮬레이션은 이러한 길을 제공합니다. 
* 딥 러닝 프레임워크는 아이디어를 전파하는 데 중요한 역할을 했습니다. [Caffe](https://github.com/BVLC/caffe), [Torch](https://github.com/torch), [Theano](https://github.com/Theano/Theano)는 쉽게 모델링할 수 있는 첫 번째 세대의 프레임 워크입니다. 많은 논문은 이러한 도구를 사용하여 작성되었습니다. 지금까지 그들은 [TensorFlow](https://github.com/tensorflow/tensorflow) 로 대체되었으며, 종종 높은 수준의 API [Keras](https://github.com/keras-team/keras), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2), [Apache MxNet](https://github.com/apache/incubator-mxnet). 3 세대 도구, 즉 딥 러닝을위한 필수 도구는 틀림없이 Python NumPy와 유사한 구문을 사용하여 모델을 설명하는 [Chainer](https://github.com/chainer/chainer) 에 의해 선두되었습니다. 이 아이디어는 [PyTorch](https://github.com/pytorch/pytorch) 와 MxNet의 [Gluon API](https://github.com/apache/incubator-mxnet) 에 의해 채택되었습니다. 이 과정이 딥 러닝을 가르치는 데 사용하는 것은 후자입니다.

The division of labor between systems researchers building better tools for training and statistical modelers building better networks has greatly simplified things. For instance, training a linear logistic regression model used to be a nontrivial homework problem, worthy to give to new Machine Learning PhD students at Carnegie Mellon University in 2014. By now, this task can be accomplished with less than 10 lines of code, putting it firmly into the grasp of programmers.

학습을 위한 더 나은 도구를 만드는 시스템 연구자와 더 좋은 네트워크를 만드는 통계 모델러 사이의 노동 분할은 크게 일을 단순화했습니다. 예를 들어, 선형 물류 회귀 모델을 학습하는 것은 2014 년에 카네기 멜론 대학에서 새로운 머신러닝 박사 과정 학생들에게 줄 중요한 숙제 문제로 사용 사용되었습니다. 하지만, 지금은 10줄 미만의 코드로 만들 수 있으며, 많은 프로그래머들은 이것을 이해할 수 있습니다.

## Success Stories

## 성공 스토리

Artificial Intelligence has a long history of delivering results that would be difficult to accomplish otherwise. For instance, mail is sorted using optical character recognition. These systems have been deployed since the 90s (this is, after all, the source of the famous MNIST and USPS sets of handwritten digits). The same applies to reading checks for bank deposits and scoring creditworthiness of applicants. Financial transactions are checked for fraud automatically. This forms the backbone of many e-commerce payment systems, such as PayPal, Stripe, AliPay, WeChat, Apple, Visa, MasterCard. Computer programs for chess have been competitive for decades. Machine learning feeds search, recommendation, personalization and ranking on the internet. In other words, artificial intelligence and machine learning are pervasive, albeit often hidden from sight.

인공 지능은 다른 방법으로는 달성하기 어려운 결과를 가져온 오랜 역사를 가지고 있습니다. 예를 들어 편지는 광학 문자 인식을 사용하여 정렬됩니다. 이 시스템은 90 년대부터 사용되었습니다 (결국 이것은 유명한 MNIST 및 USPS 필기 숫자 세트의 출처입니다). 은행 예금에 대한 수표를 읽고 신청자의 신용도를 채점하는 경우에도 마찬가지입니다. 금융 거래는 자동으로 사기를 확인합니다. 이것은 페이팔, 스트라이프, AliPay, 위챗, 애플, 비자, 마스터 카드와 같은 많은 전자 상거래 결제 시스템의 중추를 형성합니다. 체스용 컴퓨터 프로그램은 수십 년간 경쟁력을 유지해 왔습니다. 머신러닝은 피드 검색, 추천, 개인 설정 및 인터넷 순위에 사용되고 있습니다. 즉, 인공 지능과 머신러닝은 종종 시야에서 숨겨져 있지만 널리 퍼져 있습니다.

It is only recently that AI has been in the limelight, mostly due to solutions to problems that were considered intractable previously.

최근에야 AI가 각광을 받고 있는데, 대부분 이전에 다루기 어려운 문제들을 해결하고 있기 때문입니다.

* Intelligent assistants, such as Apple's Siri, Amazon's Alexa, or Google's assistant are able to answer spoken questions with a reasonable degree of accuracy. This includes menial tasks such as turning on light switches (a boon to the disabled) up to making barber's appointments and offering phone support dialog. This is likely the most noticeable sign that AI is affecting our lives.
* A key ingredient in digital assistants is the ability to recognize speech accurately. Gradually the accuracy of such systems has increased to the point where they reach human parity [14] for certain applications.
* Object recognition likewise has come a long way. Estimating the object in a picture was a fairly challenging task in 2010. On the ImageNet benchmark Lin et al. [15] achieved a top-5 error rate of 28%. By 2017 Hu et al. [16] reduced this error rate to 2.25%. Similarly stunning results have been achieved for identifying birds, or diagnosing skin cancer.
* Games used to be a bastion of human intelligence. Starting from TDGammon [23], a program for playing Backgammon using temporal difference (TD) reinforcement learning, algorithmic and computational progress has led to algorithms for a wide range of applications. Unlike Backgammon, chess has a much more complex state space and set of actions. DeepBlue beat Gary Kasparov, Campbell et al. [17], using massive parallelism, special purpose hardware and efficient search through the game tree. Go is more difficult still, due to its huge state space. AlphaGo reached human parity in 2015,  Silver et al. [18] using Deep Learning combined with Monte Carlo tree sampling. The challenge in Poker was that the state space is large and it is not fully observed (we don't know the opponents' cards). Libratus exceeded human performance in Poker using efficiently structured strategies; Brown and Sandholm [19]. This illustrates the impressive progress in games and the fact that advanced algorithms played a crucial part in them.
* Another indication of progress in AI is the advent of self-driving cars and trucks. While full autonomy is not quite within reach yet, excellent progress has been made in this direction, with companies such as [Momenta](https://www.momenta.ai/en), [Tesla](http://www.tesla.com), [NVIDIA](http://www.nvidia.com), [MobilEye](http://www.mobileye.com) and [Waymo](http://www.waymo.com) shipping products that enable at least partial autonomy. What makes full autonomy so challenging is that proper driving requires the ability to perceive, to reason and to incorporate rules into a system. At present, Deep Learning is used primarily in the computer vision aspect of these problems. The rest is heavily tuned by engineers.
* 애플의 시리, 아마존의 알렉사 또는 Google의 조수와 같은 지능형 조수는 합리적인 수준의 정확도로 음성 질문에 대답 할 수 있습니다. 여기에는 조명 스위치 켜기 (장애인을 위한 보온) 까지 이발사 약속 및 전화 지원 대화 상자 제공과 같은 중요한 작업이 포함됩니다. 이것은 인공지능이 우리 삶에 영향을 미치고 있다는 가장 눈에 띄는 신호일 것입니다.
* 디지털 보조자의 핵심 요소는 음성을 정확하게 인식하는 능력입니다. 점차적으로 이러한 시스템의 정확성은 인간의 패리티 [14] 특정 응용 프로그램에 대한. 
* 마찬가지로 객체 인식은 먼 길을왔다. 그림에서 개체를 추정하는 것은 2010 년에 상당히 어려운 작업이었습니다. ImageNet 벤치 마크 Lin 외. [15] 는 28% 의 상위 5 오류율을 달성했습니다. 2017 후 등. [16] 이 오류율을 2.25% 로 줄였습니다. 마찬가지로 놀라운 결과는 새를 식별하거나 피부암을 진단하기 위해 달성되었습니다. 
* 게임은 인간의 지능의 보루로 사용되었습니다. TDGammon에서 시작 [23], 시간 차이 (TD) 강화 학습, 알고리즘 및 계산 진행을 사용하여 주사위 놀이를 재생하는 프로그램은 광범위한 응용 프로그램을위한 알고리즘을 주도하고있다. 주사위 놀이와 달리 체스는 훨씬 더 복잡한 상태 공간과 일련의 행동을 가지고 있습니다. DeepBlue는 게리 카스파로프를 이길, 캠벨 등. [17], 게임 트리를 통해 대규모 병렬 처리, 특수 목적 하드웨어와 효율적인 검색을 사용하여. 이동은 거대한 상태 공간 때문에 여전히 더 어렵습니다. AlphaGo는 2015 년에 인간의 패리티를 달성, 실버 등. [18] 몬테 카를로 트리 샘플링과 결합 된 딥 러닝을 사용하여. 포커의 도전은 상태 공간이 크고 완전히 관찰되지 않는다는 것입니다 (우리는 상대방의 카드를 모릅니다). 천교는 효율적으로 구조화 된 전략을 사용하여 포커에서 인간의 성능을 초과; 브라운과 샌드 홀름 [19]. 이것은 게임의 인상적인 진전과 고급 알고리즘이 그들에게 중요한 역할을 했다는 사실을 보여줍니다. 
* AI의 진보의 또 다른 표시는 자율 주행 자동차와 트럭의 출현이다. 아직 완전한 자율성이 충분히 도달할 수 있는 것은 아니지만, [모멘타](http://www.momenta.com), [테슬라], [엔비디아](http://www.nvidia.com), [모바일아이](http://www.mobileye.com), [Waymo.com](http://www.waymo.com) 등의 회사들과 함께 이러한 방향으로 우수한 진전이 이루어졌습니다. 적어도 부분 자율성을 가능하게. 완전한 자율성을 너무 어렵게 만드는 것은 적절한 운전이 규칙을 인식하고 추론하고 시스템에 통합하는 능력이 필요하다는 것입니다. 현재 딥 러닝은 이러한 문제의 컴퓨터 비전 측면에서 주로 사용됩니다. 나머지는 엔지니어에 의해 많이 조정됩니다.

Again, the above list barely scratches the surface of what is considered intelligent and where machine learning has led to impressive progress in a field. For instance, robotics, logistics, computational biology, particle physics and astronomy owe some of their most impressive recent advances at least in parts to machine learning. ML is thus becoming a ubiquitous tool for engineers and scientists.

다시 말하지만, 위의 목록은 지능적인 것으로 간주되는 것과 머신러닝이 분야에서 인상적인 진전을 이끈 것을 간신히 긁어낸 것에 불과합니다. 예를 들어, 로봇 공학, 물류, 전산생물학, 입자 물리학, 천문학은 최소한 머신러닝에 있어서 가장 인상적인 최근의 발전의 일부에 빚지고 있습니다. 따라서 머신러닝은 엔지니어와 과학자를위한 범용적인 도구가 되고 있습니다. 

Frequently the question of the AI apocalypse, or the AI singularity has been raised in non-technical articles on AI. The fear is that somehow machine learning systems will become sentient and decide independently from their programmers (and masters) about things that directly affect the livelihood of humans. To some extent AI already affects the livelihood of humans in an immediate way - creditworthiness is assessed automatically, autopilots mostly navigate cars safely, decisions about whether to grant bail use statistical data as input. More frivolously, we can ask Alexa to switch on the coffee machine and she will happily oblige, provided that the appliance is internet enabled.

종종 AI 종말, 또는 인공 지능 특이성에 대한 질문은 AI에 대한 비기술적 인 기사에서 제기됩니다. 머신러닝 시스템이 지각을 갖게 될 것이고, 그것을 만든 프로그래머와는 독립적으로 인간의 생활에 직접적인 영향을 끼칠 것들을 결정할 것이라는 것을 두려워합니다. 어느 정도 AI는 이미 즉각적인 방식으로 인간의 삶에 영향을 미치고 있습니다. 신용도가 자동으로 평가되고, 오토파일럿(autopilot)은 주로 자동차를 안전하게 운전하며, 통계 데이터를 입력을 사용해서 보석 허용 여부를 결정하고 있습니다. 더 경솔하게, 우리는 Alexa에게 커피 머신을 켜달라고 요청할 수 있으며, Alexa는 어플라이언스에 연결되어 있다면, 요청을 수행할 합니다.

Fortunately we are far from a sentient AI system that is ready to enslave its human creators (or burn their coffee). Firstly, AI systems are engineered, trained and deployed in a specific, goal oriented manner. While their behavior might give the illusion of general intelligence, it is a combination of rules, heuristics and statistical models that underlie the design. Second, at present tools for general Artificial Intelligence simply do not exist that are able to improve themselves, reason about themselves, and that are able to modify, extend and improve their own architecture while trying to solve general tasks.

다행히도 우리는 인간 창조자를 노예로 만들거나 커피를 태울 준비가 된 지각 있는 AI 시스템과는 거리가 멀었 습니다. 첫째, AI 시스템은 특정 목표 지향적 방식으로 설계, 학습 및 배포됩니다. 그들의 행동은 일반적인 지능의 환상을 줄 수 있지만, 디자인의 기초가되는 규칙, 휴리스틱 및 통계 모델의 조합입니다. 둘째, 현재는 일반적인 일을 수행하면서 스스로 개선하고, 스스로에 대해서 사고, 스스로의 아키텍처를 개선확장하고 개선하는 일반적인 인공지능을 위한 도구는 존재하지 않습니다.

A much more realistic concern is how AI is being used in our daily lives. It is likely that many menial tasks fulfilled by truck drivers and shop assistants can and will be automated. Farm robots will likely reduce the cost for organic farming but they will also automate harvesting operations. This phase of the industrial revolution will have profound consequences on large swaths of society (truck drivers and shop assistants are some of the most common jobs in many states). Furthermore, statistical models, when applied without care can lead to racial, gender or age bias. It is important to ensure that these algorithms are used with great care. This is a much bigger concern than to worry about a potentially malevolent superintelligence intent on destroying humanity.

훨씬 더 현실적인 관심사는 AI가 일상 생활에서 어떻게 사용되는지입니다. 트럭 운전사 및 상점 보조자가 수행 하는 사소한 일들이 자동화 될 수 있고 자동화 될 가능성이 있습니다. 농장 로봇은 유기 농업 비용을 줄일 수 있있고, 또한 수확 작업을 자동화할 것입니다. 산업 혁명의 이 단계는 사회의 큰 무리에 깊은 결과를 가져올 것입니다. (트럭 운전사 및 상점 보조자는 많은 주에서 가장 일반적인 작업의 일부입니다). 더욱이 통계 모델이 주의없이 적용되면 인종적, 성별 또는 연령 편견이 발생할 수 있습니다. 이러한 알고리즘이 세심한주의를 가지고 사용되는지 확인하는 것이 중요합니다. 이것은 인류를 파괴하는 잠재적으로 악의적인 초지능의 의도에 대해 걱정하는 것보다 훨씬 더 큰 관심거리 입니다.

## Key Components

Machine learning uses data to learn transformations between examples. For instance, images of digits are transformed to integers between 0 and 9, audio is transformed into text (speech recognition), text is transformed into text in a different language (machine translation), or mugshots are transformed into names (face recognition). In doing so, it is often necessary to represent data in a way suitable for algorithms to process it. This degree of feature transformations is often used as a reason for referring to deep learning as a means for representation learning (in fact, the International Conference on Learning Representations takes its name from that). At the same time, machine learning equally borrows from statistics (to a very large extent questions rather than specific algorithms) and data mining (to deal with scalability).

머신러닝은 데이터를 사용하여 예제 간의 변환을 학습합니다. 예를 들어 숫자 이미지는 0에서 9 사이의 정수로 변환되고, 오디오는 텍스트(음성 인식)로 변환되고, 텍스트는 다른 언어의 텍스트로 변환되거나(기계 번역), 머그샷이 이름으로 변환됩니다(얼굴 인식). 그렇게 할 때, 알고리즘이 데이터를 처리하는 데 적합한 방식으로 데이터를 표현해야 하는 경우가 종종 있습니다. 이러한 특징 변환(feature transformation)의 정도는 표현 학습을 위한 수단으로 딥 러닝을 언급하는 이유로서 종종 사용됩니다 (사실, 국제 학습 표현 회의, the International Conference on Learning Representations,는 그 이름은 이것으로부터 나왔습니다). 동시에 머신러닝은 통계 (특정 알고리즘이 아닌 매우 큰 범위의 질문까지) 와 데이터 마이닝 (확장성 처리) 을 똑같이 사용합니다.

The dizzying set of algorithms and applications makes it difficult to assess what *specifically* the ingredients for deep learning might be. This is as difficult as trying to pin down required ingredients for pizza - almost every component is substitutable. For instance one might assume that multilayer perceptrons are an essential ingredient. Yet there are computer vision models that use only convolutions. Others only use sequence models.

현기증 나는 알고리즘 및 응용 프로그램 집합으로 인해 딥 러닝을 위한 성분이 무엇인지 *구체적으로* 평가하기가 어렵습니다. 이것은 피자에 필요한 재료를 고정시키는 것만큼 어렵습니다. 거의 모든 구성 요소는 대체 가능합니다. 예를 들어 다층 퍼셉트론이 필수 성분이라고 가정할 수 있습니다. 그러나 convolution 만 사용하는 컴퓨터 비전 모델이 있습니다. 다른 것들은 시퀀스 모델만 사용하기도 합니다.

Arguably the most significant commonality in these methods is the use of end-to-end training. That is, rather than assembling a system based on components that are individually tuned, one builds the system and then tunes their performance jointly. For instance, in computer vision scientists used to separate the process of feature engineering from the process of building machine learning models.
The Canny edge detector [20] and Lowe's SIFT feature extractor [21]  reigned supreme for over a decade as algorithms for mapping images into feature vectors. Unfortunately, there is only so much that humans can accomplish by ingenuity relative to a consistent evaluation over thousands or millions of choices, when carried out automatically by an algorithm. When deep learning took over, these feature extractors were replaced by automatically tuned filters, yielding superior accuracy.

틀림없이 이러한 방법에서 가장 중요한 공통점은 종단간(end-to-end) 학습을 사용하는 것입니다. 즉, 개별적으로 튜닝된 구성 요소를 기반으로 시스템을 조립하는 대신 시스템을 구축한 다음 성능을 공동으로 튜닝합니다. 예를 들어, 컴퓨터 비전 과학자들은 머신러닝 모델을 구축하는 과정과 특징 엔지니어링 프로세스를 분리하곤 했습니다. Canny 에지 검출기 [20] 와 Lowe의 SIFT 특징 추출기 [21] 는 이미지를 형상 벡터에 매핑하기 위한 알고리즘으로 10여 년간 최고로 통치했습니다. 불행히도 알고리즘에 의해 자동으로 수행 될 때 수천 또는 수백만 가지 선택에 대한 일관된 평가와 관련하여 인간이 독창성으로 성취 할 수있는 많은 것들이 있습니다. 딥 러닝이 적용되었을 때, 이러한 특징 추출기는 자동으로 튜닝된 필터로 대체되어 뛰어난 정확도를 달성했습니다.

Likewise, in Natural Language Processing the bag-of-words model of Salton and McGill [22] was for a long time the default choice. In it, words in a sentence are mapped into a vector, where each coordinate corresponds to the number of times that particular word occurs. This entirely ignores the word order ('dog bites man' vs. 'man bites dog') or punctuation ('let's eat, grandma' vs. 'let's eat grandma'). Unfortunately, it is rather difficult to engineer better features *manually*. Algorithms, conversely, are able to search over a large space of possible feature designs automatically. This has led to tremendous progress. For instance, semantically relevant word embeddings allow reasoning of the form 'Berlin - Germany + Italy = Rome' in vector space. Again, these results are achieved by end-to-end training of the entire system.

마찬가지로 자연 언어 처리에서 Salton과 McGill [22] 의 bag-of-words 모델은 오랫동안 기본으로 선택되었습니다. 여기서 문장의 단어는 벡터로 매핑되며 각 좌표는 특정 단어가 발생하는 횟수에 해당합니다. 이것은 단어 순서 ('개가 사람을 물었다' 대 '사람이 개를 물었다') 또는 구두점 ('먹자, 할머니' 대 '할머니를 먹자') 을 완전히 무시합니다. 불행히도, 더 나은 특징을 *수동으로* 엔지니어링하는 것은 다소 어렵습니다. 반대로 알고리즘은 가능한 특징(feature) 설계의 넓은 공간을 자동으로 검색 할 수 있습니다. 이것은 엄청난 진전을 이끌어 왔습니다. 예를 들어 의미상 관련성이 있는 단어 임베딩은 벡터 공간에서 '베를린 - 독일 + 이탈리아 = 로마' 형식의 추론을 허용합니다. 다시 말하지만, 이러한 결과는 전체 시스템의 end-to-end 학습을 통해 달성됩니다.

Beyond end-to-end training, the second most relevant part is that we are experiencing a transition from parametric statistical descriptions to fully nonparametric models. When data is scarce, one needs to rely on simplifying assumptions about reality (e.g. via spectral methods) in order to obtain useful models. When data is abundant, this can be replaced by nonparametric models that fit reality more accurately. To some extent, this mirrors the progress that physics experienced in the middle of the previous century with the availability of computers. Rather than solving parametric approximations of how electrons behave by hand, one can now resort to numerical simulations of the associated partial differential equations. This has led to much more accurate models, albeit often at the expense of explainability.

A case in point are Generative Adversarial Networks, where graphical models were replaced by data generating code without the need for a proper probabilistic formulation. This has led to models of images that can look deceptively realistic, something that was considered too difficult for a long time.

Another difference to previous work is the acceptance of suboptimal solutions, dealing with nonconvex nonlinear optimization problems, and the willingness to try things before proving them. This newfound empiricism in dealing with statistical problems, combined with a rapid influx of talent has led to rapid progress of practical algorithms (albeit in many cases at the expense of modifying and re-inventing tools that existed for decades).

Lastly, the Deep Learning community prides itself of sharing tools across academic and corporate boundaries, releasing many excellent libraries, statistical models and trained networks as open source. It is in this spirit that the notebooks forming this course are freely available for distribution and use. We have worked hard to lower the barriers of access for everyone to learn about Deep Learning and we hope that our readers will benefit from this.

End-to-end 학습 외에도 두 번째로 중요한 것은 파라메터 기반의 통계 설명에서 완전 비파라메터 기반의 모델로의 전환을 경험하고 있다는 것입니다. 데이터가 부족한 경우, 유용한 모델을 얻기 위해서는 현실에 대한 가정을 단순화하는 데 의존해야합니다 (예, 스펙트럼 방법을 통해). 데이터가 풍부하면 현실에 더 정확하게 맞는 비파라메터 기반의 모형으로 대체될 수 있습니다. 어느 정도, 이것은 컴퓨터의 가용성과 함께 이전 세기 중반에 물리학이 경험한 진전과 비슷합니다. 전자가  어떻게 동작하는지에 대한 파라메트릭 근사치를 직접 해결하는 대신, 이제 연관된 부분 미분 방정식의 수치 시뮬레이션에 의존 할 수 있습니다. 이것은 설명 가능성을 희생시키면서 종종 훨씬 더 정확한 모델을 이끌어 냈습니다. 

예를 들어 Generative Aversarial Networks가 있습니다. 그래픽 모델이 적절한 확률적 공식 없이도 데이터 생성 코드로 대체됩니다. 이것은 현혹적으로 현실적으로 보일 수있는 이미지의 모델을 이끌어 냈는데 이는 오랜 시간 동안 너무 어려운 것으로 여겨졌왔던 것입니다.

이전 작업의 또 다른 차이점은 볼록하지 않은 비선형 최적화 문제(nonconvex nonlinear optimization problem)를 다루면서 차선책 솔루션을 받아들이고, 이를 증명하기 전에 시도하려는 의지입니다. 통계적 문제를 다루는 새로운 경험주의와 인재의 급속한 유입은 실질적인 알고리즘의 급속한 발전으로 이어졌습니다 (많은 경우에도 불구하고 수십 년간 존재했던 도구를 수정하고 다시 발명하는 대신). 

마지막으로 딥 러닝 커뮤니티는 학술 및 기업 경계를 넘어 도구를 공유하는 것을 자랑으로 하고 있으며, 많은 우수한 라이브러리, 통계 모델 및 학습된 네트워크를 오픈 소스로 공개합니다. 이 과정을 구성하는 노트북은 배포 및 사용이 자유됩다는 것이 이러한 정신입니다. 우리는 모든 사람들이 딥 러닝에 대해 배울 수 있는 접근의 장벽을 낮추기 위해 열심히 노력했으며 독자가 이것의 혜택을 누릴 수 있기를 바랍니다.

## Summary

* Machine learning studies how computer systems can use data to improve performance. It combines ideas from statistics, data mining, artificial intelligence and optimization. Often it is used as a means of implementing artificially intelligent solutions.
* As a class of machine learning, representational learning focuses on how to automatically find the appropriate way to represent data. This is often accomplished by a progression of learned transformations.
* Much of the recent progress has been triggered by an abundance of data arising from cheap sensors and internet scale applications, and by significant progress in computation, mostly through GPUs.
* Whole system optimization is a key component in obtaining good performance. The availability of efficient deep learning frameworks has made design and implementation of this significantly easier.
* 머신러닝은 컴퓨터 시스템이 어떻게 데이터를 사용하여 성능을 향상시킬 수 있는지 연구합니다. 통계, 데이터 마이닝, 인공 지능 및 최적화의 아이디어를 결합합니다. 종종 인위적으로 지능형 솔루션을 구현하는 수단으로 사용됩니다. 
* 머신러닝의 클래스로서 표현 학습은 데이터를 나타내는 적절한 방법을 자동으로 찾는 방법에 초점을 맞 춥니 다. 이것은 종종 학습된 변환의 진행에 의해 성취됩니다. 
* 최근 진전의 대부분은 값싼 센서와 인터넷 규모 응용 프로그램에서 발생하는 풍부한 데이터와 주로 GPU를 통한 계산의 상당한 진전으로 인해 트리거되었습니다. 
* 전체 시스템 최적화가 핵심입니다. 구성 요소를 사용하여 좋은 성능을 얻을 수 있습니다. 효율적인 딥 러닝 프레임워크의 가용성으로 인해 이 프레임워크의 설계와 구현이 훨씬 쉬워졌습니다.

## Problems

1. Which parts of code that you are currently writing could be 'learned', i.e. improved by learning and automatically determining design choices that are made in your code? Does your code include heuristic design choices?
1. Which problems that you encounter have many examples for how to solve them, yet no specific way to automate them? These may be prime candidates for using Deep Learning.
1. Viewing the development of Artificial Intelligence as a new industrial revolution, what is the relationship between algorithms and data? Is it similar to steam engines and coal (what is the fundamental difference)?
1. Where else can you apply the end-to-end training approach? Physics? Engineering? Econometrics?
1. Why would you want to build a Deep Network that is structured like a human brain? What would the advantage be? Why would you not want to do that (what are some key differences between microprocessors and neurons)?
1. 현재 작성중인 코드의 어느 부분이 '학습' 될 수 있습니까? 즉, 코드에서 만들어진 디자인 선택을 학습하고 자동으로 결정함으로써 개선 될 수 있습니까? 코드에 휴리스틱 디자인 선택이 포함되어 있습니까? 
1. 어떤 문제가 발생하는지이를 해결하는 방법에 대한 많은 예가 있지만 자동화하는 구체적인 방법은 없습니다. 이들은 딥 러닝을 사용하기위한 주요 후보 일 수 있습니다. 
1. 새로운 산업 혁명으로 인공 지능의 발전을 보고, 알고리즘과 데이터의 관계는 무엇인가? 증기 엔진과 석탄과 비슷합니까 (근본적인 차이점은 무엇입니까)? 
1. End-to-end 학습 접근 방식을 어디에서 적용할 수 있습니까? 물리학? 엔지니어링? 경제학? 
1. 왜 인간의 뇌처럼 구조화된 딥 네트워크를 만들고 싶습니까? 장점은 무엇입니까? 왜 그렇게하고 싶지 않습니까 (마이크로 프로세서와 뉴런의 주요 차이점은 무엇입니까)?

## References

[1] Turing, A. M. (1950). Computing machinery and intelligence. Mind, 59(236), 433.

[2] Hebb, D. O. (1949). The organization of behavior; a neuropsychological theory. A Wiley Book in Clinical Psychology. 62-78.

[3] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.

[4] Bishop, C. M. (1995). Training with noise is equivalent to Tikhonov regularization. Neural computation, 7(1), 108-116.

[5] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

[6] Sukhbaatar, S., Weston, J., & Fergus, R. (2015). End-to-end memory networks. In Advances in neural information processing systems (pp. 2440-2448).

[7] Reed, S., & De Freitas, N. (2015). Neural programmer-interpreters. arXiv preprint arXiv:1511.06279.

[8] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

[9] Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint.

[10] Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196.

[11] Li, M. (2017). Scaling Distributed Machine Learning with System and Algorithm Co-design (Doctoral dissertation, PhD thesis, Intel).

[12] You, Y., Gitman, I., & Ginsburg, B. Large batch training of convolutional networks. ArXiv e-prints.

[13] Jia, X., Song, S., He, W., Wang, Y., Rong, H., Zhou, F., … & Chen, T. (2018). Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes. arXiv preprint arXiv:1807.11205.

[14] Xiong, W., Droppo, J., Huang, X., Seide, F., Seltzer, M., Stolcke, A., … & Zweig, G. (2017, March). The Microsoft 2016 conversational speech recognition system. In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on (pp. 5255-5259). IEEE.

[15] Lin, Y., Lv, F., Zhu, S., Yang, M., Cour, T., Yu, K., … & Huang, T. (2010). Imagenet classification: fast descriptor coding and large-scale svm training. Large scale visual recognition challenge.

[16] Hu, J., Shen, L., & Sun, G. (2017). Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 7.

[17] Campbell, M., Hoane Jr, A. J., & Hsu, F. H. (2002). Deep blue. Artificial intelligence, 134 (1-2), 57-83.

[18] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., … & Dieleman, S. (2016). Mastering the game of Go with deep neural networks and tree search. Nature, 529 (7587), 484.

[19] Brown, N., & Sandholm, T. (2017, August). Libratus: The superhuman ai for no-limit poker. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence.

[20] Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6), 679-698.

[21] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.

[22] Salton, G., & McGill, M. J. (1986). Introduction to modern information retrieval.

[23] Tesauro, G. (1995), Transactions of the ACM, (38) 3, 58-68

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2310)

![](../img/qr_deep-learning-intro.svg)
