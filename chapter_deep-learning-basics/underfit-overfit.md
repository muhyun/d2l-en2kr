# Model Selection, Underfitting and Overfitting

# 모델 선택, Underfitting, Overfitting

In machine learning, our goal is to discover general patterns. For example, we might want to learn an association between genetic markers and the development of dementia in adulthood. Our hope would be to uncover a pattern that could be applied successfully to assess risk for the entire population.

머신러닝에서 우리의 목표는 일반적인 패턴을 발견하는 것입니다. 예를 들면, 유전자 표지와 성인기의 치매 발병간의 관련성을 배우기를 원할 수 있습니다. 여기서 우리의 원하는 것은 전인류에 대한 위험을 평가하기 위해서 성공적으로 적용될 수 있는 패턴을 발견하는 것입니다.

However, when we train models, we don’t have access to the entire population (or current or potential humans). Instead, we can access only a small, finite sample. Even in a large hospital system, we might get hundreds of thousands of medical records. Given such a finite sample size, it’s possible to uncover spurious associations that don’t hold up for unseen data.

하지만, 우리가 모델을 학습시킬 때,  전체 인구에 대한 정보를 사용하는 것은 불가능합니다. 대신, 우리는 작은 유한한 샘플만을 사용할 수 있습니다. 대형 병원 시스템에서도 조차도 수십만건의 의료 기록 정도를 얻을 수도 있습니다. 이렇게 한정된 표본의 크기를 감안하면, 보이지 않은 데이터에 존재하는 않는 가짜 연관성을 발견할 수 있습니다.

Let’s consider an extreme pathological case. Imagine that you want to learn to predict which people will repay their loans. A lender hires you as a data scientist to investigate the case and gives you complete files on 100 applicants, of which 5 defaulted on their loans within 3 years. The files might include hundreds of features including income, occupation, credit score, length of employment etcetera. Imagine that they additionally give you video footage of their interview with a lending agent. That might seem like a lot of data!

극단적인 병리학적 사례를 생각해보겠습니다. 어떤 사람들이 대출금을 상환할 것인지를 예측하는 것을 배우기를 원한다고 상상해보십시오. 대출 기관이 캐이스를 조사하기 위해서 당신을 데이터 과학자로 고용해서, 100명의 지원자에 대한 완전한 파일을 제공합니다. 100명의 지원자 중에 5명이 3년안에 채무 불이행을 했었습니다. 제공된 파일에는 수입, 직업, 신용점수, 취업 기간 등을 포함한 100여개의 feature들이 있습니다. 더불어 대출기관과의 인터뷰 비디오가 추가로 제공한다고 상상해보겠습니다. 이것들이 아주 많은 데이터처럼 보일 수 있습니다!

Now suppose that after generating an enormous set of features, you discover that of the 5 applicants who defaults, all 5 were wearing blue shirts during their interviews, while only 40% of general population wore blue shirts. There’s a good chance that any model you train would pick up on this signal and use it as an important part of its learned pattern.

엄청난 양의 feature 세트를 만든 후, 채무 불이행을 한 5명이 모두 인터뷰 중에 파란 셔츠를 입였다는 것을 발견했다고 가정하겠습니다. 반면, 일반 인구의 40%만이 파란 셔츠를 입었습니다. 여러분이 학습시킨 모델이 이 신호를 포착해서, 학습한 패던의 중요한 부분으로 사용할 가능성이 큽니다.

Even if defaulters are no more likely to wear blue shirts, there’s a 1% chance that we’ll observe all five defaulters wearing blue shirts. And keeping the sample size low while we have hundreds or thousands of features, we may observe a large number of spurious correlations. Given trillions of training examples, these false associations might disappear. But we seldom have that luxury.

채무 불이행자가 더 이상 파란 셔츠를 입지 않을지라도, 모든 5명의 채무 불이행자가 파란 셔츠를 입을 것이라고 관찰할 확률이 1% 입니다. 수백 또는 수천개의 feature들을 가지고 있으면서 샘플의 크기를 적게 유지한다면, 아주 많은 가짜 상관 관계를 관찰할 것입니다. 수조개의 학습 샘플이 주어진다면, 이 잘못된 연관성은 사라질 것입니다. 하지만, 실제 그렇게 많은 데이터를 얻을 수 있는 경우가 드뭅니다.

The phenomena of fitting our training distribution more closely than the real distribution is called overfitting, and the techniques used to combat overfitting are called regularization.
More to the point, in the previous sections we observed this effect on the Fashion-MNIST dataset. If you altered the model structure or the hyper-parameters during the experiment, you may have found that for some choices the model might not have been as accurate when using the testing data set compared to when the training data set was used.

모델이 실제 분포보다 학습 샘플들 분포에 더 근접하게 학습되는 현상을 overfitting 이라고 하며, overfitting 을 피하는 방법을 정규화(regularization)라고 합니다. 더 정확하게는 이전 절의 Fashion-MNIST 데이서셋에서 이 현상이 나왔었습니다. 실험 중에 모델의 구조나 hyperparameter들을 바꾸면, 어떤 조합에서는 모델이 학습 데이터와 비교해서 테스팅 데이터셋을 사용했을 때 정확하지 않게 나오는 것을 현상을 찾아냈을 수 있습니다.

## Training Error and Generalization Error

## 학습 오류와 일반화 오류

Before we can explain this phenomenon, we need to differentiate between training and a generalization error.  In layman's terms, training error refers to the error exhibited by the model during its use of the training data set and generalization error refers to any expected error when applying the model to an imaginary stream of additional data drawn from the underlying data distribution. The latter is often estimated by applying the model to the test set. In the previously discussed loss functions, for example, the squared loss function used for linear regression or the cross-entropy loss function used for softmax regression, can be used to calculate training and generalization error rates.

이 현상에 대해서 설명하기 전에, 학습 오류와 일반화 오류에 대해서 구분할 필요가 있습니다. Layman의 용어에 의하면, 학습 오류는 학습 데이터셋을 사용했을 때 나오는 오류이고, 일반화 오류는 기본 데이터 분포에서 추가로 데이터를 뽑아낸 가상의 스트림에 모델을 적용할 때 예상되는 오류를 의미합니다. 종종 일반화 오류는 테스트 셋에 모델을 적용해서 추정합니다. 예를 들면, 이전에 논의한 loss 함수들 중에 선형회귀에 사용된 squared loss 함수나 softmax regression에 사용된 cross-entropy loss 함수를 이용해서 학습 오류와 일반화 오류 비율을 계산할 수 있습니다.

The following three thought experiments will help illustrate this situation better. Consider a college student trying to prepare for his final exam. A diligent student will strive to practice well and test his abilities using exams from previous years. Nonetheless, doing well on past exams is no guarantee that he will excel when it matters. For instance, the student might try to prepare by rote learning the answers to the exam questions. This requires the student to memorize many things. He might even remember the answers for past exams perfectly. Another student might prepare by trying to understand the reasons for giving certain answers. In most cases, the latter student will do much better.

다음 세가지 사고 실험이 이 상황을 더 잘 설명하는데 도움이 될 것입니다. 마지막 시험을 준비하는 대학생을 생각해봅시다. 근면한 학생은 준비를 잘 하고, 이전 년도의 시험 문제를 통해서 본인의 능력을 테스트를 하는 등의 노력할 것입니다. 하지만, 이전 시험을 잘 푸는 것이 꼭 그 학생이 실제 시험을 잘 본다는 것을 보장하지는 못합니다. 예를 들면, 그 학생은 시험 문제의 답을 기계적으로 학습하면서 준비하려고 노력 할지도 모릅니다. 이렇게 하면 그 학생은 많은 것을 외워야 합니다. 이렇게 해서 그 학생은 이전 시험의 답을 완벽하게 암기할 수도 있습니다. 다른 학생은 문제에 대한 특정 답이 나오게되는 이유를 이해하려고 노력하면서 준비했습니다. 대부분의 경우에는, 후자의 경우에 실제 시험에서 더 좋은 성적을 냅니다.

Likewise, we would expect that a model that simply performs table lookup to answer questions. If the inputs are discrete, this might very well work after seeing *many* examples. Nonetheless, such a model is unlikely to work well in practice, as data is often real-valued and more scarce than we would like. Furthermore, we only have a finite amount of RAM to store our model in.

마찬가지로, 질문에 대한 답을 테이블에서 조회하는 역할을 수행하는 모델을 생각하보겠습니다. 입력이 이산적(discrete)인 경우, 많은 샘플을 보는 것을 통해서 잘 동작할 수 있습니다. 하지만, 이 모델은 데이터가 실수값이거나 우리가 원하는 것보다 더 부족한 경우 실제 상황에서는 잘 동작하지 않을 가능성이 높습니다. 더군다나, 우리는 모델을 저장할 수 있는 한정된 양의 메모리만 가지고 있습니다.

Lastly, consider a trivial classification problem. Our training data consists of labels only, namely 0 for heads and 1 for tails, obtained by tossing a fair coin. No matter what we do, the generalization error will always be $\frac{1}{2}$. Yet the training error may be quite a bit less than that, depending on the luck of the draw. E.g. for the dataset {0, 1, 1, 1, 0, 1} we will 'predict' class 1 and incur an error of $\frac{1}{3}$, which is considerably better than what it should be. We can also see that as we increase the amount of data, the probability for large deviations from $\frac{1}{2}$ will diminish and the training error will be close to it, too. This is because our model 'overfit' to the data and since things will 'average out' as we increase the amount of data.

마지막으로, 간단한 분류 문제를 고려해보겠습니다. 공정한 동전을 던져서 앞면이 나오면 0, 뒷면이 나오면 1로 분류한 label을 갖는 학습 데이터가 있습니다. 우리가 무엇을 하던지 상관없이, 일반화 오류는 항상  $\frac{1}{2}$ 입니다. 하지만, 학습 오류는 동전을 던지는 운에 따라서 더 작아질 수 있습니다. 예를 들어 {0, 1, 1, 1, 0, 1}인 학습 데이터를 사용하는 경우에는, 1을 예측한다면,  $\frac{1}{3}$  의 오류가 발생하는데, 이는 실제 보다 더 좋은 값입니다. 데이터 양을 늘릴 수록,  확률의 편차가 $\frac{1}{2}$ 로 부터 감소하고, 학습 오류도 이에 근접하게될 것입니다. 그 이유는 우리의 모델이 데이터에 overfit 되어 있고, 데이터의 양을 늘리면 모든것이 평균상태가 되기 때문입니다.

### Statistical Learning Theory

## 통계적 학습 이론(Statistical Learning Theory)

There is a formal theory of this phenomenon. Glivenko and Cantelli derived in their [eponymous theorem](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem) the rate at which the training error converges to the generalization error. In a series of seminal papers [Vapnik and Chervonenkis](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) extended this to much more general function classes. This laid the foundations of [Statistical Learning Theory](https://en.wikipedia.org/wiki/Statistical_learning_theory).

이 현상에 대한 공식 이론이 있습니다. Glivenko와 Cantelli는 그들의  [eponymous theorem](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem) 에서 학습 오류가 일반화 오류로 수렴하는 비율을 도출했습니다. [Vapnik와 Chervonenkis](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) 는 여러 논문을 통해서 이것을 더 일반적인 함수의 클래스들로 확장했고, [Statistical Learning Theory](https://en.wikipedia.org/wiki/Statistical_learning_theory) 의 기초가 되었습니다.

Unless stated otherwise, we assume that both the training set and the test set are drawn independently and identically drawn from the same distribution. This means that in drawing from the distribution there is no memory between draws. Moreover, it means that we use the same distribution in both cases. Obvious cases where this might be violated is if we want to build a face recognition by training it on elementary students and then want to deploy it in the general population. This is unlikely to work since, well, the students tend to look quite from the general population. By training we try to find a function that does particularly well on the training data. If the function is very fleible such as to be able to adapt well to any details in the training data, it might do a bit too well. This is precisely what we want to avoid (or at least control). Instead we want to find a model that reduces the generalization error. A lot of tuning in deep learning is devoted to making sure that this does not happen.

특별히 이야기하지 않으면, 학습 데이터셋과 테스트 데이터셋은 동일한 분포로 부터 독립적으로 추출되었다고 가정합니다. 즉, 이 분포로부터 추출을 할 때 추출들간의 어떤 기억도 없다는 것을 의미합니다. 더 나아가, 두 경우 동일한 분포를 사용하는 것을 의미합니다. 이를 위반하는 명확한 사례로 초등학교 학생들의 얼굴 데이터로 학습한 얼굴 인식 모델을 이용해서 일반 인구에 적용하는 것을 들 수 있습니다. 초등학교 학생들과 일반 사람들은 아주 다르게 보일 것이기 때문에 잘 동작하지 않을 가능성이 높습니다. 학습을 통해서 우리는 학습 데이터에 잘 맞는 함수를 찾으려고 노력합니다. 만약 학습 데이터의 자세한 것에 아주 잘 적응할 정도로 유연하다면, 지나치게 잘하게 될 것입니다. 이것이 바로 우리가 피하려고 또는 통제하려는 것입니다. 대신, 우리는 일반화 오류를 줄이는 모델을 찾는 것을 원합니다. Deep learning의 많은 튜닝은 이런 것이 일어나지 않도록 하는데 이용됩니다.

### Model Complexity

## 모델 복잡도

When we have simple models and abundant data, we expect the generalization error to resemble the training error. When we work with more complex models and fewer examples, we expect the training error to go down but the generalization gap to grow. What precisely constitutes model complexity is a complex matter. Many factors govern whether a model will generalize well. For example a model with more parameters might be considered more complex. A model whose parameters can take a wider range of values might be more complex. Often with neural networks, we think of a model that takes more training steps as more complex, and one subject to early stopping as less complex.

우리는 간단한 모델들과 많은 데이터가 있을 경우, 일반화 오류가 학습 오류와 비슷해 지기를 예상합니다. 반면에 모델이 복잡하고 데이터가 적을 때는, 학습 오류는 작아지지만, 일반화 오류는 커질 것을 예상합니다. 무엇이 정확하게 모델의 복잡성을 구성하는지는 복잡한 문제입니다. 모델이 일반화를 잘 할 수 있을지는 많은 것들에 의해서 영향을 받습니다. 예를 들면, 더 많은 파라메터를 갖는 모델이 더 복잡하다고 여기질 수 있고, 값의 범위가 더 넓은 파라메터를 갖는 모델이 더 복잡하다고 여겨질 수도 있습니다. 뉴럴 네트워크의 경우에는 학습을 더 오래한 모델이 더 복잡한 것이라고 생각될 수도 있고, 일찍 학습을 종료한 모델은 덜 복잡하다고 생각될 수도 있습니다.

It can be difficult to compare the complexity among members of very different model classes (say decision trees versus neural networks). For now a simple rule of thumb is quite useful: A model that can readily explain arbitrary facts is what statisticians view as complex, whereas one that has only a limited expressive power but still manages to explain the data well is probably closer to the truth. In philosophy this is closely related to Popper’s criterion of [falsifiability](https://en.wikipedia.org/wiki/Falsifiability) of a scientific theory: a theory is good if it fits data and if there are specific tests which can be used to disprove it. This is important since all statistical estimation is [post hoc](https://en.wikipedia.org/wiki/Post_hoc), i.e. we estimate after we observe the facts, hence vulnerable to the associated fallacy. Ok, enough of philosophy, let’s get to more tangible issues.

다양한 모델 종류들 간의 복잡성을 비교하는 것은 어려운 일 수 있습니다. 예를 들면 decision tree와 neural network의 복잡성을 비교하는 것은 어렵습니다. 이런 경우, 간단한 경험의 법칙을 적용하는 것이 유용합니다. 통계학자들은 임의의 사실을 잘 설명하는 모델을 복잡하다고 하고, 제한적인 설명을 하는 능력을 갖으나 데이터를 여전히 잘 설명하는 모델은 진실에 좀 더 가깝다고 합니다.  철학에서 이것은 포퍼의 과학 이론의 허위 진술성( [falsifiability](https://en.wikipedia.org/wiki/Falsifiability)과 밀접한 관련이 있습니다. 어떤 이론이 데이터에 적합하고, 오류를 입증할 수 있는 특정 테스트가 있다면, 그 이론을 좋다고 합니다. 모든 통계적 추정이 [post-hoc](https://en.wikipedia.org/wiki/Post_hoc)이기에 이는 매우 중요합니다. 즉, 우리는 어떤 사실을 관찰한 후에 추정을 합니다. 따라서, 관련 오류에 취약하게 됩니다. 자, 철학에 대해서는 충분히 이야기했으니, 더 구체적인 이슈를 살펴보겠습니다.

To give you some intuition in this chapter, we’ll focus on a few factors that tend to influence the generalizability of a model class:

여러분이 이 장에 대한 직관을 갖을 수 있도록, 모델 클래스의 일반화에 영향을 줄 수 있는 몇가지 요소들에 집중하겠습니다.

1. The number of tunable parameters. When the number of tunable parameters, sometimes denoted as the number of degrees of freedom, is large, models tend to be more susceptible to overfitting.
1. The values taken by the parameters. When weights can take a wider range of values, models can be more susceptible to over fitting.
1. The number of training examples. It’s trivially easy to overfit a dataset containing only one or two examples even if your model is simple. But overfitting a dataset with millions of examples requires an extremely flexible model.
1. 튜닝이 가능한 파라메터의 개수. 자유도라고 하기도 하는 튜닝 가능한 파라메터의 수가 클 경우, 모델이 overfitting 에 더 취약한 경향이 있습니다.
1. 파라메터에 할당된 값. weight들이 넒은 범위의 값을 갖을 경우, 모델은 overfitting에 더 취약할 수 있습니다.
1. 학습 예제의 개수. 모델이 간단할 지라도 학습 데이터가 한 개 또는 두 개인 경우에는 overfit 되기가 아주 쉽습니다. 하지만, 수백만개의 학습 데이터를 이용해서 모델을 overfitting 시키기 위해서는 모델이 아주 복잡해야 합니다.

## Model Selection

## 모델 선택

In machine learning we usually select our model based on an evaluation of the performance of several candidate models.  This process is called model selection. The candidate models can be similar models using different hyper-parameters.  Using the multilayer perceptron as an example, we can select the number of hidden layers as well as the number of hidden units, and activation functions in each hidden layer.  A significant effort in model selection is usually required in order to end up with an effective model.  In the following section we will be describing the validation data set often used in model selection.

머신러닝에서, 우리는 보통 여러 후보 모델들의 성능을 평가해서 모델을 선정합니다. 이 과정을 모델 선택 (model selection)이라고 합니다. 후보 모델들은 다른 hyper-parameter들을 적용한 간단한 모델들일 수 있습니다. Multilayer perceptron을 예로 들면, 우리는 hidden 레이어의 개수, hidden unit의 개수, 각 hidden 레이어의 activation 함수를 선택할 수 있습니다. 효과적인 모델을 찾기 위해서는 모델 선택에 상당한 노력이 필요합니다. 다음 절에서 모델 선택에 종종 사용되는 검증 데이터셋 (validation data set)에 대해서 설명하겠습니다.

### Validation Data Set

### 검증 데이터셋 

Strictly speaking, the test set can only be used after all the hyper-parameters and model parameters have been selected. In particular, the test data must not be used in model selection process, such as in the tuning of hyper-parameters.  We should not rely solely on the training data during model selection, since the generalization error rate cannot be estimated from the training error rate.  Bearing this in mind, we can reserve a portion of data outside of the training and testing data sets to be used in model selection.  This reserved data is known as the validation data set, or validation set.  For example, a small, randomly selected portion from a given training set can be used as a validation set, with the remainder used as the true training set.

엄밀하게 이야기하면, 테스트 셋은 모든 hyper-parameter들과 모델 파라메터들이 선택된 후에만 사용되어야 합니다. 특히, 테스트 데이터셋은 hyper-parameter 선택과 같은 모델 선택 과정에서 사용되서는 안됩니다. 모델 선택 과정에서 학습 데이터에만 의존해서도 안됩니다. 그 이유는 일반화 오류율이 학습 오류율로 예상될 수 없기 때문입니다. 이를 고려해서, 학습 데이터와 테스트 데이터 이외의 데이터를 확보해서 모델 선택에 사용할 수 있습니다. 이렇게 확보한 데이터는 검증 데이터 셋(validation data set) 또는 검증셋(validation set)이라고 합니다. 예를 들면, 학습 데이터에서 임의로 선택한 일부의 데이터를 검증 셋으로 사용하고, 나머지를 실제 학습 데이터로 사용할 수 있습니다.

However, in practical applications the test data is rarely discarded after one use since it’s not easily obtainable.  Therefore, in practice, there may be unclear boundaries between validation and testing data sets.  Unless explicitly stated, the test data sets used in the experiments provided in this book should be considered as validation sets, and the test accuracy in the experiment report are for validation accuracy. The good news is that we don't need too much data in the validation set. The uncertainty in our estimates can be shown to be of the order of $O(n^{-\frac{1}{2}})$.

하지만, 실제 응용의 경우에는 테스트 데이터를 구하기 어렵기 때문에 한번 사용하고 버리는 경우가 드뭅니다. 따라서, 실제의 경우에는 검증 데이터와 테스트 데이터 셋의 구분이 명확하지 않을 수도 있습니다. 명시적으로 별도로 언급하지 않는 경우 이 책에서 실험으로 사용하는 테스트 데이터셋은 검증 데이터 셋이라고 하고, 실험 결과의 테스트 정확도는 검증 정확도를 의미하겠습니다. 좋은 소식은 검증 셋에 아주 많은 데이터가 필요하지 않다는 것입니다. 우리의 예측의 불명확성은  $O(n^{-\frac{1}{2}})​$ 오더로 보여질 수 있습니다.


### $K$-Fold Cross-Validation

When there is not enough training data, it is considered excessive to reserve a large amount of validation data, since the validation data set does not play a part in model training.  A solution to this is the $K$-fold cross-validation method. In  $K$-fold cross-validation, the original training data set is split into $K$ non-coincident sub-data sets. Next, the model training and validation process is repeated $K$ times.  Every time the validation process is repeated, we validate the model using a sub-data set and use the $K-1$ sub-data set to train the model.  The sub-data set used to validate the model is continuously changed throughout this $K$ training and validation process.  Finally, the average over $K$ training and validation error rates are calculated respectively.

학습 데이터가 충분하지 않을 경우 검증 데이터를 많이 확보하는 것은 과하다고 간주됩니다. 왜냐하면, 검증 데이터는 모델 학습에 어떤 역할도 할 수 없기 때문입니다. 이에 대한 해결책으로 $K$-fold cross-validation 방법이 있습니다.  $K$-fold cross-validation 에서는 원래 학습 데이터를 겹치지 않는 K개의 부분 데이터셋으로 나누고, 모델 학습과 검증을  $K$ 번 반복합니다. 검증이 수행될 때마다  $K-1$ 개의 부분 데이터셋으로 학습을 하고, 1개의 부분 데이터셋으로 검증을 수행합니다. 모델을 검증하는데 사용하는 부분 데이터셋은 계속 바꾸면서 $K$ 번 학습과 검증을 수행하게 됩니다. 마지막으로, $K$ 번의 학습과 검증 오류에 대한 평균을 각각 구합니다.


## Underfitting과 Overfitting

Next, we will look into two common problems that occur during model training.  One type of problem occurs when the model is unable to reduce training errors since the model is too simplistic. This phenomenon is known as underfitting.  As discussed, another type of problem is when the number of model training errors is significantly less than that of the testing data set, also known as overfitting.  In practice, both underfitting and overfitting should be dealt with simultaneously whenever possible.  Although many factors could cause the above two fitting problems, for the time being we’ll be focusing primarily on two factors: model complexity and training data set size.

다음으로는 모델 학습을 진행하면서 만나게 되는 일반적인 두가지 문제에 대해서 살펴보겠습니다. 첫번째 문제는 모델이 너무 간단하기 때문에 학습 오류가 줄어들지 않는 것입니다. 이 현상을 underfitting 이라고 합니다. 두번째 문제는 앞에서 이야기했던 overfitting 으로, 이는 학습 오류가 테스트 데이터셋에 대한 오류보다 아주 작은 경우입니다. 실제로 이 두 문제는 가능한 경우 항상 동시에 해결이 되어야 합니다. 이 두 문제의 원인은 여러 요소들이 있지만, 여기서는 두가지 요소에 대해서 집중하겠습니다. 이 두가지는 모델 보잡성과 학습 데이터 셋 크기입니다.

### Model Complexity
### 모델 복잡도

We use polynomials as a way to illustrate the issue. Given training data consisting of the scalar data feature $x$ and the corresponding scalar label $y$, we try to find a polynomial of degree $d$

이 이슈를 설명하기 위해서 다항식을 예로 들겠습니다. 스칼라 데이터 feature $x$ 와 이에 대한 스칼라 label $y$  로 구성된 학습 데이터가 주어진 경우, $y$ 를 추정하는 $d$ 차원 다항식을 찾는다고 하겠습니다.

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

to estimate $y$. Here $w_i$ refers to the model’s weight parameter. The bias is implicit in $w_0$ since $x^0 = 1$. Similar to linear regression we also use a squared loss for simplicity (note that $d = 1$ we recover linear regression).

여기서  $w_i$ 는 모델의 weight 파라메터를 의미하고, bias는 $x^0 = 1$ 이기 때문에  $w_0$ 이 됩니다. 간단하게 하기 위해서, 선형 회귀(linear regression)의 경우와 같이 squared loss를 사용하겠습니다. (사실 $d=1$ 인 경우 이 모델은 선형 회귀(linear regression) 입니다.)

A higher order polynomial function is more complex than a lower order polynomial function, since the higher-order polynomial has more parameters and the model function’s selection range is wider.  Therefore, using the same training data set, higher order polynomial functions should be able to achieve a lower training error rate (relative to lower degree polynomials).  Bearing in mind the given training data set, the typical relationship between model complexity and error is shown in the diagram below. If the model is too simple for the dataset, we are likely to see underfitting, whereas if we pick an overly complex model we see overfitting. Choosing an appropriately complex model for the data set is one way to avoid underfitting and overfitting.

고차원의 다항 함수는 저차원의 다항 함수보다 더 복잡합니다. 이유는 차원이 더 높아지면, 더 많은 파라메터를 갖게 되고, 모델 함수의 선택 범위가 더 넓어지기 때문입니다. 따라서, 같은 학습 데이터셋을 사용하는 경우, 더 높은 차원의 다항 함수에 대한 학습 오류는 그 보다 낮은 차원의 다항 함수의 오류보다 낮을 것입니다. 이를 염두하면, 학습 데이터셋이 고정되어 있을 때 모델의 복잡도와 오류의 일반적인 상관관계는 아래 그림으로 설명됩니다. 데이터에 비해서 모델이 너무 간단하면, underfitting 이 발생하고, 모델을 너무 복잡하게 선택하면 overfitting 이 발생합니다. 데이터에 대한 모델을 적절한 복잡성을 선택하는 것이 overfitting 과 underfitting 문제를 피하는 방법 중에 하나입니다.


![Influence of Model Complexity on Underfitting and Overfitting](../img/capacity_vs_error.svg)


### Data Set Size
### 데이터 셋 크기

Another influence is the amount of training data. Typically, if there are not enough samples in the training data set, especially if the number of samples is less than the number of model parameters (count by element), overfitting is more likely to occur. Additionally, as we increase the amount of training data, the generalization error typically decreases. This means that more data never hurts. Moreover, it also means that we should typically only use complex models (e.g. many layers) if we have sufficient data.

다른 원인은 학습 데이터의 양입니다. 일반적으로 학습 데이터셋의 샘플 개수가 충분하지 않은 경우, 특히 모델의 파라메터 개수보다 적은 수의 샘플을 사용하는 경우, overfitting 이 쉽게 발생합니다. 학습 데이터의 양을 늘리면, 일반화 오류는 일반적으로 줄어듭니다. 즉, 더 많은 데이터는 모델 학습에 나쁜 영향을 미치지 않다는 것을 의미합니다. 더 나아가서, 이는 충분한 데이터가 있다면, 일반적으로 많은 레이어들을 갖는 복잡한 모델을 사용해야한다는 것을 의미합니다.


## 다항식 회귀(Polynomial Regression)

Let us try how this works in practice by fitting polynomials to data. As before we start by importing some modules.

데이터를 이용해서 다항식을 학습시켜 보면서 이것이 어떻게 동작하는지 보겠습니다. 우선 몇가지 모듈을 import 합니다.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
```

### Generating Data Sets
### 데이터 셋 생성하기

First we need data. Given $x$ we will use the following cubic polynomial to generate the labels on training and test data:

우선 데이터가 필요합니다. 주어진 $x$ 에 대해서, 다음 3차원 방적식을 사용해서 학습 데이터와 테스트 데이터로 사용할 label을 만들겠습니다.

$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0,0.1)$$

The noise term $\epsilon$ obeys a normal distribution with a mean of 0 and a standard deviation of 0.1.  The number of samples for both the training and the testing data sets  is set to 100.

노이즈 항인  $\epsilon$ 은 평균이 0이고 표준 편차가 0.1인 정규 분포를 따릅니다. 학습과 테스트 데이터셋의 샘플의 개수는 각각 100로 하겠습니다.

```{.python .input  n=2}
maxdegree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 1000  # Training and test data set sizes
true_w = nd.zeros(maxdegree)  # Allocate lots of empty space
true_w[0:4] = nd.array([5, 1.2, -3.4, 5.6])

features = nd.random.normal(shape=(n_train + n_test, 1))
features = nd.random.shuffle(features)
poly_features = nd.power(features, nd.arange(maxdegree).reshape((1, -1)))
poly_features = poly_features / (
    nd.gamma(nd.arange(maxdegree) + 1).reshape((1, -1)))
labels = nd.dot(poly_features, true_w)
labels += nd.random.normal(scale=0.1, shape=labels.shape)
```

For optimization we typically want to avoid very large values of gradients, losses, etc.; This is why the monomials stored in `poly_features` are rescaled from $x^i$ to $\frac{1}{i!} x^i$. It allows us to avoid very large values for large exponents $i$. Factorials are implemented in Gluon using the Gamma function, where $n! = \Gamma(n+1)$.

최적화를 위해서, gradient, loss 등이 큰 값을 갖는 것을 피해야합니다. `poly_features` 에 저장되는 단항들이 $x^i$ 에서 $\frac{1}{i!} x^i$ 로 스캐일을 조정하는 이유입니다. 이렇게 하면 큰 차원 $i$ 의 값들이 아주 커지는 것을 방지할 수 있습니다. 팩토리얼은 Gluon의 Gamma 함수를 이용해서 구현합니다. ( $n! = \Gamma(n+1)$)

Take a look at the first 2 samples from the generated data set. The value 1 is technically a feature, namely the constant feature corresponding to the bias.

생성된 데이터 셋에서 처음 두 샘플을 확인해봅니다. **The value 1 is technically a feature, namely the constant feature corresponding to the bias.**

```{.python .input  n=3}
features[:2], poly_features[:2], labels[:2]
```

### Defining, Training and Testing Model
### 모델 정의, 학습, 그리고 테스트

We first define the plotting function`semilogy`, where the $y$ axis makes use of the logarithmic scale.

우선 그래프를 그리는 함수 `semilogy`  를 정의합니다. $y$ 축은 logarithm 단위를 사용합니다.

```{.python .input  n=4}
# This function has been saved in the d2l package for future use
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
```

Similar to linear regression, polynomial function fitting also makes use of a squared loss function.  Since we will be attempting to fit the generated data set using models of varying complexity, we insert the model definition into the `fit_and_plot` function. The training and testing steps involved in polynomial function fitting are similar to those previously described in softmax regression.

선형 회귀(Linear regression)와 비슷하게, 다항 함수 학습에 squared loss 함수를 이용하겠습니다. 생성된 데이터를 이용해서 여러 복잡도를 갖는 모델들을 학습시킬 것이기 때문에, 모델 정의를 `fit_and_plot` 함수에 전달하도록 하겠습니다. 다항 함수에 대한 학습과 테스트 단계는 softmax regression과 비슷합니다.

```{.python .input  n=5}
num_epochs, loss = 200, gloss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net.add(nn.Dense(1, use_bias=False))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy())
```

### Third-order Polynomial Function Fitting (Normal)
### 3차 다항 함수 피팅(Third-order Polynomial Function Fitting (Normal))

We will begin by first using a third-order polynomial function with the same order as the data generation function. The results show that this model’s training error rate when using the testing data set is low. The trained model parameters are also close to the true values $w = [5, 1.2, -3.4, 5.6]$.

우선, 데이터를 생성한 것과 같은 3차원 다항 함수를 이용해보겠습니다. 테스트 데이터를 이용해서 얻은 모델의 오류는 낮게 나오는 것이 보여집니다.  학습된 모델 파라메터 역시 실제 값  $w = [5, 1.2, -3.4, 5.6]$ 과 비슷합니다.

```{.python .input  n=6}
num_epochs = 1000
# Pick the first four dimensions, i.e. 1, x, x^2, x^3 from the polynomial
# features
fit_and_plot(poly_features[:n_train, 0:4], poly_features[n_train:, 0:4],
             labels[:n_train], labels[n_train:])
```

### Linear Function Fitting (Underfitting)
### 선형 함수 피팅 (Underfitting)

Let’s take another look at linear function fitting.  Naturally, after the decline in the early epoch, it’s difficult to further decrease this model’s training error rate.  After the last epoch iteration has been completed, the training error rate is still high.  When used in data sets generated by non-linear models (like the third-order polynomial function) linear models are susceptible to underfitting.

선형 함수의 경우를 보겠습니다. 초기 epoch를 수행하면서 학습 오류가 감소한 후로 더 이상 모델 학습 오류가 감소하지 않는 것은 자연스러운 현상입니다. 마지막 epoch까지 마친 후에도 학습 오류는 여전히 높습니다. 선형 모델은 비선형 모델 (3차 다항 함수)로 만들어진 데이터 셋에 대해서 underfitting 에 민감합니다.

```{.python .input  n=7}
num_epochs = 1000
# Pick the first four dimensions, i.e. 1, x from the polynomial features
fit_and_plot(poly_features[:n_train, 0:3], poly_features[n_train:, 0:3],
             labels[:n_train], labels[n_train:])
```

### Insufficient Training (Overfitting)
### 부족한 학습 (Overfitting)

In practice, if the model hasn’t been trained sufficiently, it is still easy to overfit even if a third-order polynomial function with the same order as the data generation model is used.  Let's train the model using a polynomial of too high degree. There is insufficient data to pin down the fact that all higher degree coefficients are close to zero. This will result in a model that’s too complex to be easily influenced by noise in the training data.  Even if the training error rate is low, the testing error data rate will still be high.

실제 상황에서, 데이터를 생성할 때 사용한 것과 같은 3차 다항 함수를 이용할 경우에도 학습을 충분히 오래하지 않은 경우에는 overfit이 쉽게 발생할 수 있습니다. 아주 높은 차원의 다항식을 사용해서 모델을 학습시켜보겠습니다. 모든 높은 차수의 계수들이 0에 가깝다는 사실을 학습하기에는 데이터가 너무 적습니다. 이 경우에는 모델이 너무 복잡해서 학습 데이터의 노이즈에 쉽게 영향을 받는 결과가 나옵니다. 학습 오류가 낮을지라도, 테스트 오류는 여전히 높습니다.

Try out different model complexities (`n_degree`) and training set sizes (`n_subset`) to gain some intuition of what is happening.

다른 모델의 복잡도 (`n_degreee`)와 학습 셋 크기(`n_subset`)를 적용해서 어떤 일이 발생하는지에 대한 직감을 얻어보세요.

```{.python .input  n=8}
num_epochs = 1000
n_subset = 100  # Subset of data to train on
n_degree = 20   # Degree of polynomials
fit_and_plot(poly_features[1:n_subset, 0:n_degree],
             poly_features[n_train:, 0:n_degree], labels[1:n_subset],
             labels[n_train:])
```

Further along in later chapters, we will continue discussing overfitting problems and methods for dealing with them, such as weight decay and dropout.

다음 장들에서 overfitting 문제들을 계속 논의하고, 이를 해결하기 위한 weight decay와 dropout 과 같은 방법을 알아보겠습니다.


## Summary
## 요약

* Since the generalization error rate cannot be estimated based on the training error rate, simply minimizing the training error rate will not necessarily mean a reduction in the generalization error rate. Machine learning models need to be careful to safeguard against overfitting such as to minimize the generalization error.
* A validation set can be used for model selection (provided that it isn't used too liberally).
* Underfitting means that the model is not able to reduce the training error rate while overfitting is a result of the model training error rate being much lower than the testing data set rate.
* We should choose an appropriately complex model and avoid using insufficient training samples.
* 일반화 오류율은 학습 오류율을 이용해서 추정될 수 없기 때문에, 단순히 학습 오류율을 줄이는 것이 일반화 오류를 줄이는 것을 의미하지 않습니다. 머신 러닝 모델은 일반화 오류를 줄이기를 통해서 overfitting에 조심스럽게 대비 해야합니다.
* 검증 셋은 모델 선택에 사용됩니다. (너무 남용되지 않는다는 가정에서)
* Underfitting 은 모델이 학습 오류를 줄이지 못하는 상황을 의미하고, overfitting은 모델 학습 오류가 테스트 데이터의 오류보다 훨씬 작은 경우를 의미합니다.
* 우리는 적절한 모델의 복잡성을 선택해야하고, 부족한 학습 샘플을 이용하는 것을 피해야합니다.


## Problems
## 문제

1. Can you solve the polynomial regression problem exactly? Hint - use linear algebra.
1. Model selection for polynomials
    * Plot the training error vs. model complexity (degree of the polynomial). What do you observe?
    * Plot the test error in this case.
    * Generate the same graph as a function of the amount of data?
1. What happens if you drop the normalization of the polynomial features $x^i$ by $1/i!$. Can you fix this in some other way?
1. What degree of polynomial do you need to reduce the training error to 0?
1. Can you ever expect to see 0 generalization error?
1. 다항 regression 문제를 정확하게 풀 수 있나요? 힌트 - 선형대수를 이용합니다.
1. 다항식에 대한 모델 선택에 대해서
    - 학습 오류와 모델 복잡도(다항식의 차원 수)를 도식화해보세요. 무엇이 관찰되나요?
    - 이 경우 테스트 오류를 도식화해보세요.
    - 같은 그래프를 데이터 양에 따라서 그려보세요.
1. 다항식의 feature  $x^i$ 에 적용한 정규화 $1/i!$ 를 제거하면 어떤 일이 일어날까요? 다른 방법으로 이를 해결할 수 있나요?
1. 학습 오류를 0으로 줄이기 위해서 몇 차원을 사용하나요?
1. 일반화 오류를 0으로 줄이는 것이 가능한가요?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2341)

![](../img/qr_underfit-overfit.svg)
