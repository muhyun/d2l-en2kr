# Image Classification Data (Fashion-MNIST)

Before introducing the implementation for softmax regression, we need a suitable dataset. To make things more visually compelling we pick one on classification.
It will be used multiple times in later chapters to allow us to observe the difference between model accuracy and computational efficiency between comparison algorithms. The most commonly used image classification data set is the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit recognition data set. It was proposed by LeCun, Cortes and Burges in the 1990s. However, most models have a classification accuracy of over 95% on MNIST, hence it is hard to spot the difference between different models. In order to get a better intuition about the difference between algorithms we use a more complex data set. [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) was proposed by [Xiao, Rasul and Vollgraf](https://arxiv.org/abs/1708.07747) in 2017.

softmax regression 구현에 앞서 적절한 데이터셋이 필요합니다. 시각적으로 돋보이는 것을 만들기 위해서, 분류 문제에서 선택해보겠습니다. 

다음 장들에서 모델 정확도의 차이를 관찰하거나, 비교 알고리즘의 연산 효율성에 대한 이야기를 하는데에도 반복해서 사용할 예제입니다. 가장 흔한 이미지 분류 데이터셋은 MNIST 손글씨 숫자 인식 데이터셋이 있습니다. 이 데이터셋은 1990년 대에 Lecun, Cortes와 Burges에 의해서 제안되었습니다. 하지만, 거의 모든 모델이 MNIST 데이터셋에 대해서 95% 이상의 정확도를 보여주기 때문에, 모델들 사이의 차이를 설명하는데는 적합하지 않습니다. 알고리즘들의 차이를 보다 직관적으로 보여주기 위해서, 더 복잡한 데이터셋을 사용하겠습니다. 이 데이터셋은 Fashion-MNIST라를 것으로 2017년에 Xio, Rasul 그리고 Vollgraf가 제안했습니다. 

## Getting the Data

First, import the packages or modules required in this section.

우선, 이 절에서 필요한 패키지와 모듈을  import 합니다.

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet.gluon import data as gdata
import sys
import time
```

Next, we will download this data set through Gluon's `data` package. The data is automatically retrieved from the Internet the first time it is called. We specify the acquisition of a training data set, or a testing data set by the parameter `train`. The test data set, also called the testing set, is only used to evaluate the performance of the model and is not used to train the model.

다음으로, Gluon의 `data` 패키지를 이용해서 이 데이터셋을 다운로드합니다. 데이터셋은 처음 불려였을 때, 인터넷으로부터 자동으로 다운로드됩니다. `train` 파라메터를 통해서 학습 데이터셋을 받을 것인지 테스트 데이터셋을 받을 것인지를 정할 수 있습니다. 테스트 데이터셋 또는 테스팅 데이터셋은 모델의 성능을 평가할 때만 쓰이고, 학습에는 사용되지 않는 데이터입니다.

```{.python .input  n=23}
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
```

The number of images for each category in the training set and the testing set is 6,000 and 1,000, respectively. Since there are 10 categories, the number of examples for the training set and the testing set is 60,000 and 10,000, respectively.

학습 데이터셋과 테스트 데이터셋은 각 카테고리별로 각각 6,000개와 1,000개의 이미지들로 구성되어 있습니다. 카테고리 개수는 10개이기에, 학습 데이터는 총 60,000개의 이미지들로 테스팅 셋은 10,000개 이미지들을 가지고 있습니다.

```{.python .input}
len(mnist_train), len(mnist_test)
```

We can access any example by square brackets `[]`, and next, we will get the image and label of the first example.

`[]` 을 이용하면 각 샘플을 접근할 수 있습니다. 첫번째 데이터의 이미지와 label를 얻어보겠습니다.

```{.python .input  n=24}
feature, label = mnist_train[0]
```

The variable `feature` corresponds to an image with a height and width of 28 pixels. Each pixel is an 8-bit unsigned integer (uint8) with values between 0 and 255. It is stored in a 3D NDArray. Its last dimension is the number of channels. Since the data set is a grayscale image, the number of channels is 1. For the sake of simplicity, we will record the shape of the image with the height and width of $h$ and $w$ pixels, respectively, as $h \times w$ or `(h, w)`.

`feature` 변수는 높이와 넒이가 모두 28 픽셀인 이미지 데이터를 가지고 있습니다. 각 픽셀은 8-bit unsigned integer (uint8)이고, 0부터 255 사이의 값을 갖습니다. 이는 3차원 NDArray에 저장됩니다. 마지막 차원은 채널의 개수를 의미합니다. 데이터 셋이 회색 이미지이기 때문에, 채널의 수는 1이 됩니다. 간단하게 하기 위해서, 이미지의 모양이 높이 `h`, 넓이는 `w` 픽셀인 경우 이미지의 shape을  $h \times w$ 또는  `(h, w)` 로 표기하도록 하겠습니다.

```{.python .input}
feature.shape, feature.dtype
```

The label of each image is represented as a scalar in NumPy. Its type is a 32-bit integer.

각 이미지에 대한 label은 NumPy의 scalar로 저장되어있고, 이는 32-bit integer 형태입니다.

```{.python .input}
label, type(label), label.dtype
```

There are 10 categories in Fashion-MNIST: t-shirt, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. The following function can convert a numeric label into a corresponding text label.

Fashion-MNIST에는 10개의 카테고리가 있는데, 이들은 티셔츠, 바지, 풀오버, 드래스, 코드, 센달, 셔츠, 스니커, 가방, 발목 부츠입니다. 숫자 형태의 label을 텍스트 label로 바꿔주는 함수를 아래와 같이 정의합니다.

```{.python .input  n=25}
# This function has been saved in the d2l package for future use
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

The following defines a function that can draw multiple images and corresponding labels in a single line.

아래 함수는 한 줄에 여러 이미지와 그 이미지의 label을 그리는 것을 정의합니다.

```{.python .input}
# This function has been saved in the d2l package for future use
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # Here _ means that we ignore (not use) variables
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
```

Next, let's take a look at the image contents and text labels for the first nine examples in the training data set.

학습 데이터셋의 처음 9개의 샘플들에 대한 이미지와 텍스트 label을 살펴보겠습니다.

```{.python .input  n=27}
X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))
```

## Reading a Minibatch

To make our life easier when reading from the training and test sets we use a `DataLoader` rather than creating one from scratch, as we did in the section on ["Linear Regression Implementation Starting from Scratch"](linear-regression-scratch.md). The data loader reads a mini-batch of data with an example number of `batch_size` each time.

학습 데이터나 테스트 데이터를 읽는 코드를  ["Linear Regression Implementation Starting from Scratch"](linear-regression-scratch.md) 에서 처럼 직접 작성하지 않고 `DataLoad` 를 사용하도록 하겠습니다. 데이터 로더는 매 번 `batch_size` 개수의 샘플을 갖는 미니 배치를 읽습니다.

In practice, data reading is often a performance bottleneck for training, especially when the model is simple or when the computer is fast. A handy feature of Gluon's `DataLoader` is the ability to use multiple processes to speed up data reading (not currently supported on Windows). For instance, we can set aside 4 processes to read the data (via `num_workers`).

실제 수행을 할 때, 데이터를 읽는 것이 성능의 병목이 되는 것을 볼 수 있습니다. 특히, 모델이 간단하거나 컴퓨터가 빠를 경우에 더욱 그렇습니다. `DataLoader` 의 유용한 특징은 데이터 읽기 속도를 빠르게 하기 위해서 멀티 프로세스들 사용할 수 있다는 것입니다. (단 현재 Windows에서는 지원되지 않습니다) 예를 들면, `num_workers` 설정을 통해서 4개의 프로세스가 데이터를 읽도록 만들 수 있습니다.

In addition, we convert the image data from uint8 to 32-bit floating point numbers using the `ToTensor` class. Beyond that we divide all numbers by 255 so that all pixels have values between 0 and 1. The `ToTensor` class also moves the image channel from the last dimension to the first dimension to facilitate the convolutional neural network calculations introduced later. Through the `transform_first` function of the data set, we apply the transformation of `ToTensor` to the first element of each data example (image and label), i.e., the image.

추가적으로 `ToTensor` 클래스를 이용해서 이미지 데이터를 uint8에서 32 bit floating point number로 변환합니다. 이 후, 모든 숫자를 255로 나눠서 모든 픽셀의 값이 0과 1사이가 되도록 합니다. `ToTensor` 클래스는 이미지 채널을 마지막 차원에서 첫번째 차원으로 바꿔주는 기능이 있는데, 이는 다음에 소개할 convolutional neural network 계산과 관련이 있습니다. 데이터셋의 `transform_first` 함수를 이용하면,  `ToTensor` 의 변환을 각 데이터 샘플 (이미지와 label)의 첫번째 원소인 이미지에 적용할 수 있습니다.

```{.python .input  n=28}
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    # 0 means no additional processes are needed to speed up the reading of
    # data
    num_workers = 0
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)
```

The logic that we will use to obtain and read the Fashion-MNIST data set is encapsulated in the `d2l.load_data_fashion_mnist` function, which we will use in later chapters. This function will return two variables, `train_iter` and `test_iter`. As the content of this book continues to deepen, we will further improve this function. Its full implementation will be described in the section ["Deep Convolutional Neural Networks (AlexNet)"](../chapter_convolutional-neural-networks/alexnet.md).

Fashion-MNIST 데이터 셋을 가지고 와서 읽는 로직은 `g2l.load_data_fashion_mnist` 함수 내부에 구현되어 있습니다. 이 함수는 다음 장들에서 사용될 예정입니다. 이 함수는 `train_iter` 와 `test_iter` 두 변수를 리턴합니다. 이 책에서는 내용이 깊어지에 따라 이 함수를 향상시켜보겠습니다. 전체 구현에 대한 자세한 내용은  ["Deep Convolutional Neural Networks (AlexNet)"](../chapter_convolutional-neural-networks/alexnet.md) 절에서 설명하겠습니다.

Let's look at the time it takes to read the training data.

학습 데이터를 읽는데 걸리는 시간을 측정하보겠습니다.

```{.python .input}
start = time.time()
for X, y in train_iter:
    continue
'%.2f sec' % (time.time() - start)
```

## Summary

* Fashion-MNIST is an apparel classification data set containing 10 categories, which we will use to test the performance of different algorithms in later chapters.
* We store the shape of image using height and width of $h$ and $w$ pixels, respectively, as $h \times w$ or `(h, w)`.
* Data iterators are a key component for efficient performance. Use existing ones if available.
* Fashion-MNIST는 의류 분류 데이터 셋으로 10개의 카테고리로 분류되어 있습니다. 다음 장들에서 다양한 알고리즘의 성능을 테스트하는데 사용할 예정입니다.
* 이미지의 shape은 높이 `h` 픽셀, 넓이 `w` 픽셀을 이용해서  $h \times w$ 나 `(h, w)` 로 저장됩니다.
* 데이터 이터레이터는 효율적인 성능을 위한 중요한 컴포넌트입니다. 가능하면 제공되는 것들을 사용하세요.

## Problems

1. Does reducing `batch_size` (for instance, to 1) affect read performance?
1. For non-Windows users, try modifying `num_workers` to see how it affects read performance.
1. Use the MXNet documentation to see which other datasets are available in `mxnet.gluon.data.vision`.
1. Use the MXNet documentation to see which other transformations are available in `mxnet.gluon.data.vision.transforms`.
1. `batch_size` 를 줄이면 (예를 들면 1) 읽기 성능에 영향을 미칠까요?
1. Windows 사용자가 아니라면, `num_workers` 을 바꾸면서 읽기 성능이 어떻게 영향을 받는지 실험해보세요.
1. `mxnet.gluon.data.vision` 에서 어떤 데이터셋들이 제공되는지 MXNet 문서를 통해서 확인해보세요.
1. `mxnet.gluon.data.vision.transforms` 에서 어떤 변환들이 제공되는지 MXNet 문서를 통해서 확인해보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2335)

![](../img/qr_fashion-mnist.svg)
