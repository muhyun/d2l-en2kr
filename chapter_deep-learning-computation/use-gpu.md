# GPUs

In the introduction to this book we discussed the rapid growth of computation over the past two decades. In a nutshell, GPU performance has increased by a factor of 1000 every decade since 2000. This offers great opportunity but it also suggests a significant need to provide such performance.

이 책의 소개에서 우리는 지난 2세기동안의 연산 능력의 급격한 증가에 대해서 논의했습니다. 간단하게 말하면, GPU 성능이 2000년부터 10년마다 약 1000배씩 증가해왔습니다. 이런 것이 우리에게 엄청난 기회를 준기도 했지만, 그러한 성능이 필요한 상황도 많이 제안해왔습니다.

|Decade|Dataset|Memory|Floating Point Calculations per Second|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (social network)|100 GB|1 PF (NVIDIA DGX-2)|

|연도(10년단위)|데이터셋|메모리|초당 부동소수점 연산수|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (social network)|100 GB|1 PF (NVIDIA DGX-2)|

In this section we begin to discuss how to harness this compute performance for your research. First by using single GPUs and at a later point, how to use multiple GPUs and multiple servers (with multiple GPUs). You might have noticed that MXNet NDArray looks almost identical to NumPy. But there are a few crucial differences. One of the key features that differentiates MXNet from NumPy is its support for diverse hardware devices.

여러분의 연구를 위해서 이 컴퓨팅 성능을 활용하는 방법에 대해서 논의하는 것으로 시작해보겠습니다. 우선은 하나의 GPU를 사용해보겠고, 이후에는 여러 GPU 및 여러 GPU를 갖는 여러 서버를 사용하는 방법에 대해서 다루겠습니다. 이미 눈치 채었겠지만, MXNet NDArray는 NumPy와 거의 유사합니다. 하지만, 몇가지 중요한 차이점들이 있습니다. MXNet를 NumPy와 다르게 만드는 중요한 특징 중에 하나는 다양한 하드웨어 디바이스를 지원한다는 점입니다.

In MXNet, every array has a context. In fact, whenever we displayed an NDArray so far, it added a cryptic `@cpu(0)` notice to the output which remained unexplained so far. As we will discover, this just indicates that the computation is being executed on the CPU. Other contexts might be various GPUs. Things can get even hairier when we deploy jobs across multiple servers. By assigning arrays to contexts intelligently, we can minimize the time spent transferring data between devices. For example, when training neural networks on a server with a GPU, we typically prefer for the model’s parameters to live on the GPU.

MXNet에 모든 배열은 context를 갖습니다. 사실, 설명을 하지는 않았지만 지금까지 NDArray를 출력할 때마다, `@cpu(0)` 라는 이상한 내용이 결과에 함께 출력되었었습니다. 이것이 의미하는 것은 해당 연산이 CPU에서 수행되었다는 것입니다. 다른 context들로는 다양한 GPU들이 될 수 있습니다. 작업을 여러 서버 배포하는 경우에는 상황이 더 어려워질 수 있습니다. 배열을 context들에 지능적으로 할당하면, 디바이스간에 데이터가 전송되는 시간을 최소화할 수 있습니다. **예를 들면, GPU 하나를 가지고 있는 서버에서 뉴럴 네트워크를 학습시키는 경우, 모델 파라메터가 GPU에 상주하게 하는 것을 선호합니다.**

In short, for complex neural networks and large-scale data, using only CPUs for computation may be inefficient. In this section, we will discuss how to use a single NVIDIA GPU for calculations. First, make sure you have at least one NVIDIA GPU installed. Then, [download CUDA](https://developer.nvidia.com/cuda-downloads) and follow the prompts to set the appropriate path. Once these preparations are complete, the `nvidia-smi` command can be used to view the graphics card information.

요약하면, 복잡한 뉴럴 네트워크나 큰 스케일의 데이터를 다룰 때, CPU 하나 또는 여러개를 사용해서 연산을 수행하는 것은 비효율적일 수 있습니다. 이 절에서 우리는 하나의 NVIDIA GPU를 사용해서 연산을 수행하는 것을 설명하겠습니다. 우선, 여러분의 시스템에 적어도 한개의 NVIDIA GPU가 설치되어 있는지 확인하세요. 준비가 끝났다면, `nvidia-smi` 명령을 사용해서 그래픽 카드 정보를 조회해볼 수 있습니다.

```{.python .input  n=1}
!nvidia-smi
```

Next, we need to confirm that the GPU version of MXNet is installed. If a CPU version of MXNet is already installed, we need to uninstall it first. For example, use the `pip uninstall mxnet` command, then install the corresponding MXNet version according to the CUDA version. Assuming you have CUDA 9.0 installed, you can install the MXNet version that supports CUDA 9.0 by `pip install mxnet-cu90`. To run the programs in this section, you need at least two GPUs.

그 다음, GPU 버전의 MXNet이 설치되어있는지 확인하세요. 만약 CPU 버전의 MXNet이 이미 설치되어 있는 경우에는 우선 MXNet을 제거해야합니다. 즉, `pip uninstall mxnet` 명령으로 제거하고, 시스템에 설치된 CUDA 버전에 대응하는 MXNet 번을 설치합니다. CUDA 9.0이 설치되어 있다고 가정하면, CUDA 9.0을 지원하는 MXNet 버전 설치는 `pip install mxnet-cu90` 명령으로 합니다. 이 절의 프로그램들을 수행하기 위해서는 최소 두개 GPU들이 필요합니다.

Note that this might be extravagant for most desktop computers but it is easily available in the cloud, e.g. by using the AWS EC2 multi-GPU instances. Almost all other sections do *not* require multiple GPUs. Instead, this is simply to illustrate how data flows between different devices.

대부분의 데스크탑 컴퓨터에 GPU 두개가 설치된 경우는 드물지만, 클라우드에서는 이런 시스템을 구하기 쉽습니다. 예를 들면, AWS 클라우드의 멀티 GPU를 제공하는 EC2 인스턴스를 사용할 수 있습니다. 거의 모든 다른 절들에서는 다중 GPU를 필요로하지는 않습니다. 여기서는 데이터가 서로 다른 디바이스간에 어떻게 이동하는지를 설명하기 위해서 여러 GPU가 필요합니다.

## Computing Devices

MXNet can specify devices, such as CPUs and GPUs, for storage and calculation. By default, MXNet creates data in the main memory and then uses the CPU to calculate it. In MXNet, the CPU and GPU can be indicated by `cpu()` and `gpu()`. It should be noted that `mx.cpu()` (or any integer in the parentheses) means all physical CPUs and memory. This means that MXNet's calculations will try to use all CPU cores. However, `mx.gpu()` only represents one graphic card and the corresponding graphic memory. If there are multiple GPUs, we use `mx.gpu(i)` to represent the $i$-th GPU ($i$ starts from 0). Also, `mx.gpu(0)` and `mx.gpu()` are equivalent.

MXNet은 값의 저장과 연산에 사용될 CPU나 GPU와 같은 디바이스를 지정할 수 있습니다. 기본 설정으로 MXNet은 메인 메모리에 데이터를 생성하고, CPU를 사용해서 연산을 수행합니다. MXNet에서는 CPU와 GPU는 각각 `cpu()` 와 `gpu()` 로 표현됩니다. `mx.cpu()` (또는 괄호안에 아무 정수를 사용)는 모든 물리적인 CPU들과 메모리를 의미한다는 것을 기억해두세요. 즉, MXNet은 연산을 수행할 때 모든 CPU 코어를 사용하려고 합니다. 반면에 `mx.gpu()` 는 하나의 그래픽 카드와 그 카드의 메모리를 지정합니다. 만약 여러 GPU를 가지고 있다면,  $i$ 번째 GPU를 ($i$는 0부터 시작) 지정하는 방법은 `mx.gpu(i)` 라고 명시하면됩니다. 참고로 `mx.gpu(0)` 과 `mx.gpu()` 는 같은 표현입니다.

```{.python .input}
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

mx.cpu(), mx.gpu(), mx.gpu(1)
```

## NDArray and GPUs

By default, NDArray objects are created on the CPU. Therefore, we will see the `@cpu(0)` identifier each time we print an NDArray.

앞에서도 말했듯이 기본 설정은 NDArray 객체를 CPU에 생성합니다. 따라서, NDArray를 출력할 때, `@cpu(0)` 라는 식별자를 보게됩니다.

```{.python .input  n=4}
x = nd.array([1, 2, 3])
x
```

We can use the `context` property of NDArray to view the device where the NDArray is located. It is important to note that whenever we want to operate on multiple terms they need to be in the same context. For instance, if we sum two variables, we need to make sure that both arguments are on the same device - otherwise MXNet would not know where to store the result or even how to decide where to perform the computation.

NDArray의 `context` 속성을 사용해서 NDArray 객체가 위치한 디바이스를 확인할 수 있습니다. 여러 객체들에 대한 연산을 수행할 때는 항상 그 객체들은 모두 같은 context에 있어야하한다는 것을 명심하세요. 즉, 두 변수를 더하는 경우, 두 변수가 같은 디바이스에 있어야한다는 의미입니다. 그렇지 않을 경우에는 MXNet은 결과를 어느 곳에 저장할지 또는 연산을 어느 곳에서 수행해야할지를 알 수가 없습니다.

```{.python .input}
x.context
```

### Storage on the GPU

There are several ways to store an NDArray on the GPU. For example, we can specify a storage device with the `ctx` parameter when creating an NDArray. Next, we create the NDArray variable `a` on `gpu(0)`. Notice that when printing `a`, the device information becomes `@gpu(0)`. The NDArray created on a GPU only consumes the memory of this GPU. We can use the `nvidia-smi` command to view GPU memory usage. In general, we need to make sure we do not create data that exceeds the GPU memory limit.

GPU에 NDArray를 저장하는 방법은 여러가지가 있습니다. NDArray 객체를 생성할 때,  `ctx` 파라메터를 이용해서 저장할 디바스 지정이 가능합니다. 예를 들어,  `gpu(0)` 에 NDArray 변수 `a` 를 생성합니다. `a` 를 출력하면, 디바이스 정보가 `@gpu(0)` 으로 나오는 것을 확인해보세요. GPU에서 만들어진 NDArray는 그 GPU의 메모리만 사용합니다. GPU 메모리 사용양은 `nvidia-smi` 명령으로 확인이 가능합니다. 일반적 우리는 GPU 메모리 크기를 넘어서 데이터를 생성하지 않도록 해야합니다.

```{.python .input  n=5}
x = nd.ones((2, 3), ctx=mx.gpu())
x
```

Assuming you have at least two GPUs, the following code will create a random array on `gpu(1)`.

최소한 두개 GPU가 있다고 하면, 아래 코드는 난수 배열을 `gpu(1)` 에 생성합니다.

```{.python .input}
y = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(1))
y
```

### Copying

If we want to compute $\mathbf{x} + \mathbf{y}$ we need to decide where to perform this operation. For instance, we can transfer $\mathbf{x}$ to `gpu(1)` and perform the operation there. **Do not** simply add `x + y` since this will result in an exception. The runtime engine wouldn't know what to do, it cannot find data on the same device and it fails.

 $\mathbf{x} + \mathbf{y}$ 를 계산하고자 한다면, 이 연산을 어느 디바이스에서 수행할지를 결정해야합니다.  $\mathbf{x}$ 를 `gpu(1)`로 옮기고, 연산을 거기서 수행할 수 있습니다. **단순히 `x + y` 를 수행하지 마세요.** 만약 그렇게 할 경우, 예외가 발생할 것입니다. 왜냐하면, 런타임 엔진은 무엇을 해야할지 모르고, 같은 디이바이스에서 데이터를 찾을 수 없어서 연산이 실패하기 때문입니다.

![Copyto copies arrays to the target device](../img/copyto.svg)

`copyto` copies the data to another device such that we can add them. Since $\mathbf{y}$ lives on the second GPU we need to move $\mathbf{x}$ there before we can add the two.

`copyto` 메소드는 데이터를 다른 디바이스로 복사해서, 연산을 할 수있도록 해줍니다.  $\mathbf{y}$ 는 두번째 GPU에 있으니, 우리는 연산을 수행하기 전에 $\mathbf{x}$ 를 그 디바이스로 옮겨야합니다.

```{.python .input  n=7}
z = x.copyto(mx.gpu(1))
print(x)
print(z)
```

Now that the data is on the same GPU (both $\mathbf{z}​$ and $\mathbf{y}​$ are), we can add them up. In such cases MXNet places the result on the same device as its constituents. In our case that is `@gpu(1)`.

자 이제 데이터가 모두 같은 GPU에 있으니, 두 값을 더할 수 있습니다. MXNet은 연산 결과를 다시 같은 디바이스에 저장합니다. 지금 예의 경우는 `@gpu(1)` 입니다.

```{.python .input}
y + z
```

Imagine that your variable z already lives on your second GPU (gpu(0)). What happens if we call z.copyto(gpu(0))? It will make a copy and allocate new memory, even though that variable already lives on the desired device!
There are times where depending on the environment our code is running in, two variables may already live on the same device. So we only want to make a copy if the variables currently lives on different contexts. In these cases, we can call `as_in_context()`. If the variable is already the specified context then this is a no-op. In fact, unless you specifically want to make a copy, `as_in_context()` is the method of choice.

변수 $\mathbf{z}$ 는 두번째 GPU, gpu(1),에 있는데, 만약 `z.copyto(gpu(1))` 를 수행하면 어떻게 될까요? 답은 이미 같은 GPU에 값이 있더라도 새로운 메모리를 할당해서 값을 복사합니다. 프로그램이 수행되는 환경에 따라서 두 변수가 이미 같은 디바이스에 있는 경우도 있습니다. 우리는 변수가 다른 context에 있을 때만 복사를 수행하기 원합니다. 이 경우, `as_in_context()` 를 이용하면 됩니다. 먄약 변수가 지정된 context에 있는 경우리면, 아무 일이 일어나지 않습니다. 진짜로 데이터의 복제본을 만드는 경우가 아니라면, `as_in_context()` 를 사용하세요.

```{.python .input}
z = x.as_in_context(mx.gpu(1))
z
```

It is important to note that, if the `context` of the source variable and the target variable are consistent, then the `as_in_context` function causes the target variable and the source variable to share the memory of the source variable.

소스와 타겟 변수의 `context` 가 동일하다면, `as_in_context1 함수는 타켓 변수와 소스 변수가 소스 변수의 메모리를 공유한다는 사실을 기억해두는게 중요합니다.

```{.python .input  n=8}
y.as_in_context(mx.gpu(1)) is y
```

The `copyto` function always creates new memory for the target variable.

반면, `copyto` 함수는 타겟 변수를 위해서 항상 새로운 메모리를 만듭니다.

```{.python .input}
y.copyto(mx.gpu()) is y
```

### Watch Out

People use GPUs to do machine learning because they expect them to be fast. But transferring variables between contexts is slow. So we want you to be 100% certain that you want to do something slow before we let you do it. If MXNet just did the copy automatically without crashing then you might not realize that you had written some slow code.

사람들은 빠른 속도를 기대하면서 머신러닝을 수행할 때 GPU들을 사용합니다. context들 사이에 변수를 이동하는 것은 느립니다. 우리가 그렇게 하라고 하기전에 이미 많은 경우 사람들은 느린 무언가을 수행합니다. 예를 들면, MXNet이 복사를 문제를 발생하지 않고 자동으로 수행했다면, 느리게 동작하는 코드를 작성했다는 것을 눈치채지 못할 것입니다.

Also, transferring data between devices (CPU, GPUs, other machines) is something that is *much slower* than computation. It also makes parallelization a lot more difficult, since we have to wait for data to be sent (or rather to be received) before we can proceed with more operations. This is why copy operations should be taken with great care. As a rule of thumb, many small operations are much worse than one big operation. Moreover, several operations at a time are much better than many single operations interspersed in the code (unless you know what you're doing). This is the case since such operations can block if one device has to wait for the other before it can do something else. It's a bit like ordering your coffee in a queue rather than pre-ordering it by phone and finding out that it's ready when you are.

디바이스간(CPU, GPU, 다른 머신)에 데이터를 옮기는 것은 연산보다 **훨씬 느립니다.** 더군다나 병렬화(parallelization)를 더 어렵게 만듭니다. 연산을 계속 수행하기 전에 데이터가 보내지거나 받아지는 것이 끝날 때까지 대기해야하기 때문입니다. 그렇게 때문에 복사 연산은 아주 조심해서 수행해야합니다. 경험적인 법칙으로 작은 연산을 많이 하는 것은 큰 연산보다 훨씬 나쁘고, 여러 연산을 동시에 수행하는 것은 하나의 연산을 여러개를 수행하는 것보다 나쁩니다. 이런 경우들은 다른 무언가를 하기전에 한개의 디바이스가 다른 디바이스를 기달려야하는 예들입니다. 스마트폰으로 미리 주문한 후 도착하면 커피가 준비되어 있는 것이 아닌 줄을 서서 커피를 주문하는 것과 유사합니다.

Lastly, when we print NDArray data or convert NDArrays to NumPy format, if the data is not in main memory, MXNet will copy it to the main memory first, resulting in additional transmission overhead. Even worse, it is now subject to the dreaded Global Interpreter Lock which makes everything wait for Python to complete.

마지막으로는 메인 메모리에 데이터가 있는 경우가 아닐 때, NDArray 데이터를 출력하거나 NDArray를 NumPy 형태로 바꾸는 경우에 MXNet은 먼저 데이터를 메인 메모리에 복사합니다. 즉, 전송 오버해드가 발생합니다. 더 나쁜 사실은 모든 것이 Python이 완려되기를 기다리는 글로벌 인터프린터 락에 종속된다는 것입니다.


## Gluon and GPUs

Similar to NDArray, Gluon's model can specify devices through the `ctx` parameter during initialization. The following code initializes the model parameters on the GPU (we will see many more examples of how to run models on GPUs in the following, simply since they will become somewhat more compute intensive).

NDArray와 비슷하게 Gluon의 모델도 초기화에 `ctx` 파라메터를 통해서 context를 지정할 수 있습니다. 아래 코드는 모델 파라메터를 GPU에서 초기화합니다.

```{.python .input  n=12}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())
```

When the input is an NDArray on the GPU, Gluon will calculate the result on the same GPU.

입력이 GPU에 있는 NDArray 객체라면, Gluon은 같은 GPU에서 연산을 수행합니다.

```{.python .input  n=13}
net(x)
```

Let us confirm that the model parameters are stored on the same GPU.

모델 파라메터들이 같은 GPU에 저장되어 있는지 확인해보겠습니다.

```{.python .input  n=14}
net[0].weight.data()
```

In short, as long as all data and parameters are on the same device, we can learn models efficiently. In the following we will see several such examples.

요약하면, 모든 데이터와 파라메터들이 같은 디바이스에 있어야 모델을 효과적으로 학습시킬 수 있습니다. 앞으로 그런 예제들을 여러개 보게될 것입니다.

## Summary

* MXNet can specify devices for storage and calculation, such as CPU or GPU. By default, MXNet creates data in the main memory and then uses the CPU to calculate it.
* MXNet requires all input data for calculation to be **on the same device**, be it CPU or the same GPU.
* You can lose significant performance by moving data without care. A typical mistake is as follows: computing the loss for every minibatch on the GPU and reporting it back to the user on the commandline (or logging it in a NumPy array) will trigger a global interpreter lock which stalls all GPUs. It is much better to allocate memory for logging inside the GPU and only move larger logs.
* MXNet은 저장과 연산을 수행할 디바이스 (GPU, GPU)를 지정할 수 있습니다. 기본 설정으로 MXNet은 메인 메모리에 데이터를 생성하고, CPU를 사용해서 연산을 수행합니다.
* MXNet은 모든 입력 데이터가 **동일한 디바이스** (CPU 또는 같은 GPU)에 있어야 연산을 수행할 수 있습니다.
* 데이터를 조심하게 옮기지 않을 경우 상단한 성능 손실이 발생합니다. 전형적인 실수는 다음과 같습니다. GPU를 이용해서 미니 배치의 loss를 계산하고, 매번 화면에 출력 (또는 NumPy 배열에 추가)을 하는 경우. 이 경우, 글로벌 인터프린터 락이 필요하기 때문에 모든 GPU가 멈춰야합니다. 권장하는 방법은 GPU에 로깅을 위한 메모리를 할당하고, 큰 로그를 옮기는 것입니다.

## Problems

1. Try a larger computation task, such as the multiplication of large matrices, and see the difference in speed between the CPU and GPU. What about a task with a small amount of calculations?
1. How should we read and write model parameters on the GPU?
1. Measure the time it takes to compute 1000 matrix-matrix multiplications of $100 \times 100​$ matrices and log the matrix norm $\mathrm{tr} M M^\top​$ one result at a time vs. keeping a log on the GPU and transferring only the final result.
1. Measure how much time it takes to perform two matrix-matrix multiplications on two GPUs at the same time vs. in sequence on one GPU (hint - you should see almost linear scaling).
1. 큰 행렬의 곱같은 큰 연산을 수행하면서 CPU와 GPU의 속도 차이를 관찰해보세요. 작은 크기의 연산은 어떤가요?
1.  GPU에 파라에터를 읽고 쓰기를 어떻게 하나요?
1.  $100 \times 100$ 행렬들의 행렬 곱 1000개를 수행하고, 행럴 norm $\mathrm{tr} M M^\top$ 매번 출력하는 것과 GPU에 로그를 저장한 후 마지막에 최종 결과만 옮길 때 각 수행 시간을 측정해보세요
1. 두개의 GPU에서 두 행렬 곱을 동시에 수행하는 것과, 하나의 GPU에서 순서대로 수행하면서 수행 시간을 측정해보세요. 힌트 - 선형적인 성능 수치를 볼 것입니다.

## References

[1] CUDA download address. https://developer.nvidia.com/cuda-downloads

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2330)

![](../img/qr_use-gpu.svg)
