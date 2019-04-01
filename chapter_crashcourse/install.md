# Getting started with Gluon

To get started we need to download and install the code needed to run the notebooks. Although skipping this section will not affect your theoretical understanding of sections to come, we strongly recommend that you get some hands-on experience. We believe that modifying and writing code and seeing the results thereof greatly enhances the benefit you can gain from the book. In a nutshell, to get started you need to do the following steps:

시작하기에 앞서 노트북들을 수행하는데 필요한 코드를 다운로드하고 설치해야합니다. 이 절을 그냥 넘어가도 다음 절들에서 설명하는 이론적인 내용을 이해하는데는 문제가 없지만, 직접 코드를 실행해보는 것을 꼭 권장합니다. 코드를 작성하고, 수정하고 결과를 보는 것이 이 책으로 부터 더 많은 것을 얻을 수 있는 방법이기 때문입니다. 요약하자면, 아래 단계들을 수행하면됩니다.

1. Install conda
1. Download the code that goes with the book
1. Install GPU drivers if you have a GPU and haven't used it before
1. Build the conda environment to run MXNet and the examples of the book
1. conda 설치하기
1. 이책에 필요한 코드를 다운로드하기
1. GPU를 가지고 있는데 아직 설치하지 않았다면 GPU 드라이버 설치하기
1. MXNet와 이 책의 예제들을 수행할 conda 환경 빌드하기


## Conda

For simplicity we recommend [conda](https://conda.io), a popular Python package manager to install all libraries.

라이브러리를 설치의 간편함을 위해서, 유명한 Python 패키지 관리자인 [conda](https://conda.io) 를 권장합니다.

1. Download and install [Miniconda](https://conda.io/en/latest/miniconda.html) at [conda.io/en/latest/miniconda.html](https://conda.io/en/latest/miniconda.html) based on your operating system.
1. Update your shell by `source ~/.bashrc` (Linux) or `source ~/.bash_profile` (macOS). Make sure to add Anaconda to your PATH environment variable.
1. Download the tarball containing the notebooks from this book. This can be found at [www.d2l.ai/d2l-en-1.0.zip](https://www.d2l.ai/d2l-en-1.0.zip). Alternatively feel free to clone the latest version from GitHub.
1. Uncompress the ZIP file and move its contents to a folder for the tutorials.
1. [conda.io/en/latest/miniconda.html](https://conda.io/en/latest/miniconda.html) 를 방문해서 여러분의 OS에 맞는  [Miniconda](https://conda.io/en/latest/miniconda.html) 를 다운로드한 후 설치하세요.
1. 리눅스인 경우  `source ~/.bashrc`, macOS인 경우 `source ~/.bash_profile`를 수행해서 쉘을 업데이트 합니다. PATH 환경 변수에 Anaconda를 꼭 추가하세요.
1. 이 책의 노트북들이 담긴 tarball 를 다운로드하세요. 이 파일은 [www.d2l.ai/d2l-en-1.0.zip](https://www.d2l.ai/d2l-en-1.0.zip) 에 있습니다. 또는 GitHub 레파지토리에서 최신 버전을 복사해도 됩니다.
1. ZIP 파일의 압축을 풀어서 원하는 디렉토리에 옮겨놓으세요.

On Linux this can be accomplished as follows from the command line; For MacOS replace Linux by MacOSX in the first two lines, for Windows follow the links provided above.

리눅스의 경우 아래 명령들을 명령행에서 수행하면됩니다. MacOS를 사용하는 경우, 처음 두줄에 있는 Linux를 MacOS로 바꾸면 됩니다. Windows 사용자는 위 가이드에 있는 링크들을 참고하세요.

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
mkdir d2l-en
cd d2l-en
curl https://www.d2l.ai/d2l-en-1.0.zip -o d2l-en.zip
unzip d2l-en-1.0.zip
rm d2l-en-1.0.zip
```

## GPU Support

By default MXNet is installed without GPU support to ensure that it will run on any computer (including most laptops). If you should be so lucky to have a GPU enabled computer, you should modify the conda environment to download the CUDA enabled build. Obviously you need to have the appropriate drivers installed. In particular you need the following:

기본 설정인 경우 MXNet은 모든 컴퓨터에서 수행되는 것을 보장하기 위해서 GPU 지원을 하지 않도록 설치됩니다. 만약 여러분의 컴퓨터에 GPU가 있다면, conda 환경을 수정해서 CUDA가 활성화된 빌드를 다운로드해야합니다. 당연하지만, 필요한 드라이버들이 설치되어 있어야 합니다. 아래 단계들을 수행하세요.

1. Ensure that you install the [NVIDIA Drivers](https://www.nvidia.com/drivers) for your specific GPU.
1. Install [CUDA](https://developer.nvidia.com/cuda-downloads), the programming language for GPUs.
1. Install [CUDNN](https://developer.nvidia.com/cudnn), which contains many optimized libraries for deep learning.
1. Install [TensorRT](https://developer.nvidia.com/tensorrt), if appropriate, for further acceleration.
1. 여러분이 가지고 있는 GPU에 맞는 [NVIDIA Drivers](https://www.nvidia.com/drivers) 가 설치되어 있는지 확인하세요.
1. GPU를 위한 프로그래밍 언어인 [CUDA](https://developer.nvidia.com/cuda-downloads) 를 설치하세요.
1. 딥러닝을 위한 다양한 최적화 라이브러리를 제공하는 [CUDNN](https://developer.nvidia.com/cudnn) 을 설치하세요.
1. 추가적인 가속을 위해서 필요한 경우, [TensorRT](https://developer.nvidia.com/tensorrt) 를 설치합니다.

The installation process is somewhat lengthy and you will need to agree to a number of different licenses and use different installation scripts for it. Details will depend strongly on your choice of operating system and hardware.

설치는 다소 오래 걸리고, 라이센스 동의를 해야하기도 하고 여러 설치 스크립트를 사용해야합니다. 여러분의 OS나 하드웨어에 따라 설치 방법은 달라질 수 있습니다.

Next update the environment description in `environment.yml`. Replace `mxnet` by `mxnet-cu92` or whatever version of CUDA that you've got installed. For instance, if you're on CUDA 8.0, you need to replace `mxnet-cu92` with `mxnet-cu80`. You should do this *before* creating the conda environment. Otherwise you will need to rebuild it later. On Linux this looks as follows (on Windows you can use e.g. Notepad to edit `environment.yml` directly).

위 과정을 마친후,  `environment.yml` 의 환경 설명을 업데이트 합니다. `mxnet` 을 `mxnet-cu92` 또는 설치된 CUDA 버전에 맞도록 바꿉니다. 만약 CUDA 8.0 버전이 설치된 경우, `mxnet-cu92` 를 `mxnet-cu80` 으로 바꿔야 합니다. 이것은 꼭 conda 환경을 *만들기 전*에 해야합니다. 그렇지 않은 경우, 빌드를 다시 수행해야하기 때문입니다. 리눅스 사용자는 아래 명령으로 수정할 수 있습니다. (Windows 사용자는 Notepad 등을 사용해서 `environment.yml`를 직접 수정합니다.)

```
cd d2l
emacs environment.yml
```

## Conda Environment

In a nutshell, conda provides a mechanism for setting up a set of Python libraries in a reproducible and reliable manner, ensuring that all software dependencies are satisfied. Here's what is needed to get started.

요약하면, conda는 Python 라이브러리들을 반복가능하고 안정적인 방법으로 설정하는 방법을 제공합니다. 이를 통해서 모든 소프트웨어의 의존성을 만족시킬 수 있습니다. 시작하기에 필요한 것을 다음과 같습니다.

1. Create and activate the environment using conda. For convenience we created an `environment.yml` file to hold all configuration.
1. Activate the environment.
1. Open Jupyter notebooks to start experimenting.
1. conda를 이용해서 환경을 생성하고 활성화합니다. 편리하게 하기 위해서 `environment.yml` 파일에 모든 설정을 넣어놨습니다.
1. 환경을 활성화합니다.
1. Jupyter 노트북을 열어서 실험을 시작합니다.

### Windows

As before, open the command line terminal.

명령창을 엽니다.

```
conda env create -f environment.yml
cd d2l-en
activate gluon
jupyter notebook
```

If you need to reactivate the set of libraries later, just skip the first line. This will ensure that your setup is active. Note that instead of Jupyter Notebooks you can also use JupyterLab via `jupyter lab` instead of `jupyter notebook`. This will give you a more powerful Jupyter environment (if you have JupyterLab installed). You can do this manually via `conda install jupyterlab` from within an active conda gluon environment.

이후에 라이브러리들을 다시 활성화할 필요가 있다면, 첫번째 줄은 넘어가세요. 이는 여러분의 설정이 활성화되어 있음을 확인시켜줍니다. Jupyter Notebook(`jupyter notebook`)을 대신에, JupyterLab(`jupyter lab`)을 사용할 수 있음을 알아두세요. 활성화된 conda gluon 환경에서 `conda install jupyterlab` 을 실행해서 직접 설치할 수 있습니다.

If your browser integration is working properly, starting Jupyter will open a new window in your browser. If this doesn't happen, go to http://localhost:8888 to open it manually. Some notebooks will automatically download the data set and pre-training model. You can adjust the location of the repository by overriding the `MXNET_GLUON_REPO` variable.

웹 브라우저 통합이 잘 작동한다면, Jupyter 를 실행하면 웹 브라우저의 새 창이 생서됩니다. 만약 그렇지 않다면, 직접 http://localhost:8888를 열어보세요. 어떤 노트북은 필요한 데이터와 pre-trained 모델을 자동으로 다운로드하기도 합니다. `MXNET_GLUON_REPO` 변수를 수정해서 리파지토리 위치를 바꿀 수도 있습니다.

### Linux and MacOSX

The steps for Linux are quite similar, just that anaconda uses slightly different command line options.

리눅스에서도 비슷하게 합니다. 단지, anacond 명령 옵션이 약간 다릅니다.

```
conda env create -f environment.yml
cd d2l-en
source activate gluon
jupyter notebook
```

The main difference between Windows and other installations is that for the former you use `activate gluon` whereas for Linux and macOS you use `source activate gluon`. Beyond that, the same considerations as for Windows apply. Install JupyterLab if you need a more powerful environment

Windows와 다른 설치들의 주요 차이는 Windows에서는 `activate gluon` 을 사용하지만, Linux나 MacOS는 `source activate gluon` 을 사용한다는 것입니다. 그 외에는 Windows에서 했던 것과 동일합니다. 더 강력한 환경을 원한다면 JupyterLab을 설치하세요.

## Updating Gluon

In case you want to update the repository, if you installed a new version of CUDA and (or) MXNet, you can simply use the conda commands to do this. As before, make sure you update the packages accordingly.

새로운 CUDA 버전이나 MXNet을 설치해서 리파지토리를 업데이트하고 싶다면, conda 명령으로 간단하게 할 수 있습니다. 아래 명령으로 패키지를 업데이트 합니다.

```
cd d2l-en
conda env update -f environment.yml
```

## Summary

* Conda is a Python package manager that ensures that all software dependencies are met.
* `environment.yml` has the full configuration for the book. All notebooks are available for download or on GitHub.
* Install GPU drivers and update the configuration if you have GPUs. This will shorten the time to train significantly.
*  conda는 모든 소프트워어의 의존성을 만족하도록 해주는 Python 패키지 매니저입니다.
* `environment.yml` 은 이 책에 필요한 모든 설정을 가지고 있습니다. 모든 노트북에 대한 다운로드 링크를 제공하고, GitHub를 통해서도 제공합니다.
* GPU를 가지고 있다면, GPU 드라이버와 관련 설정을 업데이트하세요. 학습 시간을 아주 많이 줄일 수 있습니다.

## Problems

1. Download the code for the book and install the runtime environment.
1. Follow the links at the bottom of the section to the forum in case you have questions and need further help.
1. Create an account on the forum and introduce yourself.
1. 이 책의 코드를 다운로드하고, 실행 환경을 설치하세요.
1. 질문이나 도움이 필요하면 이 절의 아래에 있는 링크를 따라 포럼을 이용하세요.
1. 포럼의 계정을 만들고 여러분을 소개하세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2315)

![](../img/qr_install.svg)
