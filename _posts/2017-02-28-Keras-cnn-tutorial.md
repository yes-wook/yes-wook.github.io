---
layout: post
title: Keras CNN tutorial
comments: true
---

본 글은 [Keras-tutorial-deep-learning-in-python](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)의 내용을  
제 상황에 맞게 수정하면서 CNN(Convolution neural network)을 만들어보는 예제이며,  
CNN의 기본데이터라 할 수 있는 MNIST(흑백 손글씨 숫자인식 데이터)를 이용할 것입니다.  
저도 Keras는 처음이고 하니, 시행착오가 있더라도 그대로 서술하겠습니다.  

그리고 Markdown의 사용이 익숙치 않아서 사진보다는 글이나 코드가 많습니다...이점 양해 부탁드립니다 ㅠㅠ    

우선 상단에 링크된 글의 목차는 다음과 같습니다.

>1. Set up your environment.  
>2. Install Keras.  
>3. Import libraries and modules.  
>4. Load image data from MNIST.  
>5. Preprocess input data for Keras.  
>6. Preprocess class labels for Keras.  
>7. Define model architecture.  
>8. Compile model.  
>9. Fit model on training data.  
>10. Evaluate model on test data.  

위의 순서를 따라 하나씩 차근차근 해보도록 하겠습니다.

# 1. Set up your environment. & 2. Install Keras. 
원본 글에서는 아래와 같은 설정을 권하고 있습니다.
>* Python 2.7+ (Python 3 is fine too, but Python 2.7 is still more popular for data science overall)  
>* SciPy with NumPy  
>* Matplotlib (Optional, recommended for exploratory analysis)  
>* Theano

Keras에 대한 간략한 설명을 하자면 Keras는 Backend로 **Theano와 Tensorflow 둘중 하나를 선택**할 수 있습니다.  
그러나 저는 윈도우 환경이기 때문에...Theano보다는 Tensorflow가 더 설치하기 편합니다.  

특히, 이것저것 많이 깔리는걸 싫어하는데 Theano같은 경우에는 MinGW나...뭐 그런걸 깔아줘야하더라구요..  
Tensorflow같은 경우에는 0.12버전 이후로 Windows를 정식 지원하게 되면서 간단히 설치할 수 있습니다. 

Tensorflow 설치방법은 [Tensorflow install](https://www.tensorflow.org/install/)에 자세히 나와있습니다.  
본 글에서는 기본적인 Python이나 여타 Tensorflow, Keras, NumPy 등의 라이브러리들은 설치되어있다고 가정하겠습니다.  

따라서 제 환경은 아래와 같습니다.  
>* Python 3.5.2 (with Anaconda)
>* Tensorflow 1.0 (GPU version, SciPy와 NumPy는 같이 설치됩니다)
>* Keras
>* Matplotlib (Optional)

설치부터 하셔야 하는 분들은 Anaconda를 설치하시면 Matplotlib이나 SciPy, NumPy같은것들이 포함되어있습니다.  
따라서 아나콘다를 설치하시길 권장드립니다.  

현재 본인의 설치버전은 다음과 같은 방법으로 알아 볼 수 있습니다.  
``` Python
CMD> python  

Python 3.5.2 |Anaconda custom (64-bit)| (default, Jul  5 2016, 11:41:13) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import keras
Using TensorFlow backend.
 c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:128] successfully opened CUDA library cublas64_80.dll locally
 c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:128] successfully opened CUDA library cudnn64_5.dll locally
 c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:128] successfully opened CUDA library cufft64_80.dll locally
 c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:128] successfully opened CUDA library nvcuda.dll locally
 c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:128] successfully opened CUDA library curand64_80.dll locally
>>> keras.__version__
'1.2.2'
>>> tensorflow.__version__
'1.0.0'
>>> numpy.__version__
'1.12.0'
>>> matplotlib.__version__
'1.5.3'
```
  
위와 같이 뜬다면 이제 Keras사용을 위한 설정은 (아마도?)완료되었습니다.  
이제  **keras\_cnn_example.py**라는 이름으로 빈 파일을 하나 만들면 됩니다.

# 3. Import libraries and modules.
본 예제에 사용될 라이브러리와 모듈들을 불러와봅시다.  
``` Python
>>> import numpy as np                  # NumPy
>>> np.random.seed(123)                 # 랜덤시드를 지정하면, 재실행시에도 같은 랜덤값을 추출합니다(reproducibility)
>>> from keras.models import Sequential # 이하 Keras 모듈들입니다.
Using TensorFlow backend.
>>> from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling
>>> from keras.utils import np_utils
```

# 4. Load image data from MNIST.  
뉴럴넷 예제의 기본데이터라고 할 수 있는 MNIST데이터를 불러와봅시다.  
MNIST는 흑백으로 된 손글씨 데이터로 0~9까지 총 10개의 숫자로 이루어져 있습니다.  
그리고 기본데이터답게, 웬만한 프레임워크들엔 쉽게 불러올 수 있도록 되어있습니다. Keras도 마찬가지입니다.  

``` Python
>>> from keras.datasets import mnist
# Load pre-shuffled MNIST data into train and test sets
>>> (X_train, y_train), (X_test, y_test) = mnist.load_data()
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.pkl.gz
>>> print(X_train.shape)
(60000, 28, 28)
>>> print(X_test.shape)
(10000, 28, 28)
>>> print(y_train.shape)
(60000,)
>>> print(y_test.shape)
(10000,)
```
위와같이 출력되면 로드가 완료된 것 입니다.  

보시면 아시겠지만 6만개의 트레이닝셋과 10000개의 테스트셋으로 구성되어있습니다.  
입력값(X)의 차원이 28*28인 것은 아시겠지만 MNIST는 가로28픽셀\*세로28픽셀로 이루어져있습니다.  

# 5. Preprocess input data for Keras.
데이터를 Keras에 맞게 변환(Reshape)해주는 과정입니다..  
MNIST에서 이게 왜필요하지?라는 생각이 들긴하는데,  
MNIST는 흑백이라서 어차피 채널이 1개뿐이지만, 만약 컬러이미지를 이용하여 트레이닝한다고 하면,  
컬러이미지는 RGB 각 1채널씩 총 3개의 채널로 구성되어있기 때문에 해줘야합니다.  

**단, Backend로 Theano를 사용할 경우엔 (channel, width, height)로 표현되지만**  
**Tensorflow를 사용할 경우엔 (width, height, channel)로 표현됩니다.**  

따라서, 원본글에는 
``` Python
>>> X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
>>> X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
```
과 같이 표현되어있지만,

저는 아래와 같이
``` Python
>>> X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
>>> X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
```
로 바꿔서 해보겠습니다(내맘)  

이제 확인해보면 Shape가 달라진 것을 볼 수 있습니다.
``` Python
>>> print(X_train.shape)
(60000, 28, 28, 1)
```

X데이터는 흑백이지만, 그 진하기에 따라 0~255까지의 숫자로 되어있습니다.  
0은 하얀색이고, 255는 검정색입니다.  
이를 0~1사이의 값으로 Nomalize해줍니다.

``` Python
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
```

# 6. Preprocess class labels for Keras. 
X데이터(인풋)을 reshape 해줬으니, 이번엔 Y데이터(아웃풋, class or label)도 수정을 해줘야합니다.  
One-hot encoding으로 수정을 해줄 것인데,  
간단하게 설명드리면 0~9까지의 숫자중에서 5를 One-hot encoding으로 표현하게 되면,  
5 => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 로 바뀌게 됩니다.  

현재 Y데이터는 
``` Python
>>> print (y_train[:10])
[5 0 4 1 9 2 1 3 1 4]
```
이렇게 숫자로 되어있습니다.
이를 One-hot encoding으로바꿔주면 되는데,  
다행히 NumPy에 이를 쉽게 해주는 기능이 있습니다. (는 저도 처음안 사실)

``` Python
>>> Y_train = np_utils.to_categorical(y_train, 10)
>>> Y_test = np_utils.to_categorical(y_test, 10)
>>> print (Y_train[:10])
[[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
 [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]]
```
위의 두 결과를 비교해보시면 One-hot encoding으로 성공적으로 변환 된 것을 보실 수 있습니다.  
(원본데이터는 소문자 y이고, 전처리된 데이터는 대문자 Y로 표현되었습니다)  
이제 모델 학습을 위한 데이터 준비는 끝났습니다.  
남은건 모델 구축과 학습, 평가 뿐입니다!  

# 7. Define model architecture. 
실제로 연구자들은 1%의 정확도를위해 다양한 모델들을 도입하지만,  
우리는 간단히 CNN을 이용해보겠습니다.

``` Python
>>> model = Sequential()
>>> model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(28,28, 1)))
>>> print(model.output_shape)
(None, 26, 26, 32)
```
첫번째 Convolution레이어를 추가했습니다.  
Convolution레이어를 하나 더 추가해보겠습니다.  

``` Python
>>> model.add(Conv2D(32, 3, 3, activation='relu'))
>>> model.add(pooling.MaxPooling2D(pool_size=(2,2)))
>>> model.add(Dropout(0.25))
>>> print(model.output_shape)
(None, 12, 12, 32)
```

두번째 Convolution레이어까지 추가되었습니다.
이 작업을 몇번 더 반복해도 되지만, 이정도까지만 하고 Fully-connected network를 추가해보겠습니다.  

``` Python
>>> model.add(Flatten())
>>> model.add(Dense(128, activation='relu'))
>>> model.add(Dropout(0.5))
>>> model.add(Dense(10, activation='softmax'))
>>> print(model.output_shape)
(None, 10)
```
# 8. Compile model.  
이제 모형(네트워크)의 구축이 완료되었습니다.  
Keras에서는 모형을 컴파일하는 과정이 필요한데, 이 과정에서 loss function이나 optimizer 등을 설정해주게 됩니다.  

간단히 설정해보겠습니다.
``` Python
>>> model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

보시는대로, loss function은 cross_entropy, optimizer는 Adam옵티마이저를 사용하여 Accuracy를 측정하도록 했습니다.

# 9. Fit model on training data.  

이제 모형을 학습해봅시다!  
``` Python
>>> model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
Epoch 1/10
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:885] Found device 0 with properties:
name: GeForce GTX 950
major: 5 minor: 2 memoryClockRate (GHz) 1.2155
pciBusID 0000:01:00.0
Total memory: 2.00GiB
Free memory: 1.65GiB
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:906] DMA: 0
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:916] 0:   Y
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 950, pci bus id: 0000:01:00.0)
60000/60000 [==============================] - 19s - loss: 0.2040 - acc: 0.9380
Epoch 2/10
60000/60000 [==============================] - 17s - loss: 0.0869 - acc: 0.9736
Epoch 3/10
60000/60000 [==============================] - 17s - loss: 0.0667 - acc: 0.9797
Epoch 4/10
60000/60000 [==============================] - 17s - loss: 0.0552 - acc: 0.9829
Epoch 5/10
30848/60000 [==============>...............] - ETA: 8s - loss: 0.0468 - acc: 0.9856
```
실행중인 내용입니다.
이전에 Tensorflow를 써봤던 경험에 비추어보면, 전반적으로 편리합니다.  
물론 텐서플로우의 자유도가 더 높긴 하지만, 저는 알고리즘의 개발이 아닌, 응용을 목적으로 하는 사람으로써 세세한 설정을 알아서 해주고, 특히 얼마나 남았는지가 눈에 보이는 점이 가장 좋습니다.  


# 10. Evaluate model on test data.  
이제 학습이 얼마나 잘 되었는지를 평가해봐야합니다.  
Keras에서는 이것조차 쉽습니다.

``` Python
>>> score = model.evaluate(X_test, Y_test, verbose=0)
>>> print(model.metrics_names)
['loss', 'acc']
>>> print(score)
[0.03182098812119552, 0.99129999999999996]
```
loss는 약 0.03, test데이터에 대한 accuracy는 약 0.99, 즉 99.12%가 됩니다.  

이렇게 Keras를 이용한 CNN의 구성과 MNIST 예제가 끝났습니다.  
사용된 코드는 본 글의 제일 하단에 있습니다.  

조만간 시간이 좀 나면 Autoencoder나 GAN등으로 돌아오겠습니다.  
왜냐면 제가 요즘 관심두고있거든요...ㅋㅋㅋㅋㅋ  

# Full code
``` Python
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling
from keras.utils import np_utils

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(28,28, 1)))
print(model.output_shape)

model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
              
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print(model.metrics_names)
print(score)
```