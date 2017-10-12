---
layout: post
title: Image denoising with Autoencoder in Keras
---

본 글은 [building-autoencoders-in-keras](https://blog.keras.io/building-autoencoders-in-keras.html)의 내용을 참고하여 작성되었습니다.
그림보다는 글과 코드 중심으로 작성되었으며, 최대한 상세하게 서술하겠지만 그림이 없어서 이해하기 어려울 수 있습니다.

# Autoencoder란?
Autoencoder는 비지도학습 알고리즘 중의 하나로, 인풋데이터를 학습하여 최대한 인풋데이터와 비슷한 아웃풋을 내는 것을 목표로 하는 신경망입니다.  

간단하게 과정을 설명하자면  아래와 같이 설명할 수 있습니다.
> X -> Encoding -> Decoding -> X'  

일반적으로 인코딩과정은 앞 단계의 노드수 보다 적은 수의 노드를 사용하는 레이어를 연결하여 구성됩니다.(128 -> 64 -> 32와 같은 식)  
따라서 **인코딩 과정은 데이터를 압축하는 과정**으로 볼 수 있습니다.

또한 디코딩과정은 인코딩과정과 반대로 앞 단계의 노드 수보다 많은 수의 노드를 사용하는 레이어를 연결하여 구성됩니다. (32 -> 64 -> 128과 같은 식)
따라서 **디코딩과정은 압축된 데이터를 다시 원본데이터로 복구하는 과정**입니다. 

전체 오토인코더는 Input데이터의 변수 갯수를 200개라고 가정했을 때, 아래와 같은 노드 수를 가지는 Fully-connected Network를 이용하여 학습됩니다.(레이어 별 노드 수는 다를 수 있습니다.)
> 200 -> 128 -> 64 -> 32 -> 64 -> 128 -> 200 

흔히 오토인코더를 두고 인코딩(압축)과 디코딩(복구)를 거치기 때문에 이러한 과정을 **_'데이터를 재구성한다'_** 라고 표현합니다.   

오토인코더는 데이터 자체의 구조를 학습하여 재구성해주기 때문에 데이터의 노이즈를 제거해줄 수 있습니다.
따라서 전처리에 많이 사용되며, 일반적으로 많이 사용되는 PCA(주성분분석)의 역할을 대신한다는(혹은 훨씬 잘한다는)결과도 존재합니다.

#  Keras를 이용한 Denoising autoencoder 
본 절에서는 Keras를 이용하여 Autoencoder를 구성하고,  MNIST데이터에 노이즈를 추가하여 이를 학습데이터로 사용하고, 타겟데이터로 노이즈를 추가하지 않은 데이터를 사용할 것입니다.  
본 코드의 최종목적은 Test 데이터에 대하여, 노이즈가 제거된 이미지를 얻는 것 입니다.

## 라이브러리 및 데이터 로딩
우선 필요한 라이브러리들을 로딩하고, MNIST데이터를 불러옵니다.  
오토인코더의 학습에는 라벨은 필요없기 때문에, 라벨은 로딩하지 않습니다.

``` Python
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

(x_train, _), (x_test, _) = mnist.load_data()
```

## 데이터 전처리
학습에 사용될 데이터를 전처리하는 과정입니다. 본 과정에서는 원본데이터를 정규화하고, 노이즈가 들어간 데이터를 생성 할 것입니다.  

MNIST데이터는 그 진하기에 따라 0~255까지의 숫자로 되어있기 때문에, 0~1의 범위로 정규화 해줍니다.  
원본데이터의 Min값이 0 이고 Max값이 255이기 때문에, 단순히 각 셀들을 255로 나눠주면 되는 작업입니다.  
또한 Numpy의 reshape를 이용하여 2-dimension으로 되어있는 MNIST데이터를 1-dimension으로 변경해줍니다.
``` Python
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))
```  
  
다음으로 노이즈가 들어간 데이터를 생성해보겠습니다.  
Numpy에서 무작위로 정규분포를 따라 값을 추출하고, 여기에 Noise factor를 곱하여 원본데이터를 조작합니다.  
노이즈팩터 값이 클수록 생성된 데이터가 알아보기 힘들 것 입니다.  
``` Python
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```
또한, Matplotlib을 통해 노이즈가 추가된 데이터를 확인할 수 있습니다.
``` Python
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
## 모형 구축 및 학습
이제 데이터의 준비가 다 끝났습니다. 모형을 구축할 차례입니다.  
MNIST의 Input dimension이 784(28*28)이기 때문에, 시작과 끝은 784개의 노드로 구성되어야 합니다.  
간단하게 784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784 로 모형을 구성하겠습니다.   
앞쪽 절반인 784 ~ 32까지가 인코딩과정, 뒤쪽 절반인 32~784까지가 디코딩과정입니다.  
``` Python
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(784, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
```
또한 학습까지 한번에 시켜봅시다. 
``` Python
model.fit(x_train_noisy, x_train, 
          nb_epoch=100,
          batch_size=256,
          shuffle=True,
          validation_data=(x_test_noisy, x_test))
```
저의 경우에는 100회 반복했을 때, 대략 아래와 같은 결과를 얻었습니다. 
>Epoch 98/100  
60000/60000 [==============================] - 1s - loss: 0.1095 - val_loss: 0.1143  
Epoch 99/100  
60000/60000 [==============================] - 1s - loss: 0.1094 - val_loss: 0.1128  
Epoch 100/100  
60000/60000 [==============================] - 1s - loss: 0.1094 - val_loss: 0.1126  


또한 Test데이터에 대해서, 아래와 같은 코드를 통해 원본, 노이즈, 디노이징 이미지를 확인 할 수 있습니다. 
``` Python
decoded_imgs = model.predict(x_test)
n = 10
plt.figure(figsize=(20, 6))
for i in range(1, n):
    # display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(3, n, i + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
그 결과는 아래와 같습니다. 노이즈가 상당함에도, 잘 제거된 것을 확인하실 수 있습니다.  
![no caption](/images/post/2017-03-03-Image.png)
위 : 원본 이미지, 중간 : 노이즈가 추가된 모습, 아래 : 노이즈가 제거된 모습

전체 코드는 하단에 있습니다.  

# Full code
``` Python
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

# MNIST 로딩 (라벨은 필요없기 때문에 버림)
(x_train, _), (x_test, _) = mnist.load_data()

# 데이터 정규화 및 Reshape
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))

# 원본데이터에 Noise 추가
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Noise가 추가된 데이터 확인
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 모형 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(784, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')

# 모형 학습
model.fit(x_train_noisy, x_train, 
          nb_epoch=100,
          batch_size=256,
          shuffle=True,
          validation_data=(x_test_noisy, x_test))

# 결과 확인
decoded_imgs = model.predict(x_test)
n = 10
plt.figure(figsize=(20, 6))
for i in range(1, n):
    # display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(3, n, i + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```