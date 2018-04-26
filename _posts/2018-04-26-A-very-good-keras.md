---
layout: post
title: A Very Good Keras!!
description: "in this post, i consider about keras. keras is very useful Library using TF"
modified: 2018-03-26
tags: [Keras, Tensorflow, Machine Learning]
category: Tensorflow Basic
image:
  feature: posts/tensorflow_bg.jpg
  credit: tensorflow
  creditlink:
---

# Keras 소개
2017년도경 Keras가 Tensorflow의 코어 라이브러리로 지원되기 시작했지만, 나는 최근까지 이 유용한 라이브러리를 사용하지 않는 고집을 부리고 있었다..<br />
내가 Keras를 사용하지 않고, pure한 Tensorflow를 사용하기를 고집한 이유는 Keras를 사용하면 왠지 세세한 weight 조정이나 여러가지 제약이 있지 않을까라는 무의미, 무책임한 사고방식에서였다.
그런데 최근 Keras를 사용해보고나니 말도 안되게 편한 Library였다는 것을 새삼 깨닫는다..
<br />
Keras를 잠깐 사용하면서 느꼇던 것은, pure한 Tensorflow를 사용해서 코드를 구현할 때에 비해 Data Reading부터 Model, 최적화등을 약 2~3배 이상 빠르게 구현할 수 있을 정도로 간편하다는 것이다.
게다가, 앞서 말했던 세세한 weight 파라미터, Layer의 조정등도 매우 간편하게 수행할 수 있다.

<br />
따라서! 이번 포스트에서는 그렇게 간단하게 조작할 수 있었던 Keras에 대해 살펴보도록 하자. <br />
아직 나 자신도 Keras나 Tensorflow에서 매우 뛰어난 전문가가 아니고, 죄다 인터넷에서 검색하고 찾아본 내용을 정리한 것이니, 전문적인 내용은 Tensorflow나 Keras 공식 홈페이지의 Document를 확인하는 것이 가장 이해하기 쉬울 수 있다.

### 기본 활용 방법 구성
- 수집된 데이터 가공
    + 폴더 구성
- Data Augmentation
- Reading Data
- Model 구현
- Optimizing
- Classification

### 수집된 데이터 가공
앞서 Tensorflow에서는 수집한 데이터를 train, validation으로 나누고, csv 파일로 라벨링을 하는 등의 데이터 가공 프로세스를 별도로 구현할 필요가 있었다. <br />
그러나 Keras에서는 정말 매우매우 간단/간편하게 데이터를 폴더별로 구분만 해 놓으면 된다!! <br />
예를들어 다음 폴더 구성과 같다.

```
 .
├── train(folder)
│   └── label1
│           └── XXX-3.jpeg
│           └── ...
│   └── label2
│           └── XXX-3.jpeg
│           └── ...
├── validation(folder)
│   └── label1
│           └── XXX-3.jpeg
│           └── ...
│   └── label2
│           └── XXX-3.jpeg
│           └── ...
 .
```

이렇게만 구성해놓으면, keras의 ImageDataGenerator 라이브러리가 다 알아서 읽어들여서, 다 알아서 Augmentation하고, 다 알아서 라벨링도 한다!!

### Data Augmentation

Keras에서는 앞서 소개한 ImageDataGenerator 라이브러리를 제공하는데, 뛰어나게도 이 라이브러리는 Augmentation도 지원한다. <br />
아마 심플한 이미지 분류 작업에서 필요한 왠만한 Augmentation 방법은 다 제공하는것 같다. 대표적으로 horizontal/vertical filp, shift, rotate같은 것들이다. 자세한 코드는 뒤에서 소개한다.

### Reading Data

Tensorflow Document에서는 이전에 작성했던 포스트에서와같이 Data를 읽어들이는 방법으로 3가지를 제시하는데, 이 3가지 방식을 적용해서 model로 읽어들이는데에도 물론 코딩이 필요하다. <br />
특히 Disk에서 파이프라인 형태로 데이터를 조금씩 읽어들이면서 모델을 학습시키려면, tread도 구현해야되고, queue도 구현해야되고,,,,,,<br />
다행스럽게도 Keras의 ImageDataGenerator 라이브러리는 이러한 기능도 제공을 해서, 단순히 폴더 경로와 이미지 크기, batch_size 등등만 설정해서 호출하면, 마치 우리가 정제된 MNIST 데이터셋을 불러다 쓰는 것처럼 편리하게 사용할 수 있다.
이제 ImageDataGenerator 라이브러리에 대한 코드를 살펴보자.

```python
   from keras.preprocessing.image import ImageDataGenerator

   train_datagen = ImageDataGenerator(
       rescale=1./255,
       horizontal_flip=True,
       vertical_flip=True)

   validation_datagen = ImageDataGenerator(rescale=1./255)

   train_generator = train_datagen.flow_from_directory(
       'path/to/.../train',
       target_size=(128, 128),
       batch_size=32,
       class_mode='binary')

   validation_generator = validation_datagen.flow_from_directory(
       'path/to/.../validation',
       target_size=(128, 128),
       batch_size=32,
       class_mode='binary')
   ```

위 코드를 살펴보면 알수있다싶이 ImageDataGenerator는 keras.preprocessing.image 모듈내에 있으니까 상단 코드와 같이 import를 시켜주고, train, validation 각각 ImageDataGenerator에 대한 인스턴스를 생성한다.
인스턴스를 생성할 때, 생성자 인자로 보이는 것중에 rescale, horizontal_filp, vertical_filp이 보이는데 이 인자들이 augmentation 방법과 그 파라미터이다.
이후, 생성한 인스턴스의 flow_from_directory 함수를 호출해서 Disk로부터 데이터를 읽어들이면 끝! 정말 간단하지 않나요?? 그 함수 인자로도 위 코드에서 보이는것처럼 train, validation 폴더 path랑 입력 이미지 크기, 배치사이즈가 끝이라는게...

### Model 구현

이제 Keras를 통해, 커스텀 데이터를 쭉 읽어들일 수 있게 되었으니 Classification 모델을 구성할 차례이다. 먼저 모델을 구성하는 코드를 살펴보자.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(30, (5, 5), strides=(2, 2), input_shape=(128, 128, 3), activation='relu', kernel_regularizer=l2(0.0005)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(0.0005)))
classifier.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(0.0005)))
classifier.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(0.0005)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(0.0005)))
classifier.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(0.0005)))
classifier.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(0.0005)))
classifier.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(0.0005)))
classifier.add(Dropout(0.25))
classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0005)))
```

간단하게 CNN 모델을 구성하는 것으로 살펴보도록하자. CNN에는 기본적으로 Convolutional Layer, Pooling Layer, Fully Connected Layer 등등이 있는데, keras.layers 모듈에는 각 Layer를 구성할 수 있는 라이브러리가 다 있다. 따라서 코드 상단과 같이 Layer에 해당하는 라이브러리를 import한다. <br />
이후 모델을 구성하기 위해 Sequential을 import 하자. import한 이후에는 당연히 모델 인스턴스를 Sequential로 생성하고 그 이후에는 간단하게 add()를 호출해서 네트워크 모델을 구성할 수 있다. <br />
위 코드에서 보이는것처럼 add()함수의 인자로 앞서 import한 각 layer들의 인스턴스를 생성해서 넘겨주면 말 그대로 sequential하게 모델이 구성된다. <br />
보통 Neural Network의 학습에서 Overfitting을 방지하기 위해서 종종 regularization을 사용하는데, 모델을 구성할 때, 각 layer의 인스턴스 생성자 인자로 kernel_regularizer를 위 코드와같이 넘겨주면 간단하게 추가할 수 있다.

### Optimizing

이제 CNN 학습을 시작할 차례이다. CNN 학습에 대한 코드는 다음과 같다.

```python
sgd = SGD(lr=0.01, momentum=0.99, decay=0.1, nesterov=False)

# Compiling the CNN
classifier.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit_generator(train_generator,
                         steps_per_epoch=250000,
                         epochs=300,
                         validation_data=validation_generator,
                         validation_steps=2000)
```

위 코드와 같이, 학습을 위한 Optimizer로 Stochastic Gradient Descent를 적용하고, classifier.compile()함수의 optimizer 파라미터 인자로 넘겨주면, 모델의 optimizer 설정까지 셋팅할 수 있다. <br />
위 코드에서 loss는 'binary_corssentropy'로 설정했다. <br />

마지막으로 앞서 ImageDataGenerator로 읽어들이는 데이터를 가지고 모델을 학습(fitting)하는 방법은 classifier.fit_generator() 함수를 사용하면 된다. <br />
이렇게 설정한 이후 학습을 시작하면, 콘솔 화면에 알아서 각 epoch마다 걸리는 시간, training loss, training accuracy와
각 validation_step마다 validation loss와 validation accuracy가 알아서 프린팅된다.

## Additional Contents
### Setting weights
training이 끝난 이후, weights가 어떻게 바뀌었는지 궁금할 경우 다음과 같은 코드를 실행하면 쉽게 확인할 수 있다.

```python
classifier.layers[0].get_weights()
```

앞서 구성한 모델은 conv, pooling, conv, conv, conv ...로 구성되어있기 때문에, 레이어의 0번째 인자는 conv layer를 가리킨다. 따라서 위 코드를 실행하면, 학습된 모델의 첫 번째 conv 레이어의 weights 파라미터 값을 뽑아올 수 있다. <br />
뿐만 아니라, training 전에 weights 값을 내가 정한 값으로 설정하고 싶은 경우, 아래 코드를 실행하면 쉽게 적용할 수 있다.

```python
classifier.layers[0].set_weights(weight_param)
```

당연히 여기서 weight_param의 shape는 layers[0]의 shape와 동일해야 하며, numpy array를 넘겨주면 된다.

<br />
<br />
ps. 이제까지 정말 Tensorflow로 삽질 많이 한것 같은데,,, 이제 keras 써서 겁나 편하게 코딩해야겠다.
ps. 블로그 포스트 말투가 오락가락하네요.. 블로그 시작한지 얼마 안되서 글 쓰는 형식이 안갖춰져서 끙,, 조만간 갖춰지겠죠?ㅋ