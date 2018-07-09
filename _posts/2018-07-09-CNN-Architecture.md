---
layout: post
title: CNN Models
description: "이 포스트에서는 LeNet, AlexNet, VGGNet부터 ResNet, GoogLeNet까지 다양한 CNN Model들과 Model들에서 사용한 주요 컨셉 등에 대해 다룰 예정입니다."
modified: 2018-03-26
tags: [CNN model]
category: CNN Architecture
image:
  feature: abstract-1.jpg
  credit:
  creditlink:
---

이 포스트에서는 다음과 같은 CNN Model 들과, 각 Model 에서 도입한 주요 컨셉들에 대해 살펴보겠습니다.
### 전통적인 CNN 모델
1. LeNet
2. AlexNet
3. VGGNet

### 2015년 이후의 CNN 모델
1. GoogLeNet(Inception)
2. ResNet
3. DenseNet
4. XceptionNet


## LeNet

## AlexNet

## VGGNet

## GoogLeNet(Inception)

먼저 GoogLeNet(Inception) 모델에 대해 살펴보기 전에, 먼저 1x1 Convolution 연산의 개념과 Network In Network 라는 모델 디자인적 컨셉에 대해 알아봅시다.

1x1 Convolution 연산은 말 그대로, Convolution filter 의 크기가 1x1 이라는 것을 의미하며 다음 그림과 같이 표현될 수 있습니다.

<figure>
	<img src="/images/contents/1x1Conv_1.jpg" alt="">
	<figcaption>1 Channel 입력에 대한 1x1 Convolution 연산</figcaption>
</figure>

위 그림에서 보이는 것과 같이 1x1 Convolution 연산은 단순히 각각의 입력 값에 1x1, 즉 하나의 값만 곱한 결과와 같습니다.

1 Channel 을 갖는 입력에 대해 (1개의 filter 를 갖는) 1x1 Convolution 연산은 Convolution 연산에서 얻을 수 있는 이점이 전혀 없어 보입니다. 하지만 다음 그림과 같이 여러개의 Channel 을 갖는 입력에 대해 1x1 Convolution 연산을 취하게 되면 특별한 결과를 얻을 수 있습니다.

<figure>
	<img src="/images/contents/1x1Conv_2.jpg" alt="">
	<figcaption>10 Channel 입력에 대한 1x1 Convolution 연산</figcaption>
</figure>

10개의 Channel 을 갖는 입력에 대해 (1개의 filter 를 갖는) 1x1 Convolution 연산을 취하면, 1개 Channel 을 갖는 결과를 얻을 수 있습니다.

여기서 주목할 점은, 1x1 Convolution 연산을 통해 출력 Channel 의 크기를 자유롭게 변경시킬 수 있다는 점입니다.

예를들어, 10개의 Channel 을 갖는 입력에 대해 10개의 filter 를 갖는 1x1 Convolution 연산을 취하게 되면, 다음 그림과 같이 10개의 Channel 을 갖는 결과를 얻을 수 있습니다.

<figure>
	<img src="/images/contents/1x1Conv_3.jpg" alt="">
	<figcaption>10 Channel 입력에 대한 10개 filter를 갖는 1x1 Convolution 연산</figcaption>
</figure>

마찬가지로 32개의 filter 를 갖는 1x1 Convolution 연산을 취하면 32개의 Channel 을 갖는 결과를 얻을 수 있습니다.

<figure>
	<img src="/images/contents/1x1Conv_4.jpg" alt="">
	<figcaption>10 Channel 입력에 대한 32개 filter를 갖는 1x1 Convolution 연산</figcaption>
</figure>

위와 같이 1x1 Convolution 연산을 취하게 되면, filter 의 개수에 따라 size 는 유지하면서 Channel 의 크기를 바꿀 수 있으며, Channel 의 크기를 작게함으로써 연산량을 크게 줄일 수 있습니다.

그렇다면 1x1 Convolution 연산과 Network In Network 라는 컨셉의 관계는 무엇일까요? Network In Network 라는 개념을 알아보기 위해 다음 그림을 살펴봅시다.

<figure>
	<img src="/images/contents/1x1Conv_NIN.jpg" alt="">
	<figcaption>1x1 Convolution 연산과 Network In Network</figcaption>
</figure>

이 그림은 5x5 크기와 10개의 Channel 을 갖는 입력에 대해 1개의 filter 를 갖는 1x1 Convolution 연산을 수행하는 것을 도식화한 것입니다.

그림에서 보이는 것과 같이, 1x1 Convolution 연산은 Channel 에 대해 일반적인 Neural Network(MLP)를 수행하는 것과 유사해보입니다.
따라서 연산 관점에서(Network in) Network 를 적용했기 때문에 Network In Network 라고 부른 것 같습니다. (제가 이해하기로는...)

1x1 Convolution 연산의 또 다른 이점은 비선형성(non linearity)을 부여할 수 있다는 점입니다. 즉 비선형성을 적용하면서 원하는 크기의 Channel 의 출력을 도출할 수 있다는 점입니다.
당연히 1x1 Convolution 연산 이후 "relu"와 같은 non linear activation 함수를 적용하면 비선형성을 적용한 결과를 얻을 수 있겠죠

자, 이제 이러한 1x1 Convolution 연산에 대한 내용을 숙지하고, google 연구팀이 개발한 GoogLeNet(inception)에 대해 살펴봅시다.
이름이 GoogLeNet(inception)인 이유는, CNN 연구의 시작을 알린 LeNet에 대한 경의를 표하기 위함이라고 합니다. inception 이라는 이름은 또 다른 이름인데, inception 영화의 ["We Need To Go Deeper"](http://knowyourmeme.com/memes/we-need-to-go-deeper) 라는 표현에 영감을 받아 사용했다고 하네요.

만약 CNN 모델을 디자인하고자 한다면 Convolutional Layer의 filter 크기를 1x3으로 할지, 3x3으로 할지, 5x5로 할지, 어느 시점에 어떤 크기의 pooling layer를 적용할지를 고민해야 합니다.

여기서 GoogLeNet(inception) 모델이 시도한 것은, "이것들 전부를 적용하자!" 입니다.



## ResNet

## DenseNet

## XceptionNet