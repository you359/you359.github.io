---
layout: post
title: Paper Review - All Convolutional Net
description: "Three Visualization Techniques(backpropagation, deconvnet, guided-backpropagation)"
modified: 2018-08-23
tags: [Review, Keras, Guided-Backpropoagation]
category: CNN Visualization
image:
  feature: posts/data-vis.jpg
  credit: Lauren Manning
  creditlink: https://www.flickr.com/photos/laurenmanning/5674438518
---

# All Convolutional Net
이 포스트에서는 2015년 ICLR에 실린 "STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET"에 대해서 살펴보겠습니다.

## Key Point
- max-pooling can simply be replaced by a convolutional layer with increased stride

이 논문에서 주장하는 주된 논점은 "복잡한 activation function 이나 response normalization, max-pooling 연산 없이, Convolutional layer 들로 구성된 CNN도 충분히 좋은 성능을 가지고 있다." 입니다.
또한, Guided-Backpropagation 이라는 시각화 기법을 제안하면서, backpropagation과 deconvnet 과 비교도 하고 있습니다.

이 포스트에서는 논문에서 제안한 시각화 기법인 Guided-Backpropagation을 조금 더 중점적으로 살펴볼 예정입니다.

<figure>
	<img src="/images/contents/allconvnet_pooling_conv.png" alt="">
	<figcaption>pooling and convolution</figcaption>
</figure>

CNN에서 pooling layer가 사용되는 이유가 무엇일까요? 다음과 같이 정리해볼 수 있을 것 같습니다.
- feature들을 좀 더 invariant 하게 만들 수 있음
- (spatial dimensionality reduction 을 수행함으로써) 큰 사이즈의 이미지 입력을 허용함
- feature-wise pooling은 optimization을 쉽게 만들어줌(?)

그럼 이러한 pooling layer의 단점은 무엇일까요?
- pooling을 통해 dimensionality reduction을 수행하면, 일부 feature information이 손실될 수 있음

이 논문에서는 이러한 pooling layer를 convolution layer로 대체하려고 합니다. <br />
pooling layer가 사용되는 가장 중요한 이유 중 하나를 spatial dimensionality reduction 으로 보고, 동일 기능을 하면서 feature information 을 잃지 않는 방식으로말이죠.

교체 방법은 2가지가 있습니다.
1. CNN에서 각 pooling layer를 제거하고, 이미 있는 convolution layer의 stride를 증가시켜서 연산량 유지
2. CNN에서 pooling layer를 1보다 큰 stride를 갖는 convolution layer로 교체
    - 즉, 출력 채널과 입력 채널을 갖게 유지하면서, stride로 dimensionality reduction을 수행해서 pooling과 동일한 shape의 출력을 생성하는 convolution layer로 교체

이렇게 구성한 All Convolutional Network가 꾀 효과적인 성능을 보였고,
이 논문에서는 추가로 이렇게 변환한 CNN에 대해 여러 방식의 시각화 기법을 적용했습니다.

이 논문이 발표되기 이전, 2013년도 ZFNet을 연구한 논문에서는, Deconvolution을 이용해서 neuron의 시각화를 했었는데, 이러한 시각화 방식의 큰 단점은 pooling에서 온다고 말하고 있습니다.
즉, deconvnet에서 CNN의 invert를 지원하기 위해서 pooling layer에 switch라는 개념을 도입했는데, 이러한 시각화 기법은 입력 이미지의 상태에 따라 좌지우지되고, 학습된 feature들을 직접적으로 시각화하지 못한다고 합니다.

좀더 디테일하게 정리하자면,
lower layer에는 한정된 양의 invariance를 내포한 일반적인 feature들을 학습하기 때문에, 해당 layer들이 activate하는 단순 패턴들을 reconstruct할 수 있지만, <br/>
higher layer에는 비교적 invariant한 표현들을 하기 때문에, 해당 layer의 neuron들을 최대로 activate하는 단일 이미지가 없습니다.(?) 따라서, 좀 더 합리적인 표현을 얻기 위해서는 입력 이미지에 대한 조건(condition)이 필요합니다.

## What is Guided-Backpropagation
이 논문에서 제안하는 Guided-Backpropagation에 대한 도식은 다음과 같습니다.

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/0f84a81f431b18a78bd97f59ed4b9d8eda390970/8-Figure1-1.png" alt="">
	<figcaption>Schematic of visualizing the activations of high layer neurons</figcaption>
</figure>

--- 작성중 ---

## How to implement Guided-Backpropagation?
본 포스트에서 다루는 Guided-Backpropagation에 대한 소스코드는 [[여기]](https://github.com/you359/Keras-CNNVisualization/tree/master/keras-GuidedBackpropagation)를 참고하세요. <br />
위 소스코드는 논문에서 다루고있는 backpropagation, deconvnet, guided-backpropagation 모두 다루고 있으며, 사용의 편의성을 위해서 class로 랩핑해뒀습니다.

--- 작성중 ---

## Results
몇가지 이미지에 대해 backpropagation, deconvnet, guided-backpropagation으로 시각화한 결과는 다음 그림과 같습니다.
이 결과 이미지를 만드는 코드는 [[keras-GuidedBackpropagation]](https://github.com/you359/Keras-CNNVisualization/tree/master/keras-GuidedBackpropagation)의 'GuidedBackprop Visualization.ipynb' jupyter notebook 파일을 참고하세요.

<figure>
	<img src="https://github.com/you359/Keras-CNNVisualization/raw/master/keras-GuidedBackpropagation/result.png" alt="">
	<figcaption>Results of CAM</figcaption>
</figure>

## Source Code
{% gist you359/d19449a1c64bb43519a11e5d9d430453 %}

## References
[1] Striving for simplicity: The all convolutional net, 2014  [[paper]](https://arxiv.org/pdf/1412.6806.pdf)