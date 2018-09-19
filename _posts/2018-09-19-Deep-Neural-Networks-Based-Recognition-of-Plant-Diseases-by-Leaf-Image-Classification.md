---
layout: post
title: Paper Review - Deep Neural Networks Based Recognition of Plant Diseases by Leaf Image Classification
description: "이미지 내 식물 질병 분류를 위한 데이터 셋 생성과 분류 모델을 CaffeNet으로 구성한 논문 리뷰"
modified: 2018-09-19
tags: [Review, Agriculture, Leaf Classification]
category: Agriculture
image:
  feature: posts/agriculture.jpg
  credit: democrats
  creditlink: http://www.democrats.eu/en/theme/agriculture
---

# Paper Review - Deep Neural Networks Based Recognition of Plant Diseases by Leaf Image Classification
이 포스트에서는 2016년 Computational intelligence and neuroscience 에 실린 "Deep Neural Networks Based Recognition of Plant Diseases by Leaf Image Classification" 논문에 대해 살펴보겠습니다.

## Key Point
- plant disease recognition model, based on leaf image classification, by the use of deep convolutional networks

## Dataset
1. 식물과 질병의 이름을 이용해서 인터넷에서 이미지를 검색하여 수집함
    - 식물과 질병의 이름은 여러 다른 언어(라틴어, 영어, 독일어 등등)를 바꾸어가며 선택해서 검색
    - 이미지 수집 시 500px 미만의 해상도를 가진 이미지는 고려하지 않음
2. 15개의 클래스로 구성하고, 13개의 클래스는 식물의 질병을 나타냄
    - 건강한(질병이 없는) 식물의 잎을 구분하기 위해 하나의 클래스를 추가로 구성함
    - 배경 이미지를 엑스트라 클래스로 추가함
        - 배경 이미지는 Standford Background dataset에서 수집함
3. 이미지의 metadata(이름, 크기, 날짜 등)를 비교해서 중복된 이미지는 제거함
    - 이후 human experts에 의해 다시 검증함
4. augmentation을 수행하고, 30880개의 학습 데이터, 2589개의 검증 데이터로 구분함

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/27deff0dd3d9ca4dcb73c6f8f76d2606beb3e8ba/4-Table1-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

5. Image Preprocessing and Labelling
    - 식물의 잎부분만 보이도록 이미지를 cropping
    - 256x256 크기로 resize
    - 먼저 검색한 keyword에 따라 레이블링 한 후, human expert가 다시 레이블링함

6. Augmentation Process
    - affine transformation
    - perspective transformation
    - simple image rotatations

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/27deff0dd3d9ca4dcb73c6f8f76d2606beb3e8ba/5-Figure2-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

## Approach & Model
1. Caffe 프레임워크의 CaffeNet을 사용함
    - 15개 클래스 이미지를 분류하기 위해, 기존의 1000개 output을 갖는 마지막 fc layer를 제거하고, 15개 output을 갖는 fc layer를 연결해서 모델을 재구성함

## Optimization
1. imagenet으로 학습된 pre-trained 모델을 사용함
2. Base Network 부분인 CaffeNet의 learning rate는 0.1, 새로 연결한 마지막 fc layer의 learning rate는 10으로 설정해서 finetuning함
3. 10-fold cross validation을 적용해서 validation과 evaluation을 수행함

## My Review
1. 식물 잎의 질병 검출을 위한 evaluation 을 cross-validation 을 적용해서 수행했기 때문에, 성능이 좋게 나오는 것으로 보임
    - 별도의 test dataset 을 구분해서 evaluation을 수행해야 테스트 결과가 명확할 것으로 생각됨
2. 데이터셋을 논문에서 직접 구성하고, human labeling을 수행하기 때문에 신뢰성이 떨어짐
3. 기존의 타 연구와의 비교가 없음


<!-- ## How to implement? -->
<!-- // 향후 구현 예정 // -->

<!-- ## Results -->
<!-- // 향후 구현 예정 // -->

## References
[1] Deep Neural Networks Based Recognition of Plant Diseases by Leaf Image Classification, 2016 [[paper]](https://www.hindawi.com/journals/cin/2016/3289801/abs/) <br />

