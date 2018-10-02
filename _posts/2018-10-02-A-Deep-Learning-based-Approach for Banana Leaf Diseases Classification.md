---
layout: post
title: Paper Review - A Deep Learning-based Approach for Banana Leaf Diseases Classification
description: "LeNet을 이용해서 이미지 내 바나나의 질병을 분류하는 논문 리뷰"
modified: 2018-10-02
tags: [Review, Agriculture, Leaf Classification]
category: Agriculture
image:
  feature: posts/agriculture.jpg
  credit: democrats
  creditlink: http://www.democrats.eu/en/theme/agriculture
---

# Paper Review - A Deep Learning-based Approach for Banana Leaf Diseases Classification
이 포스트에서는 2017년 BTW workshop에 실린 "A Deep Learning-based Approach for Banana Leaf Diseases Classification" 논문에 대해 살펴보겠습니다.

## Key Point
- LeNet based Banana Leaf Classification

## Dataset
1. PlantVillage project의 Banana Leaf 이미지
    healty (1643), black sigatoka (240), black speckle (1817)

## Approach & Model
1. deeplearning4j 프레임워크를 사용함
2. 60x60 크기로 resizing한 후, LeNet을 적용해서 분류처리
3. 학습/테스트 데이터의 비율을 80/60/50/40/20 으로 구성해서 테스트함

## Optimization
1. SGD
    learning rate : 0.001 <br />
    momentum : 0.9 <br />
    weight decay : 0.005 <br />
    batch size : 10 <br />
    epoch : 30 <br />

## Implementation
다음 github link 참조 <br />
[[Keras-BLDC]](https://github.com/you359/Keras-Agriculture/tree/master/Keras-BLDC(Banana%20Leaf%20Diseases%20Classification))


## References
[1] Deep Neural Networks Based Recognition of Plant Diseases by Leaf Image Classification, 2016 [[paper]](https://www.hindawi.com/journals/cin/2016/3289801/abs/) <br />

