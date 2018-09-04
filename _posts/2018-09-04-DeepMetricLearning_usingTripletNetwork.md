---
layout: post
title: Paper Review - DEEP METRIC LEARNING USING TRIPLET NETWORK
description: "Triplet Network 을 이용한 classification 성능 비교(4개 데이터 셋) 논문 리뷰"
modified: 2018-09-04
tags: [Review, Meta Learning, Triplet Network]
category: Meta Learning
image:
  feature: posts/tripletnetwork.jpg
  credit:
  creditlink:
---

# Paper Review - DEEP METRIC LEARNING USING TRIPLET NETWORK
이 포스트에서는 2015년 CVPR에 실린 "Deep metric leraning using Triplet network" 논문에 대해 살펴보겠습니다.

## Key Point
- introduce triplet network
- apply triplet network to 4 dataset(Cifar10, MNIST, SVHN, STL10)
- 2D visualization of features(embedding)
- comparison with performance of the siamese network
- propose future work on unsupervised learning

## introduce triplet network
이 논문에서는 이전에 다뤘던 "Learning Fine-grained Image Similarity with Deep Ranking" 논문에서
사용했던 triplet network 를 4가지 데이터셋에 적용해서 classification 성능을 확인하고, 기존의 Siamese Network와 비교하고 있습니다.

먼저 Triplet Network가 무엇이였는지 다시한번 상기해봅시다. 자세한 내용은 [[Paper Review - Deep Ranking]](/meta%20learning/DeepRanking/) 포스트를 참고하세요.
Triplet은 기준이 되는 데이터인 anchor example(query image[1]), anchor example와 유사한(ex, 동일 카테고리)데이터인 positive example,
그리고 유사하지 않은(ex, 다른 카테고리) 데이터인 negative example의 3개 데이터로 구성됩니다.

Triplet의 두 쌍(anchor-positive), (anchor-negative)간의 대해 유사성을 비교하는 함수는 다음과 같이 정의됩니다.

$$
D(f(p_i), f(p_i^+)) < D(f(p_i), f(p_i^-)) \\
\forall p_i,p_i^+, p_i^- \enspace such \enspace that \enspace r(p_i, p_i^+) > r(p_i, p_i^-)
$$

여기서 $$ D(f(p_i), f(p_i^+)) $$ 는 anchor example과 positive example간의 유사성을 의미하며, 다음 수식과 같이 L2 distance로 정의할 수 있습니다.

$$
D(f(p_i), f(p_i^+)) = ||f(p_i)-f(p_i^+)||_2
$$

두 쌍의 유사성을 비교하기 위해서 Triplet은 다음과 같이 2가지의 L2 distance들을 출력해야 합니다.

$$
TripleNet(x, x^-, x^+) = [\frac{||Net(x)-Net(x^-||_2}{||Net(x)-Net(x^+||_2}]
$$

이렇게 출력된 2쌍의 L2 distance에 대한 loss function을 사용해서 TripleNet Network를 학습시킵니다.
이 논문에서는 다음과 같이, 간단한 MSE를 loss 함수로 사용했습니다.

$$
Loss(d_+, d_-) = ||(d_+, d_- -1)||^2_2
$$

이 논문에서는 위의 metric으로 학습한 4개 layer로 구성된 Triplet Network를 각각 4개의 데이터셋(Cifar10, MNIST, SVHN, STL10)으로 학습한 후,
Siamese Network와 다른 Classification Network들과 비교 분석을 수행합니다.
<figure>
	<img src="/images/contents/tripletnetwork_t1.png" alt="">
	<figcaption>Triplet Network와 Siamese Network 비교 </figcaption>
</figure>

## Future work
이 논문에서는 다음과 같이 Triplet Network의 접근 방식으로 unsupervised learning 문제를 해결할 수 있는 방법에 대한 향후 연구 방향을 제시했습니다.
- Using spatial information (공간적 정보 사용)
    - 동일 이미지 내에서 추출한 patch와, 다른 이미지에서 추출한 patch를 triplet으로 구성해서 unsupervised setting으로 모델 학습
- Using temporal information (시간적 정보 사용)
    - 동영상과 같은 시간정보가 포함된 데이터에서, 특정 시간 범위 내의 샘플과 범위 밖의 샘플을 triplet으로 구성해서 unsupervised setting으로 모델 학습

## How to implement TRIPLET NETWORK?
// 향후 구현 예정 //

## Results
// 향후 구현 예정 //

## References
[1] DEEP METRIC LEARNING USING TRIPLET NETWORK, 2015 [[paper]](https://arxiv.org/pdf/1412.6622.pdf) <br/>