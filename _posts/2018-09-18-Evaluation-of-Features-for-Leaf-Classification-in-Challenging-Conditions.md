---
layout: post
title: Paper Review - Evaluation of Features for Leaf Classification in Challenging Conditions
description: "식물의 잎을 ConvNet과 HCFs(Hand Crafted Features), Random Forest를 적용해서 분류하고, 다양한 조건(translation, rotation 등등)에서 실험한 논문 리뷰"
modified: 2018-09-18
tags: [Review, Agriculture, Leaf Classification]
category: Agriculture
image:
  feature: posts/agriculture.jpg
  credit: democrats
  creditlink: http://www.democrats.eu/en/theme/agriculture
---

# Paper Review - Evaluation of Features for Leaf Classification in Challenging Conditions
이 포스트에서는 2015년 IEEE Winter Conference on Applications of Computer Vision 에 실린 "Evaluation of Features for Leaf Classification in Challenging Conditions" 논문에 대해 살펴보겠습니다.

## Key Point
- examine the robustness of a range of features to a variety of condition variations including: translation, scaling, rotation, occlusion and shadowing
- combined ConvNet features and HCFs

## Dataset
1. Flavia dataset[2]
    - 1,907개의 잎사귀 이미지가 포함되어 있음
    - 32개의 식물 종으로 구성되며, 각각의 식물 종당 약 50개의 이미지가 포함됨

## Approach & Model
1. Base Network로 CaffeNet을 사용해서 ConvNet feature로 사용함
2. 기존에 사용된 Hand-Crafted shape and statistical features 중, scale 변화에 강인한 특징들을 골라서 ScaleRobust HCFs로 구성함

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3a5d979ebbceedbc8c0c621b437150fef6240bf4/4-Table1-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

3. ConvNet feature와 ScaleRobust HCFs를 Random Forest Classifier를 이용해서 분류처리함

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3a5d979ebbceedbc8c0c621b437150fef6240bf4/2-Figure1-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

## Experiments
1. 기존의 특징들(HCF, HoCS, 등)을 이용한 분류 방법과, ConvNet + ScaleRobust HCFs 와의 비교 분석을 수행함
    - 이 때, 모델들의 강인함(Robustness) 정도를 측정하기 위해서, 테스트 데이터에 대해 다양한 조건(translation, scaling, rotation, occlustion 등등)들을 설정해서 실험함

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3a5d979ebbceedbc8c0c621b437150fef6240bf4/5-Table2-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3a5d979ebbceedbc8c0c621b437150fef6240bf4/6-Figure4-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3a5d979ebbceedbc8c0c621b437150fef6240bf4/6-Figure5-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3a5d979ebbceedbc8c0c621b437150fef6240bf4/7-Figure6-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3a5d979ebbceedbc8c0c621b437150fef6240bf4/7-Figure7-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

<figure>
	<img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/3a5d979ebbceedbc8c0c621b437150fef6240bf4/8-Figure8-1.png" alt="">
	<figcaption>출처: www.semanticscholar.org</figcaption>
</figure>

<!-- 1) 논문에서는 ConvNet features와 ScaleRobust HCFs(Hand Crafted Features)를 Random Forest Classifier를 적용해서 식물 잎의 종을 분류했으며,
        기존 HCFs만 이용해서 얻은 결과인 91.2%보다 5.7%향상된 97.3%의 정확도를 보임   -->
<!-- 2) 모델들의 강인함(Robustness) 정도를 측정하기 위해 다양한 조건(translation, scaling, rotation, occlustion 등등)들에서 실험함 -->


## References
[1] Evaluation of Features for Leaf Classification in Challenging Conditions, 2015 [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7045965) <br />
[2] Flavia leaf Dataset [[site]](http://flavia.sourceforge.net/) <br />
