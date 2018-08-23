---
layout: post
title: Tensorflow Object Detection API
description: "이 포스트에서는 Tensorflow Models에 포함된 여러 모델 중, Object Detection 기술에 대해 간략히 살펴보겠습니다."
modified: 2018-08-21
tags: [Tensorflow, Tensorflow Models, Object Detection]
category: Tensorflow Models
image:
  feature: posts/tensorflow_bg.jpg
  credit: tensorflow
  creditlink: https://www.tensorflow.org/
---

# Tensorflow Object Detection API
이번 포스트에서는 Tensorflow Models에 포함된 Object Detection 기술에 대해서 간략히 살펴보겠습니다. 원글 내용은 [여기](https://github.com/tensorflow/models/tree/master/research/object_detection)를 참고하세요.<br/>
또한, Tensorflow Models에 대한 소개는 [이전 포스트](/tensorflow models/Tensorflow-Models/)를 참고해주세요.

다음은 이번 포스트에서 소개할 Tensorflow Object Detection API의 설치 및 사용법에 관한 내용을 정리한 포스트 리스트입니다.

- **Tensorflow Object Detection API 소개**
- [Tensorflow Object Detection API 설치하기](/tensorflow%20models/Tensorflow-Object-Detection-API-Installation/)
- [Tensorflow Object Detection API를 활용한 모델 학습하기](/tensorflow%20models/Tensorflow-Object-Detection-API-Training/)


## Object Detection?
Image Classification 은 단순히 이미지가 어떤것인지 예측하는 기술이라면, Object Detection 은 다음 그림과 같이 이미지 내에 포함된 여러 개체(Object)들의 위치와 종류(class)까지 맞춰야 하는 좀 더 어려운 분야입니다.

<figure>
	<img src="https://github.com/tensorflow/models/raw/master/research/object_detection/g3doc/img/kites_detections_output.jpg" alt="">
	<figcaption>Object Detection의 예</figcaption>
</figure>

최근까지 Object Detection 분야에 대한 연구가 활발히 진행되어왔으며, 연구된 모델들은 다음과 같습니다.
- R-CNN, SPPNet, Fast R-CCNN
- Faster R-CNN with ZF, VGG, Resnet, Inception Resnet v2, MobileNet ...
- RetinaNet
- SSD(Single Shot MultiBox Detector) with Inception v3, Mobilenet, Resnet ...
- YOLO
...

이렇게 개발된 모델들은 각각의 상황에 따라 각기 다른 이점을 가지고 있습니다.

단편적인 예를 들자면, Inception Resnet v2를 feature extractor로 사용하는 Faster R-CNN은 정확도(accuracy) 측면에서 높은 성능을 보이지만, two way training/inference 구조의 한계로 속도가 느리다는 단점이 있습니다.
반면 R-FCN과 SDD는 Faster R-CNN보다는 약간 정확도가 떨어지지만 빠른 속도를 갖는 이점을 가지고 있습니다.

뿐만 아니라, 각 대회(Pascal VOC, COCO, ...)의 데이터셋에 따라서도 성능이 상이합니다.

Object Detection 기술의 비교에 대한 자세한 내용은 Jonathan Hui님이 작성한 블로그 포스트 [Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, FPN, RetinaNet and YOLOv3)](https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359)와
Google에서 발표한 [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/pdf/1611.10012.pdf)논문을 참고해주세요.

만약 자신의 분야에 Object Detection 을 적용해보고자 할 경우, 여러 모델들을 가능한 살펴보고 구현도 해봐야하는데 양적으로도, 구현 난이도적으로도 쉽지 않습니다.

다행히도, Tensorflow Models에서는 이러한 Object Detection 모델들을 한번에 구현해놓은 API를 제공합니다.
서론이 길어졌으니 바로 다음 링크를 클릭하여 Tensorflow Object Detection API를 설치하는 방법을 살펴봅시다.

[<span style="color:red">Tensorflow Object Detection API 설치하기</span>](/tensorflow%20models/Tensorflow-Object-Detection-API-Installation/)

## References
[1] Tensorflow Object Detection API - [https://github.com/tensorflow/models/tree/master/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection) <br />