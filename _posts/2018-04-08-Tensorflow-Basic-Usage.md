---
layout: post
title: Tensorflow 기본 활용
description: "in this post, i will write the research strategy for Fake(forgery) image detection"
modified: 2018-03-26
tags: [Tensorflow, Machine Learning]
category: Fake Detection
image:
  feature: posts/tensorflow_bg.jpg
  credit: tensorflow
  creditlink:
---

# 텐서플로우 기본 활용
MNIST, CIFAR 등의 가공된 데이터셋이 아닌, 직접 수집한 데이터를 가공 및 처리해서 Classification 하는 기본적인 활용 방법

github 참조 - https://github.com/you359/Tensorflow-Classification

### 기본 활용 방법 구성
- 수집된 데이터 가공
    + image파일, csv파일(path, label)
- Data Augmentation
- Reading Data
    + Feed Mechanism
    + Input Pipeline
    + PreLoaded Data
- PreTrained Model 활용
    + .npy 파일로 pretrained model 활용하기
    + caffemodel to .npy
- Classification

### 수집된 데이터 가공
일반적으로 Machine Learning/Deep Learning을 적용해서 이미지를 분류하기 전에, 수집한 데이터를 Read할 포맷에 맞게 가공할 필요성이 있다.
이미지 파일 분류의 경우, 보통 이미지 파일이 포함된 폴더와 해당 이미지 파일과 레이블이 표기된 csv 파일로 구성하여 수집한 데이터를 가공한다.
가공된 데이터의 구성은 다음과 같다.

```
 .
├── train(folder)
│   └── XXX-1.jpeg
│   └── XXX-2.jpeg
│   └── XXX-3.jpeg
│   └── ...
├── validation(folder)
│   └── XXX-1.jpeg
│   └── XXX-2.jpeg
│   └── XXX-3.jpeg
│   └── ...
├── test(folder)
│   └── XXX-1.jpeg
│   └── XXX-2.jpeg
│   └── XXX-3.jpeg
│   └── ...
├── train_labels.csv
├── validation_labels.csv
├── test_labels.csv
```

각 이미지 파일은 train, validation, test 구분에 따라 각각 해당 폴더에 저장한다.
(다른 방법으로는, 각 구분 아래에 이미지의 레이블별로 폴더를 생성하거, 레이블별로 각각 저장하기도 함)

이후, 이미지의 저장 위치(path)와 해당 이미지의 레이블을 가리키는 csv 파일을 생성하여 저장한다.
csv 파일은 마찬가지로 train, validation, test로 구분하여 저장한다.
각 csv 파일에는 0번째 row 에는 이미지의 위치경로(path)가 표시되고, 1번째 row 에는 이미지의 레이블이 표시된다.
csv 파일 구성의 예는 다음과 같다.

```
./train/pic_01.jpeg, 0
./train/pic_02.jpeg, 1
.
.
```

### Data Augmentation
Data Augmentation 이란, 이름 해석에서도 알 수 있듯이 데이터를 확장시키는 것을 의미한다.
일반적으로 Machine Learning 이나 Deep Learning 에서는 데이터 분류를 위해 모델을 학습시킬 때, 매우 많은 양의 데이터가 필요하다. (만약, 데이터가 적으면 과적합(overfitting) 문제에 직면할 수 있다.)
그러나 대게 "데이터 수집"은 상당히 많은 비용이 드는 작업이며, 경우에 따라 데이터가 매우 부족한 경우도 있다. 이러한 문제를 해결하기 위한 기술이 Data Augmentation 이다.
Data Augmentation 은 적은 양의 데이터를 바탕으로 여러 조작을 통해 데이터의 양을 늘리는 작업으로, 대표적인 Augmentation 방법은 다음과 같다.
- flip
- crop
- rotate
- shift

Data Augmentation 을 적용하기 위해, imgaug 라는 라이브러리를 활용해보자.
해당 내용은 Data sugmentation.ipynb를 참고하자.

### Reading Data
직접 수집, 가공한 데이터를 Tensorflow의 Tensor로 읽어들여야 모델의 학습이 가능하다. (당연)
가장 기본적으로 사용해온 Data Reading 방법은 Feed Mechanism 이며 다음과 같이 수행한다.
1. 먼저 python 코드로 PIL이나 opencv 등의 라이브러리를 통해 이미지를 읽어들인다.
2. 읽어들인 이미지와 레이블을 Dictionaly로 구성한다.
3. Tensorflow 모델의 placeholder로 feed_dict를 통해 전달한다.

Feed Mechanism 방식은 적용이 간단하다는 장점이 있지만, 데이터가 상당히 많을 경우 CPU Memory 나 RAM 에 부하를 줄 수 있다는 단점이 있다.
Tensorflow Tutorial 에서는 데이터를 Reading 하는 방법으로 Input pipeline 과 preloaded data 를 소개하고 있다.
input pipeline 은 Queue와 Thread를 활용해서, 모델에서 데이터가 필요할 때마다 Queue에서 데이터를 읽어오는 방식으로, 메모리를 적게 사용하는 장점이 있다.
preloaded data 는 CPU 나 RAM에 데이터를 저장하는 것이 아닌, GPU Memory 에 데이터를 Load 해놓고 쓰는 방식으로, 데이터 접근 속도가 빠르다는 장점이 있다.

### PreTrained Model 활용
지난 3~5년간 연구에서는 딥러닝 모델의 초기화 방법이 학습에 큰 영향을 미친다는 결과를 내보이고 있다. 따라서 학습 속도와 정확도를 향상시키기 위해서 절적한 모델 초기화 방법을 사용할 필요가 있다.
이미지 분류 도메인에서는 ImageNet이나 Pascal, COCO 등의 프로젝트에서 제공한 데이터를 이용한 많은 연구가 이루어지고 있다. 따라서 해당 데이터를 사용해서 학습된 pretrained model도 많이 제공되고 있는 편이다.
이러한 pretrained model을 활용하면 구성된 모델의 학습 속도와 정확도를 크게 증가시킬 수 있다.

pretrained model의 파라미터는 보통 .npy 파일에 저장되어 있으며, 이를 로드해서 모델의 파라미터에 적용하면 된다.

caffe 프레임워크의 caffemodel 을 pretrained model(.npy)과 구성 model(.py)로 변환하는 방법에 대해서도 소개한다.

[4] [Deep Learning Model Convertors](https://github.com/ysh329/deep-learning-model-convertor) - 여러 딥러닝 프레임워크간에 모델을 변환할 수 있는 Convertor에 대해 소개해놓은 git

### Classification
이제 수집, 가공된 데이터를 Reading Data를 통해 읽어들이고 Augmentation을 수행한 다음, 적절한 pretrained model을 적용해서 학습하자.
해당 내용은 Classification폴더의 ipython notebook을 참고하자.

## Reference
[1] [imgaug - image Augmentation](https://github.com/aleju/imgaug) <br />
[2] [Tensorflow Document - Reading Data](https://www.tensorflow.org/api_guides/python/reading_data) <br />
[3] [Caffe to Tensorflow - PreTrained Model](https://github.com/ethereon/caffe-tensorflow) <br />
[4] [Deep Learning Model Convertor](https://github.com/ysh329/deep-learning-model-convertor) <br />