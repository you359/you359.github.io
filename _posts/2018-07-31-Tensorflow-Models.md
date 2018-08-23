---
layout: post
title: Tensorflow Models
description: "이 포스트에서는 tensorflow github에 포함된 Tensorflow Models에 대해서 소개하겠습니다."
modified: 2018-08-21
tags: [Tensorflow, Tensorflow Models]
category: Tensorflow Models
image:
  feature: posts/tensorflow_bg.jpg
  credit: tensorflow
  creditlink: https://www.tensorflow.org/
---

# Tensorflow Models
이번 포스트에서는 Tensorflow github repo에 포함된 Tensorflow Models에 대해서 소개해보겠습니다.<br/>
원글 내용은 [<span style="color:blue">여기</span>](https://github.com/tensorflow/models)를 참고하세요.<br/>
Tensorflow Models github 페이지에는 다음과 같이 4가지 폴더로 구성되어 있습니다.<br/>

- [official models](https://github.com/tensorflow/models/blob/master/official)

    official models에는 Tensorflow의 [high-level API](/tensorflow basic/Tensorflow-High-Level-API/)를 이용한 example 모델들이 포함되어 있습니다.
    Tensorflow 그룹에서는 처음 Tensorflow를 사용하는 사용자들에게 이 예제부터 시작하라고 권장하고 있습니다.

- [research models](https://github.com/tensorflow/models/tree/master/research)

    research models에는 연구자들에의해 구현된 여러가지 모델들이 포함되어 있습니다.

    특히 이 폴더에는 CNN, RNN 등 뿐만 아니라, AutoEncorder, Object Detection 등 인공지능 분야에서 연구되었던 다양한 딥 러닝 모델들이 Tensorflow로 구현되어 있습니다.

    다음의 블로그 포스트에서는 Tensorflow Models/research에 포함된 모델들을 사용하는 방법들에 대해서 소개하겠습니다.
    + [Object Detection](/tensorflow models/Tensorflow-Object-Detection-API/) [<span style="color:blue">[원문 링크]</span>](https://github.com/tensorflow/models/tree/master/research/object_detection)

- [samples folder](https://github.com/tensorflow/models/blob/master/samples)

    samples folder에는 다양한 블로그 포스트에서 소개된 코드들이 있고, Tensorflow의 기능을 잘 소개하는 code snippet과 smaller model들이 포함되어 있습니다.

- [tutorials folder](https://github.com/tensorflow/models/blob/master/tutorials)

    tutorial folder에는 [Tensorflow tutorials](https://www.tensorflow.org/tutorials/)에 소개된 모델들이 포함되어 있습니다.


## References
[1] Tensorflow Models github repo - [https://github.com/tensorflow/models](https://github.com/tensorflow/models) <br />