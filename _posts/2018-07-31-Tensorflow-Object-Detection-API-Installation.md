---
layout: post
title: Tensorflow Object Detection API 설치하기
description: "이 포스트에서는 Tensorflow Object Detection API를 설치하는 방법에 대해 소개하겠습니다."
modified: 2018-07-31
tags: [Tensorflow, Tensorflow Models, Object Detection]
category: Tensorflow Models
image:
  feature: posts/tensorflow_bg.jpg
  credit: tensorflow
  creditlink:
---

# Tensorflow Object Detection API
이번 포스트에서는 Tensorflow Models에 포함된 Object Detection API를 설치하는 방법에 대해 소개하겠습니다. 원문 링크는 [여기](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)를 참고하세요.

Object Detection에 대한 간략한 소개는 [이전 포스트](http://localhost:4000/tensorflow%20models/Tensorflow-Object-Detection-API/)를 참고해주세요.

- [Tensorflow Object Detection API 소개](http://localhost:4000/tensorflow%20models/Tensorflow-Object-Detection-API/)
- **Tensorflow Object Detection API 설치하기**
- [Tensorflow Object Detection API를 활용한 모델 학습하기](/tensorflow%20models/Tensorflow-Object-Detection-API-Training/)

## Tensorflow Object Detection API 설치하기 [<span style="color:blue">[원문 링크]</span>](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
#### 의존성라이브러리 설치
Tensorflow의 Object Detection API를 활용하기 위해서는 먼저 다음의 library들을 설치해야 합니다.

- Protobuf 3.0.0
- Python-tk
- Pillow 1.0
- lxml
- tf Slim (which is included in the "tensorflow/models/research/" checkout)
- Jupyter notebook
- Matplotlib
- Tensorflow
- Cython
- contextlib2
- cocoapi

만약 anaconda를 이용해서 python을 설치하고, Tensorflow까지 설치하셨다면, 대부분은 이미 설치되어있습니다.
그러나 만약 설치가 안되있다면 다음의 pip 및 apt-get 명령어를 통해 설치합니다.

```shell
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu

sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib

pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
```

#### tensorflow/models 다운로드
Tensorflow Object Detection API를 사용하기위해, 먼저 github repo인 tensorflow/models를 클론(다운로드)합니다.

```shell
git clone https://github.com/tensorflow/models
```

#### COCO API 설치
다음으로 [cocoapi](https://github.com/cocodataset/cocoapi)를 다운로드받고, pycocotools라는 폴더를 tensorflow/model/research 폴더에 복사합니다.

```shell
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
```

pycocotools는 Object Detection 모델을 evaluation 할 때 사용하는 evaluation metrics로 사용됩니다. 이후 COCO evaluation metrics를 사용하지 않더라도,
Tensorflow Object Detection API는 내부적으로 COCO evaluation metrics를 기본으로 확인하기 때문에 필수적으로 설치하셔야합니다.

(원문에서는 coco evaluation metrics가 흥미로우면 설치하라고 되어있고, default로 Pascal VOC evaluation metric를 사용한다고 되어있는데, 제가 실제로 해본 결과 기본적으로 설치가 필요하고, default도 Pascal VOC가 아닌것같습니다. 나중에 정확한 확인이 되면 수정하겠습니다.)

#### Protobuf 컴파일
Tensorflow Object Detection API는 Object Detection 기술들의 설정이나 학습에 적용할 파라미터등을 Protobuf를 사용해서 설정합니다. 따라서 API를 사용하기위해서는 먼저 Protobuf 라이브러리를 컴파일해야합니다.
Protobuf 라이브러리 컴파일은 다음의 명령어를 사용해서 수행할 수 있습니다.

```shell
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

#### PYTHONPATH에 tensorflow/model/research와 slim 폴더 경로 추가
본인의 로컬 컴퓨터에서 API를 활용하기 위해서는, tensorflow/models/research/ 와 slim 폴더가 PYTHONPATH에 등록되어있어야 합니다.
다음의 명령어를 통해 PATH를 등록합니다.

```shell
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

이 명령어는 터미널을 열 때마다 실행해줘야 합니다. 자동으로 PATH 등록이 되도록 하고 싶다면 ~/.bashrc 파일에 PATH를 추가하세요.

#### 설치 확인
앞선 과정을 제대로 수행했다면, 다음의 python 파일을 실행해서 설치 결과를 확인할 수 있습니다.

```shell
python object_detection/builders/model_builder_test.py
```

#### Python 3.x 버전 오류
Tensorflow Object Detection API를 직접 사용해본 결과,
Python 3.x 버전에서는 꾀 많은 오류가 발생합니다. 향후 컴퓨팅 환경을 리셋한 후, Python 3.x 버전에서 발생하는 오류들을 정리해보도록 하겠습니다.

이러한 오류는 tensorflow/models github repo의 issues에서 쉽게 찾아보실 수 있습니다.

설치가 완료되었다면, 이제 다음 링크를 클릭하여 Tensorflow Object Detection API에서 제공하는 Object Detection 모델을 학습하는 방법에 대해 살펴봅시다.

[<span style="color:red">Tensorflow Object Detection API을 활용한 모델 학습하기</span>](/tensorflow%20models/Tensorflow-Object-Detection-API-Training/)

## References
[1] Tensorflow Object Detection API - [https://github.com/tensorflow/models/tree/master/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection) <br />