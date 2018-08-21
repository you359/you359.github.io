---
layout: post
title: Tensorflow Object Detection API를 활용한 모델 학습하기
description: "이 포스트에서는 Tensorflow Object Detection API를 활용해서 모델을 학습하는 방법에 대해 소개하겠습니다."
modified: 2018-08-21
tags: [Tensorflow, Tensorflow Models, Object Detection]
category: Tensorflow Models
image:
  feature: posts/tensorflow_bg.jpg
  credit: tensorflow
  creditlink:
---

# Tensorflow Object Detection API
이번 포스트에서는 Tensorflow Models에 포함된 Object Detection API를 활용해서 모델을 학습하는 방법에 대해 소개하겠습니다. 원글 내용은 [여기](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)를 참고하세요.

Tensorflow Object Detection API 설치 방법은 [이전 포스트](/tensorflow%20models/Tensorflow-Object-Detection-API-Installation/)를 참고해주세요.

- [Tensorflow Object Detection API 소개](/tensorflow%20models/Tensorflow-Object-Detection-API/)
- [Tensorflow Object Detection API 설치하기](/tensorflow%20models/Tensorflow-Object-Detection-API-Installation/)
- **Tensorflow Object Detection API를 활용한 모델 학습하기**

## Tensorflow Object Detection API를 활용한 모델 학습하기 [<span style="color:blue">[원문 링크]</span>](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md)
위에서 소개된 원문 링크에서는 Google Cloud환경에서 Oxford-IIIT Pets 데이터셋을 사용해서, resnet-101을 feature extractor로 사용하는 Faster R-CNN을 학습시키는 방법에 대해 소개하고 있습니다.
또한 Transfer Leraning을 위해 COCO-pretrained 모델을 사용하는 방법 또한 제시됩니다.

본 포스트에서는 위에서 소개된 원문 링크와는 달리, 로컬 컴퓨팅 환경에서 Pascal VOC 데이터셋을 이용해서 모델을 학습시키는 방법에 대해 살펴보겠습니다.

자세한 내용은 다음의 링크들을 참고하세요.
- [입력 데이터 준비(Preparing inputs)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md)
- [모델/학습/검증 파라미터 설정(Configuring an object detection pipeline)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)
- [로컬 컴퓨팅 환경에서 실행하기(Running Locally)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)
- [모델을 활용한 추론(Exporting a trained model for inference)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md)
- [모델 검증 및 추론하기(Inference and evaluation on the Open Images dataset)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/oid_inference_and_evaluation.md)

### 입력 데이터 준비하기

먼저 Object Detection 모델을 학습하기 위한 입력 데이터를 준비합니다.
Object Detection은 일반적인 Classification과 달리, 이미지내에 존재하는 개체들의 위치까지 식별해야 하는 작업이기 때문에, 학습데이터에는 개체의 위치에 해당하는 정보가 포함되어야 합니다.
Object Detection에 대한 간략한 소개는 [Tensorflow Object Detection API](/tensorflow%20models/Tensorflow-Object-Detection-API/)를 참고해주세요.

개체의 위치에 대한 정보를 함께 제공하기 위해서,
개체를 둘러싸는 Bounding Box 좌측 상단(x, y)과 너비,높이(width, height) 혹은 좌측 상단(xmin, ymin)과 우측 하단(xmax, ymax)의 pixel위치 값을 해당 개체의 클래스와 함께
개체별로 Labeling해서 (annotation)파일로 작성합니다.

labeling된 annotation은 csv파일로 작성하거나, xml파일로 작성할 수 있습니다.

이번 포스트에서 학습 데이터로 사용할 Pascal VOC 데이터셋은 다음과 같은 xml파일로 annotation되어 있습니다.
```
<annotation>
	<folder>VOC2007</folder>
	<filename>000001.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>341012865</flickrid>
	</source>
	<owner>
		<flickrid>Fried Camels</flickrid>
		<name>Jinky the Fruit Bat</name>
	</owner>
	<size>
		<width>353</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>dog</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>48</xmin>
			<ymin>240</ymin>
			<xmax>195</xmax>
			<ymax>371</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>8</xmin>
			<ymin>12</ymin>
			<xmax>352</xmax>
			<ymax>498</ymax>
		</bndbox>
	</object>
</annotation>
```

Object Detection을 위한 데이터로 Pascal VOC 2012 데이터셋을 다운로드합니다.
```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```

Tensorflow Object Detection API에서는 모든 입력 데이터를 TFRecords로 변환해서 사용합니다. 따라서 다운로드받은 Pascal VOC 2012데이터셋을 TFRecords 파일로 변환해야 합니다. <br/>
다음의 create_pascal_tf_record.py Python 스크립트를 사용해서 Pascal VOC 2012 데이터를 TFRecords로 변환합시다.

```shell
# From tensorflow/models/research/
python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record

python object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record
```

- label_map_path : 개체의 클래스 label을 맵핑해놓은 protobuf 파일의 경로(path)
- data_dir : Pascal VOC 2012 데이터셋의 경로(path)
- year : Pascal VOC의 경우 대표적으로 2007과 2012가 있으며, 이번 포스트에서는 2012를 사용하므로 VOC2012를 사용
- set : 학습(train) / 검증(val) / 테스트(test) 셋 구분
- output_path : 출력 파일 경로(임의)

앞서 모든 커멘드에 tensorflow/models/research/ 폴더에서 실행하라고 표현되어 있지만, 당연히 별도의 로컬 폴더를 생성하고, 다음과 같이 path를 지정해서 스크립트를 실행해도 괜찮습니다.
```shell
path/to/tensorflow/models/research/xxx.py
```

Pascal VOC 데이터셋이 아닌 COCO나 여타 다른 데이터셋의 형식으로 annotation되어 있을 경우, [object_detection/dataset_tools폴더](https://github.com/tensorflow/models/tree/master/research/object_detection/dataset_tools) 에 포함된 create_xxxx_tf_record.py 파이썬 스크립트 파일을 사용합니다.
- COCO dataset : create_coco_tf_record.py
- kitti dataset : create_kitti_tf_record.py
- oid dataset : create_oid_tf_record.py
- pascal : create_pascal_tf_record.py
- pet dataset : create_pet_tf_record.py
- ...

label_map 구성 또한 각 데이터셋별로 다르게 구성되어 있으며, [object_detection/data폴더](https://github.com/tensorflow/models/tree/master/research/object_detection/data) 에서 확인할 수 있습니다. pascal voc의 pascal_label_map.pbtxt은 다음과 같이 표현됩니다.

```
item {
  id: 1
  name: 'aeroplane'
}

item {
  id: 2
  name: 'bicycle'
}

item {
  id: 3
  name: 'bird'
}
...
```

### 모델/학습/검증을 파라미터 구성하기
대부분의 Object Detection 모델들은 여러가지 데이터셋을 효과적으로 학습하거나, 식별하기 위한 모델 파라미터가 존재합니다.

예를들어 Faster R-CNN의 경우, 후보 영역을 제안하기 위한 RPN을 학습시키기 위해, anchor라는 개념이 도입되어 있는데, 이 anchor의 크기나 비율(aspect ratio)를 설정해주어야 합니다.
Tensorflow Object Detection API에서는 이러한 파라미터를 설정하는 것을 pipeline configuration이라 부르며, pipeline 또한 protobuf 파일로 구성합니다.

pipeline config 파일은 다음과 같이 5개의 part로 구분되어 있습니다.
- model
    * 사용하고자 하는 meta-architecture(Faster R-CNN? SSD?)와 feature extractor(resnet? inception?)
    * class의 개수
    * 입력 이미지의 최소/최대 크기
    * anchor의 크기(scales)와 비율(aspect_ratios) 등
- train_config
    * batch_size
    * 사용하고자 하는 optimizer 및 learning rate scheduler 등
- eval_config
    * 검증을 위해 사용하고자 하는 metrics (pascal, coco, etc..)
- train_input_config
    * 학습시키고자 하는 입력 데이터 TFRecord 파일(앞서 준비한 학습 데이터 경로 - pascal_train.record)
    * label_map 정보
- eval_input_config
    * 검증하고자 하는 입력 데이터 TFRecord 파일(앞서 준비한 검증 데이터 경로 - pascal_val.record)
    * label_map 정보

자세한 내용은 [<span style="color:blue">[여기]</span>](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)를 참고합시다.

pipeline configuration 대한 예제 파일은 [object_detection/samples/configs 폴더](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)에서 살펴볼 수 있습니다.

### 모델 학습하기
지금까지 준비한 입력 데이터(TFRecord)와 configuration pipeline (xxx.config)들을 이용해서 모델을 학습해봅시다.
먼저 준비한 사항들은 다음과 같은 폴더/파일 구조로 정리합시다.

```
 .
├── data(folder)
│   └── label_map file
│   └── train TFRecord file
│   └── eval TFRecord file
├── models(folder)
│   └── pipeline config file
│   └── train(folder)
│   └── eval(folder
 .
```

Object Detection 모델을 학습시키는 python 스크립트는 object_detection/model_main.py 파일입니다.

다음의 shell 커멘드를 사용해서 학습을 진행해봅시다.

```shell
# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH={path/to/models/pipeline config file} # 앞서 구성한 pipeline configuration 파일 경로
MODEL_DIR={path/to/train} # 학습된 모델을 저장할 파일 경로
NUM_TRAIN_STEPS=50000 # Training Step
NUM_EVAL_STEPS=2000 # Eval Step
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
```

<span style="color:red">여기서 NUM_TRAIN_STEP과 NUM_EVAL_STEP은 명확하게 확인이 안되어, 확인되는데로 다시 정리해두겠습니다.</span>

앞의 커멘드를 입력하면 구성한 pipeline config파일을 참조해서 자동으로 Object Detection 모델을 학습하게됩니다.

학습의 경과는 TensorBoard를 이용해서 확인할 수 있습니다. 다음의 커멘드를 사용해서 경과를 확인합니다.

```shell
tensorboard --logdir=${MODEL_DIR} # path/to/train 위에서 지정한 모델 저장 경로
```

### 추가 사항
Faster R-CNN의 경우 4개 step으로 구성된 학습 단계가 있는데, 이 때, 1~2 step에서는 pre-trained model을 이용해서 RPN과 detection network를 각각 학습시킵니다.
만약 이 때, pre-trained model이 없으면, RPN이 재대로 학습되지 않고, RPN이 재대로 학습되지 않으면 이후 지역 제안 후보 영역(region proposal)을 재대로 생성하지 못해서 detection network로 학습이 안되는 문제가 발생합니다.

따라서, 본인의 domain에 맞는 feature extractor network를 미리 학습시키거나, imagenet과 같은 대규모 데이터셋으로 학습된 모델을 pre-trained model로 준비하여 적용하는 것이 좋습니다.
pre-trained model의 적용은 pipeline configuration 에서 train_config 항목의 fine_tune_checkpoint에 pre-trained model의 ckpt 파일 경로를 추가해주면 됩니다.

다음의 파일 내용을 참고해봅시다.

/object_detection/protos/train.proto

```protos
  // Checkpoint to restore variables from. Typically used to load feature
  // extractor variables trained outside of object detection.
  optional string fine_tune_checkpoint = 7 [default=""];

  // Type of checkpoint to restore variables from, e.g. 'classification' or
  // 'detection'. Provides extensibility to from_detection_checkpoint.
  // Typically used to load feature extractor variables from trained models.
  optional string fine_tune_checkpoint_type = 22 [default=""];
```

train.proto 파일의 내용에 따르면,

만약 Faster R-CNN 자체를 학습시킨 모델을 pre-trained model로 사용하고자 한다면, fine_tuen_checkpoint에는 해당 모델의 경로를,
fine_tune_checkpoint_type에는 "detection" 이라는 type을 명시해줍니다.

만약 feature extractor를 pre-trained model로 사용하고자 한다면, fine_tuen_checkpoint에는 해당 모델의 경로를,
fine_tune_checkpoint_type에는 "classification" 이라는 type을 명시해줍니다.

여기까지 Tensorflow Object Detection API를 활용해서 모델을 학습하는 방법에 대해 살펴보았습니다.

보통 모델 학습에 걸리는 시간은 컴퓨팅환경에 따라 다릅니다.

<!-- 컴퓨팅 파워에 따른 학습 속도 작성 -->
<!-- 이제 다음 링크를 클릭하여 학습시킨 모델을 이용해서 입력 데이터를 추론하고, 검증/테스트 데이터를 이용해서 모델의 성능을 평가하는 방법에 대해 살펴봅시다. -->

<!-- [<span style="color:red">Tensorflow Object Detection API을 활용한 모델 학습하기</span>]() -->

## References
[1] Tensorflow Object Detection API - [https://github.com/tensorflow/models/tree/master/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection) <br />