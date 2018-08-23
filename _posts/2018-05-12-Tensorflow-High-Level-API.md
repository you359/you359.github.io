---
layout: post
title: Tensorflow High Level API - Estimator
description: "in this post, i will write about Estimator that is High Level API in Tensorflow"
modified: 2018-04-26
tags: [Estimators, Tensorflow, Machine Learning]
category: Tensorflow Practice
image:
  feature: posts/tensorflow_bg.jpg
  credit: tensorflow
  creditlink: https://www.tensorflow.org/
---

# Estimators
이번 포스트에서는 텐서플로우의 High Level API인 Estimator에 대해 알아보겠습니다. 이 포스트는 Tensorflow Document-[https://www.tensorflow.org/programmers_guide/estimators](https://www.tensorflow.org/programmers_guide/estimators)를 참조해서 작성했습니다.

Estimator는 다음과 같이 4가지 액션으로 캡슐화되어 있습니다.
- training
- evaluation
- prediction
- export for serving

## Advantages of Estimators
- Estimator 기반으로 모델을 구현하면, 모델의 변경 없이 CPU, GPU, TPU 등 다른 환경의 컴퓨팅 환경에서 실행할 수 있습니다.
- Estimators는 모델 개발자들간에 쉽게 개발결과를 공유할 수 있도록 도와줍니다.
- low-level TF API 보다 개발이 쉽습니다.
- Estimators는 tf.layers로 구현되어, 쉽게 커스터마이징이 가능합니다.
- Estimators는 자동으로 graph를 빌드하기 때문에, graph 빌드를 고려하지 않아도 됩니다.
- Estimators는 안전한 training loop를 제공합니다.
    + build the graph
    + initialize variables
    + start queues
    + handle exceptions
    + create checkpoint files and recover from failures
    + save summaries for TensorBoard

Estimators로 모델을 개발할 경우, 반드시 input pipeline과 model을 나누어서 구현해야 합니다.

## Pre-made Estimators
Estimators로 모델을 구현할 경우, low level TF API를 사용할때와는 달리, Graph나 Session을 고려할 필요가 없습니다. 또한, Pre-made model을 제공하는데 일부 코드만 변경하면 다른 데이터셋에서 쉽게 적용해서 모델을 사용할 수 있습니다. [DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier)

## Structure of a pre-made Estimators program
pre-made Estimator로 TF 모델을 개발하는 방법은 다음과 같은 4가지 단계로 구분됩니다.
1. Write one or more dataset importing function
    - training을 위한 데이터셋과 test를 위한 데이터셋을 불러오기 위한 함수를 구현합니다. 각각의 함수는 반드시 2개의 object를 반환해야 합니다.
        - dictionary : keys는 feature 이름, values는 feature data에 해당하는 Tensor 또는 SparseTensor로 구성된 dictionary object
        - Tensor : 하나이상의 레이블로 구성된 Tensor
    - 함수의 예는 다음과 같습니다.
    - 데이터셋을 불러오는 자세한 방법에 대한 내용은 [https://www.tensorflow.org/programmers_guide/datasets](https://www.tensorflow.org/programmers_guide/datasets) 를 참고합시다.

```python
def input_fn(dataset):
   ...  # manipulate dataset, extracting feature names and the label
   return feature_dict, label
```

2. Define the feature columns
    - [tf.feature_column](https://www.tensorflow.org/api_docs/python/tf/feature_column)을 이용해서, feature name, type, pre-processing 등을 정의합니다.

```python
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn='lambda x: x - global_education_mean')
```

3. Instantiate the relevant pre-made Estimator
    - pre-made Estimator 의 인스턴스를 생성합니다.

```python
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.Estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )
```

4. Call a training, evaluation, or inference method
    - train, eval, predict 함수를 호출해서 학습시키거나 evaluation, prediction을 수행합니다.

```python
# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)
```

## Benefits of pre-made Estimators
Pre-made Estimators encode best practices, providing the following benefits:
- Best practices for determining where different parts of the computational graph should run, implementing strategies on a single machine or on a cluster.
- Best practices for event (summary) writing and universally useful summaries.
If you don't use pre-made Estimators, you must implement the preceding features yourself.

## Custom Estimators
- 커스텀 Estimator를 구현하기 위해서는 먼저 training, evaluation, prediction을 위한 graph를 빌드할 수 있는 model 함수를 구현해야 합니다. 이에 대한 자세한 내용은 [https://www.tensorflow.org/get_started/custom_estimators](https://www.tensorflow.org/get_started/custom_estimators) 를 참고합시다.

## Recommended workflow
We recommend the following workflow:
1. Assuming a suitable pre-made Estimator exists, use it to build your first model and use its results to establish a baseline.
2. Build and test your overall pipeline, including the integrity and reliability of your data with this pre-made Estimator.
3. If suitable alternative pre-made Estimators are available, run experiments to determine which pre-made Estimator produces the best results.
4. Possibly, further improve your model by building your own custom Estimator.

## Creating Estimators from Keras models
You can convert existing Keras models to Estimators. Doing so enables your Keras model to access Estimator's strengths, such as distributed training. Call [tf.keras.estimator.model_to_estimator](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator) as in the following sample:

```python
# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_inception_v3.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
```

Note that the names of feature columns and labels of a keras estimator come from the corresponding compiled keras model. For example, the input key names for train_input_fn above can be obtained from keras_inception_v3.input_names, and similarly, the predicted output names can be obtained from keras_inception_v3.output_names.

For more details, please refer to the documentation for [tf.keras.estimator.model_to_estimator](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator).

## References
[1] [imgaug - image Augmentation](https://github.com/aleju/imgaug) <br />
[2] [Tensorflow Document - Reading Data](https://www.tensorflow.org/api_guides/python/reading_data) <br />
[3] [Caffe to Tensorflow - PreTrained Model](https://github.com/ethereon/caffe-tensorflow) <br />
[4] [Deep Learning Model Convertor](https://github.com/ysh329/deep-learning-model-convertor) <br />