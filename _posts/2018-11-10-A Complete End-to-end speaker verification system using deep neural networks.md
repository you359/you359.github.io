---
layout: post
title: Paper Review - A COMPLETE END-TO-END SPEAKER VERIFICATION SYSTEM USING DEEP NEURAL NETWORKS - FROM RAW SIGNALS TO VERIFICATION RESULT
description: ""
modified: 2018-11-10
tags: [Review, Sound Recognition]
category: Sound Recognition
image:
  feature: posts/soundwaves.jpg
  credit: Mick Lissone
  creditlink: https://www.publicdomainpictures.net/en/view-image.php?image=68467&picture=sound-waves
---

# Paper Review - A COMPLETE END-TO-END SPEAKER VERIFICATION SYSTEM USING DEEP NEURAL NETWORKS: FROM RAW SIGNALS TO VERIFICATION RESULT
이 포스트에서는 2018년 IEEE International Conference on Acoustics 에 올라온 "A COMPLETE END-TO-END SPEAKER VERIFICATION SYSTEM USING DEEP NEURAL NETWORKS: FROM RAW SIGNALS TO VERIFICATION RESULT" 논문에 대해 살펴보겠습니다.

### 전통적인 화자 인식 시스템의 구성
2014~15년도 이전의 전통적인 화자 인식 시스템은 보통 다음과 같은 4개의 스테이지로 구성되어 있었습니다.
1. pre-processing
2. acoustic feature extraction
3. speaker feature extraction
4. binary classification

15년도 이후부터는, 이 4개의 스테이지의 일부를 DNN으로 대체하는 시도가 있었으며,
특히 d-dector나 b-vector의 경우, 전통적인 화자 인식 시스템의 3, 4번째 스테이지를 한번에 처리하도록 구성하기도 했습니다.

최근에는, 2~3 번째 스테이지의 feature extraction 단계부터, 마지막 classification 단계까지 한번에 end to end 로 DNN 을 구성해서 화자를 인식하는 방법이 제안되고 있습니다.
이러한 방법들은 먼저 입력 오디오 신호에 대해서, MFCCs나 mel-filterbank energies, spectrogram 등으로 pre-processing을 한 이후에, pre-processing 결과들에 대해서 DNN 모델을 구성해서 화자를 인식합니다.

### DNN from raw audio signal
이 논문에서는, MFCCs와 같은 전처리 혹은 feature extraction 을 사용하지 않고, 순수 raw audio signal 들을 입력으로 취하는 DNN을 구성해서 화자를 인식하는 방법에 대해 제안하고 있습니다.
모델의 구성은 다음과 같습니다.

1. pre-processing layer
2. speaker feature extraction layer
3. b-vector system

#### pre-processing layer
먼저 pre-processing layer에 대해 살펴보기 이전에, 논문에서 언급하는 바로는,
raw audio signal에 대해서 직접적으로 DNN을 적용하기 어려운 가장 큰 이유 중 하나는, raw audio signal 값의 변동이 너무 크기 때문이라고 합니다. (-32,768 ~ 32,767, 16bit) <br/>
이러한 문제를 해결하기 위해 audio signal processing 에서는 pre-emphasis 라는 기술을 적용하게 되는데, 이 pre-emphasis 는, 높은 주파수 신호를 강조해서 변조 지수를 일정하게 유지하므로 raw audio signal 의 크기를 안정화시키는 역할을 합니다.
pre-emphasis 에 대한 수식은 다음과 같습니다.

$$
    p(t) = s(t) - \alpha s(t-1)
$$

대부분의 audio-signal processing 에서 다루는 pre-emphasis 의 coefficient $$ \alpha $$ 는 0.97로 정한다고 합니다.

이 논문에서는 이러한 pre-emphasis 를 (k=2)의 convolutional layer를 이용해서 구현하며, 해당 conv layer의 2개 weights 를 [[-0.97, 1]] 로 초기화해서 사용했다고 합니다. (위 수식과 동일한 동작) <br/>
이후 이 weights 값은 학습을 통해 좀더 fine-tuning 되는데, 학습에서 이 weight 값이 급격하게 변화하는 것을 방지하기 위해, pre-emphasis 에 해당하는 conv layer의 learning rate를 다른 레이어에 비해 작게 주었습니다.

#### speaker feature extraction layer
논문에서는 화자 특징 추출을 위해 다음과 같은 2개의 모델을 제안합니다.
1. RACNN
    RACNN 은 9개의 Conv + max-pooling layer, 2개의 fc layer로 구성된 CNN 모델로, strided convolution 을 사용했다고 합니다.
    여기서 conv layer의 구성은 (k=3, s=1) 이고, fc layer의 node(unit)의 수는 512로 설정했으며,첫 번째 fc layer에는 linear activation 을 적용했습니다.

2. RACNN-LSTM
    RACNN-LSTM 은 앞서 1.의 RACNN 모델의 5번째 pooling layer의 출력 feature map(2d: time step, num of kernel)을 81d vector로 변환한 후, 해당 vector를 입력으로 취하는 LSTM + 2개 fc layer를 추가한 모델입니다.

#### b-vector system
최종 화자의 검증(verification) 으로는 b-vector classifier를 사용해서 분류했으며, 해당 classifier 는 입력 layer, 5개의 hidden layer, 1개의 출력 layer를 갖는 MLP로 구성됩니다.
여기서 입력 레이어는 총 1536(512 x 3) 차원의 입력을 받아들이는데, 이는 일종의 augmentation 으로, speaker model의 출력과, test utterance 간에 summation(+), subtraction(-), multiplication(*)을 적용한 각각의 512 vector들을 이어붙인 것을 의미하는 것 같습니다. <br/>
이후, 1536d의 입력 vector는 1024개의 unit을 갖는 5개 hidden layer를 거쳐, 2개의 output unit을 갖는 최종 출력 layer로 전달됩니다.

※ 모델의 전체 학습에 대해서, 논문에서는 joint optimization approach를 적용했다고 합니다.
    joint optimization approach 는, 각각의 모델 RACNN, RACNN-LSTM, b-vector classifier 에 대해 각각의 출력 fc layer들을 붙여 각각 학습시킨 후, 이어붙여서 fine-tuning 하는 방식으로 학습시키는 방법을 의미하는 것 같습니다.
    따라서, 이후 RACNN의 마지막 conv, fc layer는 제거합니다. (verification 에서 필요하지 않으므로..)
+ RACNN의 모델은 speaker identification 방식으로 학습시키는 것 같습니다.

### DataSet
논문에서 사용한 데이터셋으로는 RSR 2015 를 사용했는데,
이 데이터셋은 총 300명의 화자로 구분되고, 각 화자별로 270개의 발성 데이터(3.2초, 9개 세션 * 서로 다른 phrase)를 갖습니다.
또한 100(남성) + 94(여성)개 화자를 dev(training) 데이터로 사용하고, 나머지 106개 화자를 test 데이터로 사용합니다.

### 모델 요약 정리
1. input shape = 59,049 ($$ 3^10 $$)
2. pre-emphasis layer = Conv(k=2, s=1)
3. Conv(k=128, s=3)
4. RACNN model = ( (Conv(k=3, s=1) + max pooling(k=3) ) * 9 -> fc(512, linear activation) -> fc(512)
5. RACNN-LSTM model = RACNN model의 5번째 pooling layer output (81d, $$ =3^4 $$) ->  fc(512, linear activation) -> fc(512)
6. b-vector system = 1536 input -> fc(1024d) * 5 -> output layer(2d)
※ 모든 시스템에는 dropout과 batch normalization 이 적용되었음

### 결과
- pre-emphasis 의 적용 결과, 학습에 의한 fine tuning 으로 [[-0.97 1]] 에서 [[-0.83 1.12]]로 값이 변경되었고, EER(Equal Error Rate) metric 으로 검증한 결과, pre-emphasis를 적용했을 때 24% 작은 EER을 얻을 수 있었다고 합니다.
- RACNN의 출력 임베딩 벡터를 CSS(Cosine Similarity Scoring)으로 계산해서, 결과를 만들 경우, d-vector baseline보다 낮은 성능을 보였지만, end-to-end로 구성한 시스템이서는 좋은 결과를 보였으며(baseline, RACNN(CSS), RACNN(end to end) | 4.89, 5.22, 3.94)
- 또한, RACNN-LSTM의 경우 CSS로 계산해도 baseline 보다 좋은 결과를 보였고(3.82), end to end RACNN-LSTM은 3.63 EER로 가장 높은 성능을 보였습니다.

## References
[1] A COMPLETE END-TO-END SPEAKER VERIFICATION SYSTEM USING DEEP NEURAL NETWORKS: FROM RAW SIGNALS TO VERIFICATION RESULT, 2018 [[paper]](https://ieeexplore.ieee.org/abstract/document/8462575)
