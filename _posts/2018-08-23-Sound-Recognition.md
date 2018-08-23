---
layout: post
title: Sound Recognition
description: "Sound Recognition을 위한 다양한 기술 정리"
modified: 2018-08-23
tags: [Sound Recognition]
category: Sound Recognition
image:
  feature: posts/soundwaves.jpg
  credit: Mick Lissone
  creditlink: https://www.publicdomainpictures.net/en/view-image.php?image=68467&picture=sound-waves
---

# Domain: Sound Recognition
이 포스트에서는 Sound Recognition과 관련된 여러 분야와 기술들을 정리할 예정입니다.

- **음성 인식(Speech Recognition)**

- **화자 인식(Speaker Recognition)** <br/>
    화자 인식 기술은 일반적으로 화자 식별(Speaker Identification)과 화자 검증(Speaker Verification) 으로 나눠집니다. 화자 인식을 통한 보안 시스템은 이 두가지 기술이 모두 포함되어 구현됩니다.
    + 화자 식별(Speaker Identification) : <br/>
    주어진 발언(utterance)으로부터 해당하는 화자(인물)을 찾아내는 기술
    + 화자 검증(Speaker Verification) : <br/>
    주어진 발언(utterance)이 시스템에 등록된 사용자들의 목소리 중에 있는지 아닌지 검증하는 기술

    또한 화자 인식은 제한된 환경(특정 문장의 사용)에서의 인식이냐 아니냐에 따라서 문장독립(text-independent)과 문장종속(text-dependent)로 구분될 수 있습니다.
    + 문장독립(text-independent) : <br/>
    화자 인식을 위해 발언하는 문장의 형식이나 종류에 제한이 없는 방식
    + 문장종속(text-dependent) : <br/>
    화자 인식을 위해 사용자가 특정 문장의 형식이나 종류로 발언해야 하는 방식

    당연하게도, 문장독립방식의 화자인식 기술의 연구가 더 어려운 기술입니다.
    문장종속 방식의 경우 주어진 문장이라는 제한 환경이 있으므로, 사전에 화자 인식을 위한 기술 개발에서 고려해야할 사항을 제한할 수 있기 때문입니다.

- **소리 분류(Sound Classification)** <br/>
- **음원 분리(Sound Source Separation)** <br/>
- **음원 위치 추정(Sound Source Localization)** <br/>

# Sound Features
- Mel Spectrogram <br/>
- Chroma <br/>
- MFCCs (Mel Frequency Cepstrum Coefficients) <br/>
- ...

# Recognition Techniques
- GMM (Gaussian Mixture Model)
- HMM (Hidden Markov Model)
- ...

# Public DataSet
※ 도메인별로 정리 예정

[1] AudioSet - [https://research.google.com/audioset/index.html](https://research.google.com/audioset/index.html) <br/>
[2] common-voice(kaggle) - [https://www.kaggle.com/mozillaorg/common-voice](https://www.kaggle.com/mozillaorg/common-voice) <br/>
[3] Korean Single Speaker Speech Dataset - [https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset) <br/>
[3] Urban Sound Datasets - [https://urbansounddataset.weebly.com/](https://urbansounddataset.weebly.com/)
[4] VoxCeleb - [http://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/](http://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/)

## References
[1] librosa - [https://librosa.github.io/](https://librosa.github.io/) <br/>
[2] Fourier Transform YouTube Video(3Blue1Brown) - [https://www.youtube.com/watch?v=spUNpyF58BY&t=1026s](https://www.youtube.com/watch?v=spUNpyF58BY&t=1026s)