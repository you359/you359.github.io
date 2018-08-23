---
layout: post
title: Paper Review
description: "AI 특히 ML/DL과 관련된 관심 논문 리스트"
modified: 2018-08-23
tags: [Survey]
category: Paper List
image:
  feature: abstract-3.jpg
  credit:
  creditlink:
---

<style>
:root {
    --main-txt-color: coral;
    --main-txt-datset-color: LimeGreen;
}
</style>

# General Purpose
## Supervised Learning
### Convolutional Neural Network
#### Classification
- <span style="color:var(--main-txt-color)">[LeNet]</span> Gradient-Based Learning Applied to Document Recognition, 1998 [[paper]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- <span style="color:var(--main-txt-color)">[AlexNet]</span> ImageNet Classification with Deep Convolutional Neural Networks, 2012 [[paper]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- <span style="color:var(--main-txt-color)">[ZFNet]</span> Visualizing and Understanding Convolutional Networks, 2014 [[paper]](https://arxiv.org/pdf/1311.2901.pdf)
- <span style="color:var(--main-txt-color)">[VGGNet]</span> Very Deep Convolutional Networks for Large-Scale Image Recognition, 2014 [[paper]](https://arxiv.org/pdf/1409.1556.pdf)
- <span style="color:var(--main-txt-color)">[GoogLeNet(inception)]</span> Going Deeper with Convolutions, 2015 [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
- <span style="color:var(--main-txt-color)">[ResNet]</span> Deep Residual Learning for Image Recognition, 2016 [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- <span style="color:var(--main-txt-color)">[DenseNet]</span> Densely Connected Convolutional Networks, 2017 [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)
- <span style="color:var(--main-txt-color)">[XceptionNet]</span> Xception: Deep Learning with Depthwise Separable Convolutions, 2017 [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)

#### Object Detection
- <span style="color:var(--main-txt-color)">[Fast R-CNN]</span>
- <span style="color:var(--main-txt-color)">[Faster R-CNN]</span>

#### Semantic Segmantation
-

#### CNN visualization
- Deep inside convolutional networks: Visualising image classification models and saliency maps, 2013 [[paper]](https://arxiv.org/pdf/1312.6034.pdf)
- <span style="color:var(--main-txt-color)">[ZFNet]</span> Visualizing and Understanding Convolutional Networks, 2014 [[paper]](https://arxiv.org/pdf/1311.2901.pdf)
- do convnets learn correspondence?, 2014 [[paper]](http://papers.nips.cc/paper/5420-do-convnets-learn-correspondence.pdf)
- Object Detectors Emerge in Deep Scene CNNs, 2014 [[paper]](https://arxiv.org/pdf/1412.6856.pdf)
- Rich feature hierarchies for accurate object detection and semantic segmentation, 2014 [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
- Striving for simplicity: The all convolutional net, 2014 [[paper]](https://arxiv.org/pdf/1412.6806.pdf) [[review]](/cnn%20visualization/All-Convnet/)
- Deep neural networks are easily fooled: High confidence predictions for unrecognizable images, 2015 [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)
- explaining and harnessing adversarial examples, 2015 [[paper]](https://arxiv.org/pdf/1412.6572.pdf)
- Understanding deep image representations by inverting them, 2015 [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.pdf)
- <span style="color:var(--main-txt-color)">[CAM]</span> Learning Deep Features for Discriminative Localization, 2016 [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) [[review]](/cnn%20visualization/CAM/)
- Inverting visual representations with convolutional networks, 2016 [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Dosovitskiy_Inverting_Visual_Representations_CVPR_2016_paper.pdf)
- <span style="color:var(--main-txt-color)">[Grad-CAM]</span> Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, 2017 [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf) [[review]](/cnn%20visualization/GradCAM/)

### Recurrent/Recursive Neural Network
-

### Both CNN & RNN

### Meta Learning
#### Siamese Network
- Learning a Similarity Metric Discriminatively with Application to Face Verification, 2005 [[paper]](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)
- Learning Fine-grained Image Similarity with Deep Ranking, 2014 [[paper]](https://arxiv.org/pdf/1404.4661.pdf)
- Siamese Neural Networks for One-shot Image Recognition, 2015 [[paper]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- DEEP METRIC LEARNING USING TRIPLET NETWORK, 2015 [[paper]](https://arxiv.org/pdf/1412.6622.pdf)
- FaceNet: A Unified Embedding for Face Recognition and Clustering, 2015 [[paper]](https://arxiv.org/pdf/1503.03832.pdf)

## Unsupervised Learning
### Deep Belief Network
### Auto Encoder
### Generative adversarial Network

## Reinforcement Learning
### DQN

# Domain
### Fire Detection
### Image Forensics
### Financial Analysis
### Sound Recognition
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> VoxCeleb: a large-scale speaker identification dataset, 2018 [[paper]](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf) [[review]](/sound%20recognition/VoxCeleb/)
