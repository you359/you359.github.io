---
layout: post
title: Paper Review
description: "AI 특히 ML/DL과 관련된 관심 논문 리스트"
modified: 2018-09-18
tags: [Review]
category: Paper Review
image:
  feature: posts/review.jpg
  credit: Nick Youngson
  creditlink: http://www.thebluediamondgallery.com/typewriter/r/review.html
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
- <span style="color:var(--main-txt-color)">[OverFeat]</span> OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks, 2013 [[paper]](https://arxiv.org/abs/1312.6229)
- <span style="color:var(--main-txt-color)">[R-CNN]</span> Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation, 2014 [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
- <span style="color:var(--main-txt-color)">[YOLO]</span> You Only Look Once: Unified, Real-Time Object Detection, 2015 [[paper]](https://arxiv.org/pdf/1506.02640.pdf)
- <span style="color:var(--main-txt-color)">[SPP-Net]</span> Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, 2015 [[paper]](https://ieeexplore.ieee.org/abstract/document/7005506/)
- <span style="color:var(--main-txt-color)">[Fast R-CNN]</span> Fast r-cnn, 2015 [[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
- <span style="color:var(--main-txt-color)">[Faster R-CNN]</span> Faster R-CNN: towards real-time object detection with region proposal networks, 2015 [[paper]](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)
- <span style="color:var(--main-txt-color)">[SSD]</span> SSD: Single Shot MultiBox Detector, 2015 [[paper]](https://arxiv.org/pdf/1512.02325.pdf)
- <span style="color:var(--main-txt-color)">[R-FCN]</span> R-FCN: Object Detection via Region-based Fully Convolutional Networks, 2016 [[paper]](https://arxiv.org/pdf/1605.06409.pdf)
- <span style="color:var(--main-txt-color)">[FPN]</span> Feature Pyramid Networks for Object Detection, 2016 [[paper]](https://arxiv.org/pdf/1612.03144.pdf)
- <span style="color:var(--main-txt-color)">[YOLOv2, YOLO9000]</span> YOLO9000: Better, Faster, Stronger, 2016 [[paper]](https://arxiv.org/pdf/1612.08242.pdf)
- <span style="color:var(--main-txt-color)">[Mask R-CNN]</span> Mask R-CNN, 2017 [[paper]](https://ieeexplore.ieee.org/abstract/document/8237584/)
- <span style="color:var(--main-txt-color)">[RetinaNet]</span> Focal Loss for Dense Object Detection, 2017 [[paper]](https://arxiv.org/pdf/1708.02002.pdf)
- <span style="color:var(--main-txt-color)">[YOLOv3]</span> YOLOv3: An Incremental Improvement, 2018 [[paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

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

## Unsupervised Learning
### Deep Belief Network
### Auto Encoder
### Generative adversarial Network

## Reinforcement Learning
### DQN

## Meta Learning
### Siamese Network
- Learning a Similarity Metric Discriminatively with Application to Face Verification, 2005 [[paper]](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)
- Learning Fine-grained Image Similarity with Deep Ranking, 2014 [[paper]](https://arxiv.org/pdf/1404.4661.pdf)
- Siamese Neural Networks for One-shot Image Recognition, 2015 [[paper]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- DEEP METRIC LEARNING USING TRIPLET NETWORK, 2015 [[paper]](https://arxiv.org/pdf/1412.6622.pdf)
- FaceNet: A Unified Embedding for Face Recognition and Clustering, 2015 [[paper]](https://arxiv.org/pdf/1503.03832.pdf)
- In Defense of the Triplet Loss for Person Re-Identification, 2017 [[paper]](https://arxiv.org/pdf/1703.07737.pdf)

## Curriculum learning


# Domain
### image restoration/reconstruction (super-resolution)
### Fire Detection
### Image Forensics
### Financial Analysis
### Sound Recognition
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> VoxCeleb: a large-scale speaker identification dataset, 2018 [[paper]](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf) [[review]](/sound%20recognition/VoxCeleb/)

### Agriculture
- Evaluation of Features for Leaf Classification in Challenging Conditions, 2015 [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7045965)
- Deep Neural Networks Based Recognition of Plant Diseases by Leaf Image Classification, 2016 [[paper]](https://www.hindawi.com/journals/cin/2016/3289801/abs/)
- Using Deep Learning for Image-Based Plant Disease Detection, 2016 [[paper]](https://arxiv.org/ftp/arxiv/papers/1604/1604.03169.pdf)
- A Deep Learning-based Approach for Banana Leaf Diseases Classification, 2017 [[paper]](http://btw2017.informatik.uni-stuttgart.de/slidesandpapers/E1-10/paper_web.pdf)


- Deep learning in agriculture: A survey, 2018 [[paper]](https://reader.elsevier.com/reader/sd/pii/S0168169917308803?token=02BFEA0B3D0F7C34BDA46E7C9E8B866AF2CB401526D908DA60CF0EE228E1FE977C9BC95E37F2E5CD779C196671C3C080)

- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> University of Arcansas, Plants Dataset [[site1]](https://plants.uaex.edu/herbicide/) [[site2]](https://www.uaex.edu/yard-garden/resource-library/diseases/)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> EPEL, Plant Village Dataset [[site]](https://plantvillage.psu.edu/)
    + currently not available!
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> Leafsnap Dataset [[site]](http://leafsnap.com/dataset/)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> LifeCLEF Dataset [[site]](https://www.imageclef.org/2014/lifeclef/plant)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> Africa Soil Information Service(AFSIS) Dataset [[site]](http://africasoils.net/services/data/)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> UC Merced Land Use Dataset [[site]](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> MalayaKew Dataset [[site]](http://web.fsktm.um.edu.my/~cschan/downloads_MKLeaf_dataset.html)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> Crop/Weed Field Image Dataset [[paper]](https://pdfs.semanticscholar.org/58a0/9b1351ddb447e6abdede7233a4794d538155.pdf) [[site]](https://github.com/cwfid/dataset)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> University of Bonn Photogrammetry, IGG [[site]](http://www.ipb.uni-bonn.de/data/)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> Flavia leaf Dataset [[site]](http://flavia.sourceforge.net/)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> Syngenta Crop Challenge 2017 [[site]](https://www.ideaconnection.com/syngenta-crop-challenge/challenge.php)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> Plant Image Analysis [[site]](https://www.plant-image-analysis.org/dataset)
- <span style="color:var(--main-txt-datset-color)">[DATASET]</span> PlantVillage Disease Classification Challenge - Color Images [[site]](https://zenodo.org/record/1204914#.W6CpEKYzb-g)