---
layout: post
title:  "EfficientNet : Rethinking Model scaling for Convolutional Neural Networks"
categories: paper
date : 2019-07-04 10:05:05 +0900
comments: true
---

# Abstract

Convolutinal Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and the scaled up for better accuracy if more resources are available.  
CNN은 일반적으로 고정된 자원 예산 안에서 개발된다 그리고 더 많은자원이 있을경우 더 높은 정확도를 위해 규모를 키운다

In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance  
이 논문에서는 체계적으로 모델의 스케일을 학습하고 더 좋은 퍼포먼스를 이끌수있는 깊이와 넓이 그리고 해상도의 밸런스를 알아낸다

Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/ width/ resolution using a simple yet highly effective compound coefficient.  
이 관측에 기반하여 우리는 간단하고도 매우 효과적인 합성계수를 사용하여 깊이, 넓이, 해상도의 차원을 균일하게 조절할 새로운 스케일 매소드를 제안한다

We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet  
우리는 이 메소드의 상승 효과를 모바일넷과 레즈넷에서 시연하였다

To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets.

더 나아가, 우린 새로운 시작 베이스라인을 디자인하고 그것을 스케일업한 에피션넷이라 불리는모델들을 얻었다 이전의 ConvNet보다 정확도와 효율성이 좋은

In particular, our EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on Imagenet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet.  
특히 에피션넷은 이미지넷 에서 탑1에서 84.4%, 탑5에서 97.1%의 최고기록인 정확도를 달성했고 기존의 베스트 ConvNet보다 8.4배 작고 6.1배 빠른 속도로 추론했다

Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100(91.7%), Flowers(98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.  
에피션트넷의 또다른 최고기록은 CIFAR100에서 91.7%, Flowers에서 98.8%, 그리고 3개의 다른 데이터셋에서도 세웠고 매개변수의 수는 더 적었다.


Source code is at 
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet.


# 1. Introduction

Scaling up ConvNets widely used to achieve better accuracy.    
콘브넷의 스케일업은 더높은 정확도의달성을 위해 널리 사용된다

For example, ResNet can be scaled up from ResNet-18 to ResNet-200 by using more layers.  
예를들면 레즈넷은 18부터 200까지 레이어를 늘리는것으로 스케일업이 가능하다

Recently, GPipe achieved 84.3% ImageNet top-1 accuracy by scaling up a baseline model for time larger  
요즘엔 GPipe는 이미지넷 탑1에서 84.3%를 달성했다 
베이스라인의 4배로 스케일업을해서

However, the process of scaling up ConvNets has never been well understood and there are currently many ways to do it.  
그러나 콘브넷을 확장하는 과정은 쉽게 이해되지 않고 현재 많은방법이 있다

The most common way is to scale up ConvNets by their depth or width.
가장 흔한 확장방법은 깊이와 너비를 늘리는것이다

Another less common, but increasingly popular, method is to scale up models by image resolution.  
다른 아직은 흔하진 않지만 점점 유명해지고 있는 방법은 이미지 해상도를 확장하는것이다

In previous work, it is common to scale only one of the three dimensions - depth, width, and image size.

Though it is possible to scale two or three dimensions arbitrarily, arbitrary scaling requires tedious manual tuning and still often yields sub-optimal accuracy and efficiency  
그래도 두개혹은세개의 차원은 임의적으로 확장이 가능하다. 임의확장은 지루한 수동튜닝이 필요하고 여전히 정확도와 효율성에있어 최적의 선택이 아니다

In this paper, we want to study and rethink the process of scaling up ConvNets
이 논문에서 우리는 콘브넷을 확장하는 과정을 연구하고자 한다

In particular, we investigate the central question : is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency?  
특히 우리는 이 질문에 대해 연구했다
더 높은정확도와 효율성을달성하기 위한 콘브넷에 스케일업에 규칙적인 방법이 있는가

Our empirical study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with contant ratio.
우리의경험적인 학습은 보여준다
깊이, 너비, 해상도의 밸런스를 유지하는것이 매우 중요하고 놀랍게도 이러한 밸런스는 단순한 각각의 상수 비율로 나타낼 수 있다

Based on this observation, we propose a simple yet effective compound scaling method.  
이 관측에 기반하여 우리는 간단하고 효과적인 합성 스케일 방법을 제안한다

Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.  
이런 요인을 임의로 조정하던 기존의 방식과는 달리 우리의 방법은 고정된 조정계수의 집합으로 균등하게 조정한다.

For example, if we want to use $2^N$ times more computational resources, then we can simply increse the network depth by $\alpha^N$, width by $\beta^N$, and image size by $\gamma^N$, where $\alpha,\beta,\gamma$ are constant coefficients determined by a small grid search on the original small model.
예를들면 만약 $2^N$의 많은 컴퓨팅 자원이 있다면 우리는 네트워크를 단순히 $\alpha^N,\beta^N,\gamma^N$ 만큼 증가시킬수 있다 
위 변수들은 원래의 작은 모델에서의 작은 그리드서치에 의해 결정 되는 상수이다.

intuitively, the compound scaling method makes sense because if the input image is bigger, then the neywork needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.
직관적으로 이 복합조정방법은 말이된다 왜냐면 이미지가 커지면 네트워크는 증가된 필드를 수용하기위해 많은레이어를 필요로 하고 더 큰 이미지에서 좋은패턴을 얻기 위해 더 많은채널을 필요로 한다


In fact, precvious theoretical and empirical results both show that there exists certain relationship between network width anddepth, but to our best knowledge, we are the first to empirically quantify the relationship among all three dimensions of network width, depth, and resolution.  
사실상 기존 이론과 실증적인 결과는 둘다 네트워크의 너비와 깊이간의 확실한 관계가 존재하는것을 보여준다
그러나 우리의 최고 지식은 처음으로 실제적으로 세가지 차원에 대한관계를 계량한 것이다.
















