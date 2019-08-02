---
layout: post
title:  "EfficientNet : Rethinking Model scaling for Convolutional Neural Networks"
tag: paper
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

In fact, previous theoretical and empirical results both show that there exists certain relationship between network width and depth, but to our best knowledge, we are the first to empirically quantify the relationship among all three dimensions of network width, depth, and resolution.  
사실 앞의 이론적이고 경험적인 결과 둘다 네트워크의 깊이와 너비에 어떤 관계가 존재하는걸 보여주지만 우리의 최고의 성과는 이 세가지 차원의 관계를 처음으로 정량화한 것이다.

we demonstrate that our scaling method work well on existing MobileNets and ResNet.  
우리는 우리의 스케일링 방식이 레즈넷과 모바일넷에서 잘 작동하는것을 시연한다

Notably, the effectiveness of model scaling heavily depends on the baseline network; to go even further, we use neural architecture search to develop a new baseline network, and scale it up to obtain a family of models, called EfficientNets.  
특히 모델의 확장 효과는 베이스라인에 의존한다. 더 나아가 신경망 아키텍쳐를 더 좋은 새로운 베이스모델을 만들기 위해 사용하고 그것으르 스케일하여 만든 모델의 집합을 에피션넷츠라 부른다

Figure 1 summarizes the ImageNet performance, where our EfficientNets significantly outperform other ConvNets.  
그림1은 이미지넷 퍼포먼스의 요약이고 에피션트넷이 다른 컨브넷을 훨씬 능가하는것을 보여준다

In particular, out EfficientNet-B7 surpasses the best existing GPipe accuracy, but using 8.4x fewer parameters and running 6.1x faster on inference.  
특히 B7 은 현존하는 최고의 정확도를 넘어섰지만 8.4배 적은 파라미터를 가지고 추론시 6.1배 빠른 속도가 나온다

Compared to the widely used ResNet-50, our EfficientNet-B4 improves the top-1 accuracy from 76.3 to 82.6 with similar FLOPS  
널리 사용되는 레즈넷 50과 비교하면 B4는 비슷한 연산량에서 탑1 정확도를 8프로나 올렸다 

Besides ImageNet, Efficient Nets also transfer well and achieve state-of-the-art accuracy on 5 out of 8 widely used datasets, while reducing parameters by up to 21x than existing ConvNets  
이미지넷 외에 에피션넷은 또한 널리 사용되는 8개의 데이터세트중 5개에서 최고기록을 달성하면서도 파라미터는 최대 21배까지도 감소시켰다


# 2. Related Work

## ConvNet Accuracy
Since AlexNet won the 2012 ImageNet competition, ConvNets have become incresingly more accurate by going bigger: while the 2014 ImageNet winner GoogleNet achieves 74.8% top-1 accuracy with about 6.8M parameters, the 2017 ImageNet winner SENet achieves 82.7 top-1 accuracy with 145 parameters.
알렉스넷이 2012에 우승한 이후 컨브넷은 커짐으로써 점점 높은 정확도를 달성했다. 2014이미지넷에서 우승한 구글넷은 680만개, 207에서 우승한 SENet은 1억4500백만개의 파라미터를 가졌다

Recently, GPipe further pushes the state-of-the-art ImageNet top-1 validation accuracy to 84.3% using 557M parameters : it is so big that it can only be trained with a specialized pipeline parallelism library by partitioning the network and spreading each part to a different accelerator  
현재 GPipe는 최고기록 84.3%를 위해 5억5700만개의 파라미터를 사용했다
이것은 매우 커서 다른 가속기로 분산하는 특별한 파이프 병렬 파이프라인 라이브러리를 사용해야만 한다

While these models are mainly designed for ImageNet, recent studies have shown better ImageNet models also perform better across a variety of transfer learning datasets and other computer vision tasks such as object detection.  
이 모델들이 이미지넷에서 디자인되는 동안, 최근 연구들은 이미지넷모델이 또한 많은 종류의 전이학습이나 오브젝트 디텍션 같은 다른 컴퓨터비전 학습에서도 좋은 성능을 제공하는것을 보여주었다

Although higher accuracy is critical for many applications, we have already hit the hardware memory limit, and thus further accuracy gain needs better efficiency  
높은 정확도가 많은 어플리케이션에서 중요할지라도 우리는 이미 하드웨어 메모리 제한에 도달했고 그래서 정확성의 증가는 더 좋은 효율이 필요했다.

## ConvNet Efficiency
Deep ConvNets are often overparameterized.   
심층컨볼넷은 종종 오버파라미터된다. 

Model compression is a common way to reduce model size by trading accuracy for efficiency.  
모델 압축은 효율을 위해 정확도를 낮춰 모델의 크기를 줄이는 흔한 방법이다

As mobile phones become ubiquitous, it is also common to handcraft efficient mobile-size ConvNets, such as SqueezeNets, MobileNets, and ShuffleNets.  
모바일 폰은 어디에나있다. 그것은 수작업된 효율좋은 모바일사이즈 컨볼넷 또한 흔하다 스퀴즈넷, 모바일넷, 셔플넷같은.

Recently, neural archiecture search becomes increasingly popular in designing efficient mobile-size ConvNets, and achieves even better efficiency than hand-crafted mobile ConvNets by extensively tuning the network width, depth, convolution kernel types and sizes.  
최근 신경망 아키텍쳐는 효율적인 모바일크기 컨볼넷의 디자인이 인기가 많아지고있고 수작업으로 만들어진 모바일 컨브넷보다 광범위하게 튜닝 가능한 것이 훨씬 더 좋은 성취를 한다.

However, it is unclear how to apply these techniques for larger models that have much larger design space and much more expensive tuning cost.  
그러나 모델을 크게만드는 테크닉을 적용하는 방법은 아직 모른다.
그래서 큰 디자인을 설계하는데는 매우 많은 튜닝 비용이 필요하다

In this paper, we aim to study model efficiency for super large ConvNets that surpass state-of-the-art accuracy. To achieve this goal, we resort to model scaling.  
이 논문에서 최고의 정확도를 능가하는 매우 큰 컨볼넷의 효율을 연구하는것에 집중했고 이 목적을 달성하러면 우린 모델 확장에 의존해야 한다

## Model Scaling
There are many ways to scale a ConvNet for different resource constraints : ResNet can be scaled down or up by adjusting network depth, while WideResNet and MobileNets can be scaled by network width  
다양한 자원의 제약을 위해 컨볼넷을 확장하는 방법은 많이 있다
ResNet은 깊이를 조절하여 스케일 업다운이 가능하고 모바일넷과 와이드레즈넷은 네트워크의 너비를 조정 가능하다

It is also well-recognized that bigger input image size will help accuracy with the overhead of more FLOPS  
큰 입력 이미지 사이즈가 정확도 를 돕지만 연산량이 늘어난다는것 또한 잘 알려져있다.

Although prior studies have shown that network deep and width are both important for ConvNets' expressive power, it still remains an open question of how to effectively scale a ConvNet to achieve better efficiency and accuracy.  
비록 이전의 연구가 컨볼넷의 표현력을 위해 네트워크의 깊이와 넓이 가 모두 중요한 것임을 보여줬지만 여전히 어떻게 높은 효율과 정확도를위해 효과적으로 컨볼넷을 스케일하하는지는 의문으로 남아있다.

Our work systematically and empirically studies ConvNet scaling for all three dimensions of network width, depth, and resolutions  
우리의 작업은 3가지 기준을 이용한 컨볼넷의 확장을 체계적이고 경험적으로 연구하는것이다

# 3. Compound Model Scaling
In this section, we will formulate the scaling problem, study different approaches, and propose our new scaling method  
이 부분에서 스케일 문제를 공식화하고, 다른접근법을 연구하고, 새로운 스케일 메소드를 제안하겠다

## 3.1. Problem Fomulation
A ConvNet Layer $i$ can be defined as a function : $Y_i = \mathcal{F}_i(X_i)$, where $\mathcal{F}_i$ is the operator, $Y_i$ is output tensor, $X_i$ is input tensor, with tensor shape $\left\langle H_i, W_i, C_i  \right\rangle^1$, where $H_i$ and $W_i$ are spatial dimension and $C_i$ is the channel dimension.

컨볼넷 레이어 i 는  $Y_i = \mathcal{F}_i(X_i)$ 로 정의된다 F는 오퍼레이터 Y는 아웃풋텐서 X는 인풋텐서로 쉐이프는  $\left\langle H_i, W_i, C_i  \right\rangle^1$ 이고 높이와 너비 , 채널을 의미한다

A ConvNet $\mathcal{N}$ can be represented by a list of composed layers : $\mathcal{N} = \mathcal{F}_k 
\bigodot \cdots \bigodot \mathcal{F}_2 \bigodot \mathcal{F}_1(X_1) = \bigodot_{j=1\cdots k}\mathcal{F}_j(X_1)$  
컨볼넷 N은 위와같이 레이어로 구성된 리스트로 표현가능하다

In pratice, ConvNet layers are often partitioned into multiple stages and all layers in each stage share the same architecture : for example, ResNet has five stages, and all layers in each stage has the same convolutional type except the first layer performs down-sampling.  
실제로 컨볼레이언느 종종 몇개의 스테이지로 구분되어있고 각 스테이지의 모든 레이어는 같은 아키텍쳐를 공유한다.
예를들면 레즈넷은 5개의 스테이지를 가지고 각 스테이지의 모든 레이어가 다운샘플링을 위한 첫레이어를 제외하고는 같은 컨볼루션 타입을 가진다

Therefore, we can define a ConvNet as :  
(1)  
$$\mathcal{N} = \underset{i=1\cdots s}{\bigodot}\mathcal{F}^{L_i}_i(X_{\left\langle H_i,W_i,C_i \right\rangle}) $$ 
Where $\mathcal{F}^{L_i}_i$ denotes layer $\mathcal{F}_i$ is repeated $L_i$ times in stage $i$, $\left\langle H_i,W_i,C_i \right\rangle$ denotes the shape of input tensor $X$ of layer $i$.  
컨볼넷을 위처럼 정의할수도 있는데 여기서 F는 스테이지 i 에서 L만큼 반복된다는 것이고 X는 인풋텐서이다)

Figure 2(a) illustrate a representative ConvNet, where the spatial dimension is gradually shrunk but the channel dimension is expanded over layers, for example, from initial input shape <224,224,3>, to final output shape<7,7,512>.  
그림 2 는 콘브넷의 표현을 그린것인데 
차원 공간이 점차적으로 눌어들지만 채널은 레이어가갈수록 확장된다 예를들면 input은 224 224 3 이지만 아웃풋은 7 7 512 이다

Unlike regular ConvNet designs that mostly focus on finding the best layer architecture $\mathcal{F}_i$, model scaling tries to expand the network length $(L_i)$, width $(C_i)$, and/or resulution $(H_i, W_i)$ without changing $\mathcal{F}_i$ predefined in the baseline network.   
최고의 레이어 아키텍쳐를 찾는데 가장 집중된  일반적인 컨볼루션 디자인과는 다르게, 모델 스케일링은 네트워크의 길이,너비, 그리고 해상도를 미리정의된 베이스라인네트워크를 바꾸지 않고모델을 확장하려고 노력한다


By fixing $\mathcal{F}_i$, model scaling simplifies the design problem for new resource constraints, but it still remains a large design space to explore different $L_i,C_i,H_i,W_i$ for each layer.  
F를 고정하여 모델 확장의 새로운 자원 제약 디자인 문제를 간단하게 한다, 그러나 이것은 여전히 각각 레이어의 다른 변수들의 거대한 탐험 공간이 남아있다 

In order to further reduce the design space, we restrict that all layers must be scaled uniformly with constant ratio.  
디자인 공간을 더 줄이는 방법으로 모든 레이어는 균일한 상수 비율로 스케일 되어야 하는 제한을 걸었다

Our target is to maximize the model accuracy for any given resource constraints, which can be formulated as an optimization problem
우리의 목표는 최적화 문제를 공식화할 수있는 어떤 자원제약을 주어 모델의 정확도를 최대화 하는 것이다

(2)
$$
\underset{d,w,r}{max}\ \  Accuracy(\mathcal{N}(d,w,r))
$$
$$
s.t.\ \ \mathcal{N}(d,w,r) = \underset{i=1\cdots s}{\bigodot}\hat{\mathcal{F}}^{d\cdot \hat{L}_i}_i(X\langle r\cdot \hat{H}_i,r\cdot \hat{W}_i,r\cdot \hat{C}_i \rangle)
$$
$$
\mathrm{Memory}(\mathcal{N}) \le \mathrm{target\_memory}
$$
$$
\mathrm{FLOPS}(\mathcal{N}) \le \mathrm{target\_flops}
$$

where w,d,r are coefficients for scaling network width, depth, and resolution; $\hat{\mathcal{F}}_i,\hat{L}_i,\hat{H}_i,\hat{W}_i,\hat{C}_i$ are predefined parameters in baseline network(see Table 1 as an example)  
wdr은 네트워크의 스케일 계수이고 위 다섯 변수는 베이스라인 네트워크에서 사전 정의된 파라메터이다

## 3.2. Scaling Dimensions
The main difficulty of problem 2 is that the optimal $d,w,r$ depend on each other and the values change under different resource constraints.   
(2)번 문제의 가장 어려운점은 dwr의 최적화가 각각 의존하고있고 다른 자원 제약하에서 값이 바뀌는 것이다

Due to this difficulty, conventional methods mostly scale ConvNets in one of these dimensions :  
이 문제 때문에 관습적인 방법이 대부분 이런 차원중 하나로 확장을 한다

### Depth ($d$)
Scaling network depth is the most common way used by many ConvNets.  
깊이를 조정하는것은 많은 콘브넷에서 가장 흔하게 사용되는 방법이다.

The intuition is that deeper ConvNet can capture richer and more complex features, and generalize well on new tasks.   
직감적으로 깊은 콘브넷은 많고 더 복잡한 피쳐를 잡아낼 수 있고 새로운 작업에 대한 일반화가 잘된다

However, deeper networks ard also more difficult to train due to the vanishing gradient problem.  
하지만 깊은 네트워크는 그라디언트 소실 문제로 학습하기 더 어렵다

Although several techniques, such as skip connections and batch normalization, alleviate the training problem, the accuracy gain of very deep network diminishes : for example, ResNet-1000 has similar accuracy as ResNet-101 even though it has much more layers.  
스킵커넥션이나 배치 노말같은 이 문제를 완화시키는 몇몇 테크닉이 있지만 정확도는 매우깊은 네트워크에선 감소한다. 예를들면 레즈넷1000은 레즈넨 101과 비슷한 정확도를 가진다 매우 많은 레이어를 가졌음에도 불구하고.

Figure 3(middle) shows our empirical study on scaling a baseline model with different depth coefficient $d$, further suggesting the diminishing accuracy return for very deep ConvNets.  
그림3은 베이스모델을 d에 따른 확장한 실질적인 연구를 보여주고 매우 깊은 컨브넷에서의 정확도 감소를 잘 보여준다 

### Width
Scaling network width is commonly used for small size models.  
네트워크의 너비를 조정하는것은 작은 사이즈 모델에서 매우 흔하게 사용된다

As discussed in (), wider networks tend to be able to capture more fine-grained features and are easier to train.   
넓은 네트워크는 잘 정제된 피쳐를 캐치하는 경향이 있고 학습이 쉬워지게 한다

However, extremly wide but shallow networks tend to have difficulties in caputring higher level features.  
그러나 과하게 넓어넓지만 얕은 네트워크는 고수준 피쳐를캡쳐하기 어려운 경향이 생긴다.

Our empirical results in Figure 3 (left) show that the accuracy quickly saturates when networks become much wider with lager $w$  
우리의 실증적 결과는 그림 3에서 보여준다
넓이가 w보다 많이 넓어질때 정확도를 빠르게 포화시키는것을

### Resolution

With higher resulution input images, ConvNets can potentially capture more fine-grained patterns.
더 높은 해상도의 이미지와 함께하면 콘브넷은 더 좋은 패턴을 잘 캡쳐할것이다.

Starting from 224x224 in ConvNets, modern Convnets tend to use 299x299 or 331x331 for better accuracy.  
224에서 시작하여 현재는 더높은정확도를 위해 299 혹은 331 도 사용하는 경향이 있다

Recently, GPipe achieves state-of-the-art ImageNet accuracy with 480x480 resulution.  
최근 GPipe는 최고기록 달성을 위해 480을 적용했다

Higher resulutions, such as 600x600, are also widely used in object detection ConvNets.  
600처럼 더높은 해상도 또한 오브젝트 디텍션에서 널리 사용된다


Figure 3 shows the results of scaling network resolusions, where indeed higher resolutions improve accuracy, but the accuracy gain diminishes for very high resolutions ($r=1.0$ denotes resolution 224x224 and $r=2.5$ denotes resolution 560x560)  
그림3은 해상도의 조절 결과를 보여준다
실제로 높은 해상도는 정확도를 향상시켰으나 매우 높은 해상도에서는 정확도의 증가가 줄어들었다.
(1은 224, 2.5는 560을 의미한다)


##

The above analyses lead us to the first observation  
위의 관측은 우리를 첫번째 관찰로 이끈다

### Observation 1
Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.
3가지 차원을 확장하는것은 정확도를 향상시키지만
모델이 커질수록 정확도 증가량은 감소한다.

## 3.3. Compound Scaling 
We empirically observe that different scaling dimensions are not independent.

Intuitively, for higher resolution images, we should increase network depth, such that the larger receptive fields can help capture similar features that include more pixels in bigger images.  
직관적으로 높은 해상도의 이미지에서 네트워크의 깊이를 증가시켜야한다 넓은 수용 공간은 큰 이미지의 많은픽셀을 가진 비슷한 피쳐들을 잘 캐치하도록 도와줄수있다

Correspondingly, we should also increase network width when resolution is higher, in order to capture more fine-grained patterns with more pixels in high resolution images.  
비슷하게, 해상도가 올라갈땐 네트워크의 너비또한  증가시켜야 하는데, 고생도 이미지에에 많은 픽셀에서 더 좋은 정제된 패턴을 캐치하기 위해서이다

These intuitions suggest that we need to coordinate and balance different scaling dimensions rather than conventional single-dimension scaling.  
이 직관은 우리가 기존의 1개차원을 스케일하는것보단 차원들의 균형있게 조정을 해야하는것을 제안한다.

To validate our intuitions, we compare width scaling under different network depths and resolutions, as shown in Figure 4.   
우리의 직관을입증하기위해 그림 4에서 보여주듯이 다양한 네트워크에서 너비를 조정하는것을 비교한다.

If we only scale network width $w$ without changing depth ($d=1.0$) and resolution ($r=1.0$), the accuracy saturates quickly.
만약 w를 d와 r을 고정하고 변화시키면 정확도는 빠르게 수렴한다

With deeper ($d=2.0$) and higher resolution ($r=2.0$), width scaling achieves much better accuracy under the same FLOPS cost. 
깊은모델과 고해상도와 함께하면, 같은 FLOPS 비용으로 더 높은 정확도를 달성한다. 

These results lead us to the second observation :
이결과는 우리를 두번째 관측으로 이끈다.

### Observation 2
In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.  
더 좋은 정확도와 효율성을 추구하기 위해서는 컨브넷을 확장할때 3가지 차원의 밸런스가 매우 중요하다.

In fact, a few prior work have already tried to arbitrarily balance network width and depth, but they all require tedious manual tuning
사실상 몇몇 기존의 작업들은 이미 임의로 깊이와 뎁스의 균형을 시도하였지만 지루한 정해진 튜닝이 필요하다.

In this paper, we propose a new compound scaling method, which use a compound coefficient $\phi$ to uniformly scales network width, depth, and resolution in a principled way :  
이 논문에서 새로운 복합 스케일 메소드를 제안한다
합성계수 $phi$를 사용해 균일하게 네트워크의 차원을 조정하는. 아래 의 법칙에 의하여.

(3)
$$ \mathrm{depth:} d =\alpha^\phi \\
 \mathrm{width:} w =\beta^\phi \\
 \mathrm{resolution:} r =\gamma^\phi \\
 \mathrm{s.t.}\quad\alpha\ \cdot\ \beta^2\ \cdot\ \gamma^2 \approx 2\\
\alpha\ge1,\ \beta\ge1,\ \gamma\ge1 $$

where $\alpha, \beta, \gamma$ are constants that can be determined by a small grid search.  
세 변수는 작은 그리드 서치로 결정할수 있다.


Intuitively, $phi$ is a user-specified coefficient that controls how many more resources are available for model scaling, while $\alpha, \beta, \gamma$ specify how to assign these extra resources to network width, depth, and resolution respectively.
직관적으로 파이는 사용자지정 계수로 모델을 스케일하기위해 추가 리소스를 각각 3가지에 할당하는 방법을 명시한다

Notably, the FLOPS of a regular covolution ops is proportional to $d, w^2, r^2$, i.e., doubling network depth will double FLOPS, but doubling network width or resolution woll increase FLOPS by four times.
특히 규칙적인 컨볼루션 에서의 FLOPS는 3가지 변수에 비례한다 다시말하면 두배의 깊이의 네트워크는 FLOPS를 두배로 올리고, 두배의 너비와 해상도는 FLOPS를 네배로 올린다.

Since convolution ops usually dominate the computation cost in ConvNets, scaling a ConvNet with equation 3 will approximately increase total FLOPS by $(\alpha\ \beta^2\ \gamma^2)^\phi$.  
컨볼루션 부분이 대부분의 계산비용을 지불하므로 방정식3을 사용하여 컨브넷을 확장하면 토탈 FPOPS가 약 $(\alpha\ \beta^2\ \gamma^2)^\phi$ 만큼 오른다

In this paper, we constraint $\alpha\ \cdot\  \beta^2\ \cdot\ \gamma^2 \approx 2$ such that for any new $\phi$, the total FLOPS will approximately incrtease by $2^\phi$.
이 논문에서 우리는 새로운 어떤 파이에서도 위 방정식으로 제약을하면 총 FLOPS는 약 $2^\phi$ 증가한다


# 4. EfficientNet Acchitecture

Since model scaling does not change layer operations $\hat{\mathcal{F}}_i$ in baseline network, having a good baseline network is also critical.   
모델확장은 베이스라인 네트워크에 있는 레이어는 바꾸지 않기 때문에 좋은 베이스라인 모델을 고르는것 또한 매우 중요하다

We will evaluate our scaling method using existing Convnets, but in order to better demonstrate the effectiveness of our scaling method, we have also developed a new mobile-size baseline, called EfficientNet.  
우리의 확장 메소드를 현존하는 컨브넷을 통해 평가할 것인데 스키일링 방법의 효과를 더 잘 보여주기위해 에피션넷이라 불리는 새로운 모바일 사이즈 베이스라인을 개발하였다.

Inspired by (Tan et al., 2019), we develop our baseline network by leveraging a multi-objective neural architecture search that optimizes both accuracy and FLOPS. 
()에서 영감을 받았는데, 베이스라인 네트워크는 정확도와 FLOPS 양쪽의 최적화를 찾는 다목적 신경 구조를 활용하였다 

Specifically, we use the same search space as (Tan et al., 2019), and use $ACC(m) \times [FLOPS(m)/T]^w$ as the optimization goal, where $ACC(m)$ and $FLOPS(m)$ denote the accuracy and FLOPS of model $m$, $T$ is the target FLOPS and $w=-0.07$ is a hyperparameter for controlling the trade-off between accuracy and FLOPS.  
구체적으로 우린 위 논문과 같은 검색 공간을 사용하였다
위 식에서 $ACC(m)$ 와 $FLOPS(m)$는 우리 모델의 정확도와 FLOPS, T 는 목표 FLOPS이고 $w$는 정확도와 FLOPS를 Trade-off 하는 하이퍼 파라미터이다

Unlike(Tan et al., 2019; Cai et al., 2019), here we optimize FLOPS rather than latency since we are not targeting any specific hardware device.  
위 논문과는 다르게 루린 특정한 하드웨어장치를 대상으로 하지 않기 때문에 대기시간보단 FLOPS를 최적화 하였다.

Our search produces an efficient network, which we name EfficientNet-B0.

Since we use the same search space as (Tan et al., 2019), the architecture is similar to MnasNet, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M)
같은 검색공간을 사용했기 때문에 MnasNet과 비슷한 아키텍쳐를 가지고 있고, 에피션넷을 제외하고는 점차적으로 커지는  

Table 1 shows the architecture of EfficientNet-B0.  
테이블1은 에피넷B0의 구조를 보여준다

Its main building block is mobile inverted bottleneck MBConv, to which we also add squeeze-and-excitation optimization.  
메인 블록의 구성은 인버트바틀넥을 사용한 BMConv, 그리고 Squeeze-and-excitation 최적화를 더하였다.

Starting from the baseline EfficientNet-B0, we apply our compound scaling method to scale it up with two steps :  
베이스라인 에피션넷으로 시작해 우리의 메소드를  두가지 스텝에 따라 적용하였다

- STEP 1 : we first fix $\phi=1$, assuming twice more resources available, and do a smamll grid sarch of $\alpha,\beta,\gamma$ based of Equation 2 and 3. In particular, we fine the best values for EfficientNet-B0 are $\alpha=1.2, \beta=1.1, \gamma=1.15, under constraint of $\alpha\ \cdot\ \beta^2\ \cdot\ \gamma^2 \approx 2$.  
우선 파이를 1로 고정하고 두배의 자원이 더있다고 가정하였다, 그리고 방정식 2와 3을따라 작은 그리드서치를 하였다. 특히 우리는 위와같은 최고의 값을 찾았다.

- STEP 2 : we then fix $\alpha, \beta,\gamma$ as constants and scale up baseline network with different $\phi$ using Equation 3, to obtain EfficientNet-B1 to B7 (Details in Table 2)  
다음으론 알벳감을 고정하고 방정식 3을따라 다른 파이를주면서 스케일을 확장했다. 그래서 B1부터 B7까지 얻었다(표2에 설명)


Notably, it is possible to achieve even better perfomance by searching for $\alpha,\beta,\gamma$ directly around a large model, but the search cost becomes prohibitively more expensive on larger models.
특히 더 좋은 퍼포먼스를 이루기 위해 큰모델에서 직접 알벳감을 찾는것도 가능하지만 큰 모델에서는 하면 안될정도로 큰 비용이 들어간다 

Our method solves this issue by only doing search once on the small baseline network(step1), and then user the same scaling coefficients for all other models(step2)  
우리는 이 방법을 작은 베이스라인 네트워크에서 찾는것으로 이 문제를 해결했고 같은 확장계수를 이용해 다른모델을 만들었다.



























