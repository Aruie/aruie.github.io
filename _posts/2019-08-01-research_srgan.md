---
layout: post
title:  "Research and Improvement of Single Image Super Resolution Based on Generative Adversarial network"
categories: paper
date : 2019-08-01 10:05:05 +0900
comments: true
---

SRGAN 의 점수 감소로 인해... ㅠㅠ 여러가지 테스트하긴 시간이 부족하고
점수를 올려 줄 수있는 방법을 찾던중 새로운 논문 발견!
일단 내용은 SRGAN의 향상에 대한 연구로 봐야겠다.
결과를보면 PSNR 점수가 비약적으로 상승한걸로 봐서 이거다 싶어서 보기시작.
시간상 일단 중요한 부분만 보자 ㅠㅠ

## 2.1 Improved model structure

네트워크의 깊이와 너비를 증가시키는것은 네트워크의 성능을 향상시킬수 있다. 많은 실험이 같은 상황에서 깊고 얕은 네트워크를 가지고 실험을 했는데 깊은네트워크의 퍼포먼스는 는 일반적으로 좋지 않았다. 얕은 네트워크의 파라미터는 는 뒤에 1층만 남긴채 깊은 네트워크의 앞의 몇몇층이 옮겨올 수 있다. 이런 맵핑은 얕은 네트워크의 학습 효과를 볼수 있다. 예를들면 흔한 VGG 구조는 AlexNet 네트워크의 깊이를 증가시키는 것으로 성능을 비약적으로 향상시켰다. 그러나 단순하게 네트워크의 깊이를 증가시키는 것은 기울기 소실 혹은 발산을 야기한다. 이 문제에서 네트워크의 레이어중 몇개는 정규화와 배치노말을 통해 학습될 수 있다.
그러나, 새로운 성능저하 문제가 발생하는데, 레이어가 증가하면 training 셋의 정확도는 포화하거나 심지어 감소한다. 이것은 오버피팅으로는 이해 할 수 없는데 왜냐면 오버피팅은 학습셋에서는 더 좋은 성능을 가져오기 때문이다.  

He에 의해 잔차 네트워크와 뛰어넘는 연결이 제안되면서, 네트워크구조가 더 깊을때 쉽게 학습되도록 만들어지고, 인식 정확도는 레이어가 증가함에 따라 향상 될 수 있었다. residual block 기존 구조는 Figure2에 나와있다. 잔차네트워크는 기본 컨볼루션 레이어에 숏컷을 추가하고 skip connection 을 연결하여 기본 residual block을 만든다. 기존의 H(x) 는 H(x) = F(x) + x 로 표현한다. 잔차구조는 
H(x) 학습을 F(x)학습으로 전환하고 F(x) 를 학습하는것은 H(x)를 학습하는것보다 쉽다.  잔차구조는 깊은 네트워크에서의 degradation 문제를 효과적으로 완화시키고 residusl block 을 레이어처럼 축척하는것을 통해 네트워크의 성능을 향상시킨다.

원래의 SRGAN 생성 모델은 몇개의 residual block 을 포함하고 있다. 다수의 배치노말 레이어가 residual 구조에 사용되고있다. 배치노말 레이어는 학습시 배치단위의 평균과 분산을 사용하여 피쳐를 normalize 하고 테스트시엔 전체 학습 데이터의 추정된 평균과 분산을 사용한다. 학습과 테스트의 통계적 지표가 매우 다를경우 배치노말은 모델의 일반화성능을 제한하는 불편한 물건이 된다. 실험들은 배치노말을 제거하는것이 복잡도와 메모리사용, 그리고 모델의 일반화 성능을 향상 시키는것을 보여준다.

네트워크 깊이 측면에서, literature[4-6] 은 깊은 네트워크가 높은 복잡도의 맵핑을 구축하고 엄청나게 네트워크 정확도를 향상시킬 수 있는것을 보여준다. 이 논문에서 원래 모델의 생성기의 residual block을 32개로 증가시켰다. 실험에서는 fitting 현상은 발생하지 않앗고 모델은 어느정도 향상되었다. 개선된 구조는 Figure 3에서 보여준다.


## 2.2 Improved loss function

원래 SRGAN의 손실함수는 두부분으로 나뉜다. VGG기반의 constent loss와 적대적 모델 기반의 countermeasure loss. $D_{\theta D}$ 는 실제 고해상도 이미지에 해당하고, $G_{\theta G}(I^{LR})$ 재구성된 고해상도 이미지이다. 이미지의 텍스쳐 디테일은 향상되었지만 PSNR 과 SSIM 은 좋지 않았다.

 This improves the texture details of the image, but the PSNR and SSIM 지표는 좋지 못했다. 이 논문에서 MSE 기반의 손실함수를 추가하면서 손실함수를 향상시켰다. 그 이유는 VGG의 재구성 효과는 다른 레이어마다 다르고, 여기에 또 가중치 상수를 더했다 VGG 함수에. 
 향상된 손실함수는 다음과 같다. 

$$
 l^{SR} = l^{SR}_{VGG} + 10^{-3}l^{SR}_{Gen} \\

 l^{SR}_{VGG(i,j)} = \frac{1}{W_{i,j}H_{i,j}} \sum\sum(\Phi_{i,j}(I^{HR})_{x,y}-\Phi_{i,j}(G_{\theta G}(I^{LR}))_{x,y})^2 \\

 l^{SR}_{Gen} = \sum -\log D_{\theta D}(G_{\theta G}(I^{LR})) \\

 l^{SR}_{MSE} = \frac{1}{r^2WH}\sum\sum((I^{HR})_{x,y}-(G_{\theta G}(I^{LR}))_{x,y}) \\

 l^{SR} = i^{SR}_{VGG}+\alpha i^{SR}_{MSE} + \beta i^{SR}_{Gen}
$$


# 3. Experiment






