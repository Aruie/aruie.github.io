---
layout: post
title:  "Training Generative Adversarial Networks with Limited Data"
tag : "Paper Review"
date : 2020-10-28 10:05:05 +0900
comments: true
---


# Abstract
 - GAN을 학습할때 데이터가 너무 적을경우 Discriminator가 overfitting되어 훈련이 잘 안됨
 - 이런 제한된 데이터에서의 훈련에 안정성을 현저히 높여주는 Adaptive discriminator augmentation mechanism 을 제안
   - 모델의 구조 혹은 Loss를 변경하지않고 from scratch 와 fine-tuning 모두 적용 가능
   - 수천단위의 데이터만으로 좋은 결과를 냄
 - 새로운 GAN의 domain을 열었다고 생각함

# 1. Introduction
 - 최근 GAN 모델들은 성능은 좋았지만 대부분 온라인을 이용한 매우 많은 이미지를 사용하여 학습하였음
   - 하지만 라벨링이 필요업다 해도 이런 큰 데이터를 만든는것은 여전히 어려움
   - 특히 완전히 새로운 데이터에 적용 할 경우 $10^5 \sim 10^6$ 개의 데이터를 생성해야함
     - 덕분에 의학쪽에서 잘 이용이 되지 못하고 있다고 함
   - 만약 필요 이미지의 수를 줄인다면 다양한 응용프로그램에 도움이 될 수 있을 것
 - 핵심적인 문제는 Discriminator가 학습데이터에 overfit 되는것
   - Discriminator가 의미없는 정보만 주게되어 학습이 발산해버림
   - 이미지 분류에서 Data Augmentation 은 이것을 해결하는 표준 솔루션으로 사용됨
   - 하지만 Generator의 경우 이것이 Augmentation에 대한 분포를 학습할 수 있음
     - 예를 들면 노이즈가 추가될 경우 원래 데이터엔 없던 노이즈도 생성함 
     - 여기서 leak이라는 용어가 사용되는데 augmentation 된 정보가 Generator에 흘러가는 것을 뜻함
 - Discriminator가 overfit 되지 않는 새로운 augmentation방식을 적용
   - leak을 방지하기 위해 이것이 발생하는 조건을 분석
   - 최종적으로 데이터와 훈련설정에 의존하지 않는 General 한 방식을 설계

# 2. Overfitting in GANs
  - 데이터의 양이 GAN 에 미치는 영향 연구
    - 설정 
      - FFHQ 및 LSUN CAT 데이터에서 샘플링 
      - 광범위한 Hyperparameter 설정을 위해 다운샘플링한 $256 \times 256$ 버전을 사용
      - 생성된 50k 개의 이미지와 모든 실제 이미지 사이의 FID를 기준으로 평가
    - 결과(FFHQ)
![Table1](/assets/post/201028/1.png)
      - 여러가지 설정에서 특정구간 이후 전부 FID가 오르는것을 보임
      - 오른쪽 b,c 를보면 처음엔 어느정도 겹치지만 Discriminator가 과적합되면서 완전히 분리됨
      - 특히 Validation이 생성된 이미지쪽으로 완전히 치우는것을 보임 (완벽한 과적합의 신호)
      - 이것을 해결하는 augmnetation 방식을 설계하는것이 목표

## 2.1 Stochastic Discriminator Augmentation
![Table1](/assets/post/201028/2.png)
  - balanced consistency regularization (bCR)
    - Discriminator 학습시 동일한 증강을 부여하고 동일한 출력이 나오게 하는 방식 (CR손실 추가)
    - Generator 학습시엔 panalty가 적용되지 않아 자유롭게 생성 가능해 leak이 발생해 그냥 augmentation을 사용하는것과 유사한 결과가 나옴
  - Discriminator augmentations (DA)
    - 기본적으로 방식은 유사하나 새로운 손실항을 추가하지않고 Generator를 학습할때도 augmentation을 수행
    - 간단하지만 현재까지 주목받지 못한 방법이라함
      - 하지만 구조가 직관적으로 봐도 매우 불안정 할수밖에 없기에 이것이 잘 될거라 생각 자체를 못했을거라 말함
      - leak이 발생하는 조건을 모두 찾은 후 완전한 구조를 구축

## 2.2 designing augmentations that do not leak
  - DA 방식의 문제
    - DA 방식은 왜곡시킨 제품을 보여주고 원래 제품을 가져와라 라고 하는 것
    - AmbientGAN: Generative models from lossy measurements(Bora et al.) 에서는 corruption 과정이 invertible transformation 이라면 훈련이 암묵적으로 왜곡을 해결하고 원래의 분포를 찾는다는것을 발견 
    - 우리는 이런 augmentation 기법을 non-leaking 이라 부르고 이 기법들은 적용된 데이터만으로 기존 데이터와의 일치여부르 구분 할 수 있는 능력을 가짐
      - 이미지픽셀을 90프로확률로 0으로 만드는것은 invertible 함, 사람도 10프로의 이미지만으로도 90프로의 검은부분을 무시하고 추론할 수 있다
      - 하지만 90도 단위 랜덤 회전변환의 경우 실제이미지의 위치가 어디였는지 알수없기에 원복이 불가능하지만 확률 이 1보다 작아진 다면 올바른 형태의 확률이 높아지기에 원래 모양을 예측할 수있음
    - 여러가지 augmentation이 확률이 1미만인 조건에서 non-leaking 해질 수 있음

    - 고정된 순서로 non-leaking 증강을 구성하면 전체적인 non-leaking 이 가능해짐


![Table1](/assets/post/201028/3.png)
  - Isotropic image scaling 의 경우 확률에 상관없이 결과가 균일 (non-leaking)
  - 하지만 Random Rotation의 경우 확률이 높으면 generator는 랜덤한 하나를 선택하게됨 (확률이 1일때만 발생하는것은 아님)
  - 이 확률이 낮아질수록 처음엔 잘못 선택하더라도 높은 확률로 올바른 분포를 찾아감
  - Color transformations 도 0.8이전엔 균일한 분포를 보임

## 2.3 Out augmentation pipeline
  - RandAugment 에서 봤듯이 다양한 augmentation을 적용하는 것이 좋다는 가정에서 시작
  - augmentation을 6개의 카테고리로 나눔
    - pixel blitting (x-flips, rotation, integer translation)
    - more general geometric transformations
    - color tranforms
    - image space filtering
    - additive noise
    - cutout
  - augmentation을 구분할 수 있도록 Generate된 이미지에도 augmentation 적용
    - 제공되지 않는것은 딥러닝 프레임웍에서 제공되는 미분이 가능한 primitives로 구현

  - 훈련중 모든 변환에 대해 항상 동일한 p 값을 사용하는 사전 정의 세트를 사용
    - 변환에 대해 확률은 독립적으로  적용
  - 고정된 순서 사용
  
  - p값은 augmentation와 미니배치 내 이미지에 의해 조절
  - pipeline에 매우많은 증강이 있어 p가 상당히 작아도 discriminator가 원래 이미지를 볼 가능성은 매우 낮지만(Figure 2 참조) Generator는 p값의 경계가 지켜지는 한 깨끗한 이미지만을 생성 


![Table1](/assets/post/201028/4.png)

 - 여러가지 DA들은 결과를 크게 개선하는것을 보이지만 적절한 p값은 데이터양에 크게 의존함
 - (a, b) : 소규모 데이터 (2k, 10k)
   - 높은 확률에서 최적의 결과가 나오고 geom 계열이 좋은 성능을 보임
 - (c) : 대규모 데이터 (140k)
   - 모든 증강효과가 전부 역효과이며 leaking 효과가 관찰됨
 - (d) : 10k의 데이터로 수렴결과를 보면 수렴도 늦추고 과적합도 방지하는것을 보여줌
 - 데이터 세트에 민감하다는것은 적절한 p값을 찾기 위해 비용이 큰 grid search가 필요하다는 것이기에 adaptive한 프로세스를 만들어 이것을 해결

## 3. Adaptive discriminator augmentation
- 수동튜닝을 안하고 과정합정도에 따라 동적으로 제어해보자는 아이디어
  - 첫번째로 일반적으로 과적합을 확인하는 방법인 validation을 따로 나누어 훈련셋을 관찰하는것
  - 두번쨰는 StyleGAN2에서 사용되는 non-saturating loss를 관찰하는 방법
- 세부 설정
  - 연속된 4개의 미니배치를 한번에 봄 (256개의 이미지)
  ![Table1](/assets/post/201028/f1.png)
  - 두개 식 모두 r이 0이면 과적합없음, 1이면 완벽한 과적합을 의미
- 목표는 값이 적절한 목표를 달성하도록 p 를 조절하는것
  - 첫번째식은 validation set이 있을때 validation 과 generate에 대한 차이
  - 두번째식은 Discriminator 출력에 양성이 나오는 훈련세트부분을 추정
    - 첫번째 식보다 하이퍼파라미터에 덜 민감함 

- p를 0으로 설정하고 4개의 미니배치마다 한번씩 조정
  - p값이 충분히 빠르게 변할수 있도록 조정
  - 최소는 0이 되도록 제한
- 이것을 Adaptive discriminator augmentation 이라고함


![Table1](/assets/post/201028/5.png)


- rv와 rt에 다양한 목표를 주고 학습
  - grid search를 사용한것 보다 훨씬 효과가 좋았음
  - 둘다 성능은 괜찮지만 좀더 현실적인 rt를 사용 (목표값 0.6)


![Table1](/assets/post/201028/6.png)

- 적용후 여러 데이터에 적용한 결과
  - 오버피팅이 전혀 보이지 않음
- augmentation 이 없을 시 학습이 길어지면 discriminator 가 overfitting되어 몇가지 특징만으로 구분하게되어 gradient가 전체적으로 단순화 됨
  - 이상황이 되면 Generator가 너무 자유롭게 생성을 하게됨
  - ADA가 적용될시 gradient 가 훨씬 더 잘 유지 되는것이 보임

  

# 4. Evaluation

- From Scratch 와 transfer learning 두 방식으로 FFHQ와 LSUN Cat을 사용해 적용

## 4.1 Training from scratch
![Table1](/assets/post/201028/7.png)
- 소규모 데이터에서 ADA가 매우 효과적인 것이 보임
- bCR도 데이터가 어느정도 충분할 때는 효과적이나 leak이 발생되는것이 보임
  - x-y translation만을 사용하였으나 이미지가 blurring 되는것을 보임
  - bCR은 기본적으로 augmentation이기에 학습에 도움이 되는 augmentation만 사용 가능
- 두가지를 더해서 사용하였을때 큰 효과를 보임
  - ADA를 먼저 적용 후 bCR은 독립적으로 자체 증강 방식 사용

![Table1](/assets/post/201028/88.png)
- shallow mapping에 적용? 이부분은 모르겠네요
- 증강을 Multiplicative Dropout 으로 대체하려 하였으나 별로안좋음
 - p 값은 adaptive algorithm 적용
- 결론은 ADA가 좋음


## 4.2 Transfer learning
![Table1](/assets/post/201028/9.png)
- GAN에도 Transfer leaning이 적용되는 연구가 많은데 특히 Freesze-D 방식이 좋은 결과를 보였음
- Transfer leaning 의 결과는 데이터셋의 유사성보단 다양성에 달려있는 것을 보임
- baseline의 경우 급격한 FID 감소를 보이나 과적합으로 다시 발산
- 역시 ADA를 사용시 안정적인 모습을 보임
  - Freeze-D 도 같이 사용시 약간 개선이 보이지만 단독으로는 과적합을 막지 못함

## 4.3 Small datasets 
![Table1](/assets/post/201028/10.png)
- 작은 데이터셋에서는 FID가 편향으로 인해 좋은 지표가 되지 못하다고 함




# 5. Conclusions
- ADA가 효과적이긴 했으나 실제 데이터를 완벽히 대체하진 못함 (데이터를 잘 모으자)
- U-net Discriminator, Multi modal Generator 등도 효과가 있는지 확인할 가치가 있음








