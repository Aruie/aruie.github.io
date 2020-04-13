---
layout: post
title:  "EfficientDet : Scalable and Efficient Object Detection"
tag : "Paper Review"
date : 2020-04-11 10:05:05 +0900
comments: true
---


![Table1](/assets/post/200411/1-1.png)

# Abstract
 - 컴퓨터비전에서 점점 중요시되는 Efficiency를 위해 Object Detection 에서의 최적화 기법들을 제안
   - Bidirectional Feature Pyramid Network (BiFPN)
   - EfficientNet과 비슷하게 아키텍쳐내의 모든 레이어들을 균일하게 확장하는 스케일링방법 제안

# Related Work
 - 기존 모델 분류
   - 크게 1 혹은 2 Stage 모델로 나뉨
   - 2 Stage 방식
     - Region Proposal 과 Classification을 순차적으로 처리
     - R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN 등
     - 일반적으로 더 유연하고 정확한 결과가 나오지만 느림
   - 1 Stage 방식
     - Region Proposal 과 Classification을 동시에 처리
     - 사전 정의된 앵커를 사용해 간단하고 효율적임
     - YOLO 계열, RetinaNet 등
     - 이 논문에서는 이 방식을 채택

  
  - 기존의 Feature fusion 방식들
    - FPN : 다중 스케일 기능을 결합하는 하향식 경로 제안 
    - PANET  : 상향경로 집게 네트워크 추가
    - STDL : 스케일 전송모듈 제안
    - M2det : 다중스케일 기능 융합을 위한 U형 모듈 제안
    - G-FRNet : 피쳐간의 정보흐름을 위한 게이트 유닛 추가
    - NAS-FPN : Neural Architecture Search 를 사용해 네트워크설계(많은시간 필요, 해석 어려움)
    - 우리는 직관적이고 원칙적으로 다차원 피쳐의 융합을 최적화시킴

  - Model Scaling 
    - 기존엔 정확도를 위해 더큰 Backborn 모델을 사용하거나 채널크기를 늘리거나 피쳐간 연결작업을 반복해 정확도를 올림
      - 단일 혹은 제한된 차원 스케일링에 한정
    - EfficientNet에서 큰 영감을 받아 이부분을 복합적으로 효율적으로 올림


# BiFPN 
![Table1](/assets/post/200411/1-2.png)
## Problem Fomulation
  - MultisScale Feature Fusion Problem
    - 서로다른 해상도에서의 피쳐를 종합하는것이 목표
    - $P^{out} = f(P^{in})$
    - $P^{in} = \{P^{in}_{l_1}, P^{in}_{l_1}\, \cdots\}$
      - $l_i$ : i 번째 레이어의 피쳐
      - i번째 레이어의 해상도는 입력 해상도의 $1/2^i$ 
    - 예시
      - $P^{out}_7 = Conv(P^{in}_7)$
      - $P^{out}_6 = Conv(P^{in}_6 + Resize(P^{in}_7))$
## Cross-Scale Connections
  - 위의 FPN방식은 일방적인 정보의 흐름으로 제한됨
  - PANet에서는 이것을해결하기 위해 상향경로 네트워크를 추가함
  - Cross-Scale 방식에서 최적화를 제안
    - 입력이 하나밖에없는 노드는 피쳐융합을 목표로 하는 네트워크에서 큰의미가 없을거라 판단하여 제거
    - 적은비용으로 많은기능을융합하기위해 엣지 추가
    - 양방향 경로를 하나의 계층으로 취급하고 반복
      - 뒤에서 서로 다른 제약조건에서 레이어수를 정하는 방법에 대해 논의
## Weighted Feature Fusion
  - 여러 해상도에서의 융합에서 가장 일반적인 방법을 동일한 해상도로 조정 후 더하는것
  - 이전 방식들은 모든 입력치펴를 동일하게 처리하지만 다른 해상도에 있는 피쳐들이 출력 피쳐에 불평등하게 기여하는것을 관찰
  - 입력에 대한 추가적인 가중치를 넣어 피쳐의 중요성을 판단하게 만듬
### Unbounded fusion
  - Scale이 최소한의 계산비용으로 다른방법과는 비교할수 없는 정확도를 달성하는것을 발견
  - 피쳐, 채널, 픽셀별로 적용 가능
  - 범위의 제한이 없을경우 훈련에 불안정을 일으킬수 있어 가중치 정규화가 필요
### Softmax-based fusion
  - 가중치 정규화를 소프트맥스 기반으로 적용
  - 소프트맥스 레이어의 GPU 성능 저하 문제가 발생
### Fast normalized fusion
  - Softmax 방식에서 지수대신 합을 사용하여 비슷한 효과를 가짐
  - GPU 에서 30% 가량 더 빨리 실행됨

## 적용
- 위에 Fast normalized fusion과 최적화된 Bidirectinal cross scale connection을 적용함

# EfficientDet
- EfficientDet 이라는 모델 그룹을 소개

## EfficientDet Architecture
![Table1](/assets/post/200411/1-3.png)
- 이미지넷으로 학습된 EfficientNet 를 백본으로 사용
- 3번부터 7번까지의 레이어의 출력값을 입력으로 BiFPN을 반복적용
- 이 피쳐를 각각 Class 및 Box 를 예측하는 네트워크에 입력 (두 네트워크의 가중치는 공유됨)
## Compound Scaling
![Table1](/assets/post/200411/1-4.png)
- 기준이되는 EfficieintDet을 효율적으로 확장하는 방법 제시
- Backborn, BiFPN, Prediction network, 입력이미지의 해상도 4가지를 조절
- EfficientNet과는 달리 훨씬 많은 scale dimension을 가지므로 grid search가 힘들어 휴리스틱 기반 접근법을 사용(공동으로 스케일링하는 아이디어는 동일)
- 세부 과정
  - Backborn 
    - EfficientNet의 계수를 사용
  - BiFPN
    - 채널은 지수적으로, 깊이는 선형으로 증가
  - Prediction network
    - 채널은 BiFPN과 같게 유지하고 레이어는 선형적으로 증가
  - Input image resolution 
    - 7번째 레이어까지 사용하므로 128로 나눌수있는값으로 지정하고 기본은 512
  - 위 방법을 사용하여 모델을 제작하고 7이상은 메모리상 문제가 있어 7이상은 이공식으로 사용불가능해 6에서 인풋크기만 바꾼 7을 생성

# Experiments
 - COCO 2017 데이터셋 사용
 - 비교는 RetinaNet 및 AmoebaNet과 동일한 전처리를 사용
 - 전체적으로 정확도나 Flops에서 기존 모델보다 좋은 효율을 보임 ( 테이블 참조 )
 - 특히 특수설정이 필요한 amoabanet 기반 모델과 달리 병렬처리업이 동일한 3x3 컨볼루션만 사용하여 학습
 - CPU와 GPU 기반의 비교
