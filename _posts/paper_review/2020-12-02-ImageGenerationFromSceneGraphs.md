---
layout: post
title:  "Image Generation from Scene Graphs"
tag : "Paper Review"
date : 2020-12-02 10:05:05 +0900
comments: true
---



# Abstract
- 우리 모델은 단순 인식 뿐만이아닌 생성을 해낸다
- 이것의 끝은 natural anguage description 으로부터 이미지를 생성하는 작업이다
- 새나 꽃같은 제한된 도메인에서 놀라운 결과가 나오지만, 많은 오브젝트와 관계를 가진 ???
- 이 한계를 극복하려고 오브젝트와 관계를 명시적으로 추론 할수 있는 Scene graph에서 이미지를 생성하는것을 제안했다
- GCN을 사용하여 Graph를 입력받고 예측된 박스와 Segmentation 마스크로 Scene layout을 계산하고 이것을 이미지로 생성하는 연속적인 네트워크다
- 한쌍의 discriminator의 통해 적대적으로 학습된다고?




# 2. Related Work
- 현재 이미지 생성모델 크게 3가지로 분류된다
- GAN, VAE, Autoregressive approaches
- Conditional Image Synthesis
  - conditional 한 생성은 추가적인 입력을 Generator와 Discriminator양쪽에 넣거나 Discriminator에만 label을 넣기도 하는데 우리는 후자를 사용한다
  - 기존연구에서 text를 GAN을사용하여 이미지를 생성하거나 그것을 발전시켜 multistage 생성을 하게한 연구도있음
  - 우리연구와 관련 깊은 연구에는 sectence와 keypoint? 양쪽을 사용한 조건부 생성모델, 그리고 생성외에도 관측되지 않은 중요한 키포인트를 예측하기도 한다
  - Sementic Segmentation을 기반으로 percepual feature reconstruction loss와 Cascaded refinement network 를 사용 하여 고해상도 거리 이미지를 생성하였다
    - 이 CRN 구조를 사용 함
- Scene Graphs
  - 란 노드는 오브젝트를, 엣지는 관계를 표현하는 방향성 그래프로 이미지 검색이나 캡셔닝에 주로 활용되어 왔다
  - 문장을 씬그래프로 변환하거나 이미지에서 씬그래프를 예측하는 연구들도 있다
  - 대부분 연구에서 사람이 만든 주석이 달린 Visual Genome datasets을 사용한다
- Deep Learning on Graphs
  - 어떤 고정된 그래프에서 노드를 임베딩 하는 방법들이 있으나 우린 그것과 다르게 각 Forword마다 다른 그래프를 처리해야한다
  - GNN은 우리가 원하는 방향과 비슷하게 임이의 그래프에서 작동한다
  - molecular property prediction, program verification, modeling human motion 등에서 사용되고 spectral domain을 활용한 기법들도있으나 우린 사용안함

# 3. Method
- 우리 목표는 오브젝트와 관계를 설명하는 Scene graph를 입력하여 그 그래프와 관련된 Realistic한 이미지를 생성하는것이다
  - 세가지 벽이 있는데 첫째로 그래프 구조의 입력을 위한 처리방법이고
  - 두번째는 생성된 이미지가 얼마나 그래프를 잘 설명하는지 확인해야하고
  - 세번째는 생성된 이미지가 얼마나 사실적인지 확인해야한다
- $\hat{I} = f(G,z)$
  - 그래프에서 이미지를 생성하는 네트워크$f$
  - 생성된 이미지 $I$, Scene Graph $G$
  - $G$는 GCN에 의해 각 오브젝트마다 임베딩된다
  - 이 과정에서 엣지의 정보가 섞인다
- GCN에서 나온 object embedding vector를 사용해 각 object의 박스와 segmentation mask를 예측하고 이것을 합쳐 Scene layout을 구성, 스케일을 확대하는 CRN을 통해 최종적으로 이미지를 생성한다
- 한쌍의 Discriminator $D_{img}, D_{obj}$를 통해 적대적학습을 하여 현실적인 그림인지와, 현실적인 오브젝트를 담고있는지를 판단하게한다
- 이게 전체적인 그림
- $(O,E)$ : Scene Graph
  - $O = \{o_1, o_2,...,o_n\}$ : object
    - $o_i$ : i번째 object class
    - 