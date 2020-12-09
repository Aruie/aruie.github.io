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
- Scene Graph 의 구조
  - $(O,E)$
    - $O = \{o_1, o_2,...,o_n\}$ : object ($o_i \in C$)
    - $E \in O \times R \times O$
      - $(o_i, r, o_j)$ 의 형태를 가짐
  - 우선 학습된 Embedding Layer를 사용해 각 노드와 엣지를 $D_{in}$의 shape를 가진 dense vector로 변환 시킴
- Graph Convolution Network (일반적인 GCN과 매우다름)
  - $(v_i, v_r) \in D_{in}$ : 입력 shape, $(v_i', v_r') \in D_{out}$ : 출력 shape
  - $(v_i, v_r, v_j)$ : 실제 입력되는 형상
  - $g_s, g_p, g_o$ 모든 입력에 대해 3개의 함수를 사용
    - $g_p$ : Predicate (엣지 벡터 변환)
    - $g_s$ : Subject 
    - $g_o$ : Object
    - ![Table1](/assets/post/201202/table4.png)
  - $v_r' = g_p(v_i, v_r, v_j)$
    - 출력 엣지를 구함 (연산이 매우 간단)
  - $v_r' = h(V_i^s \cup V_i^o)$
    - $V_i^s = \{g_s(v_i,v_r,v_j) : (o_i,r,o_j) \in E \}$
    - $V_i^o = \{g_o(v_j,v_r,v_o) : (o_j,r,o_i) \in E \}$
    - 해당 오브젝트가 subject에 사용된 관계는 $g_s$, object로 사용된 관계는 $g_o$를 사용하여 연결

    - ![Table1](/assets/post/201202/table4.png)
    - $h$ 함수는 간단히 각 벡터의 위치별평균을 사용 후 FCL 두개 통과
    - 
- Scene Layout
  - GRAPH를 이미지로 만들기 위해서는 이미지 도메인으로의 변환이 필요
  
  - Object Layout Network
    - 2D 에서의 대략적인 Layout을 생성하기 위한 네트워크
    - Box Regrresion Network
      - ![Table1](/assets/post/201202/table6.png)
      - 이미지의 위치를 박스로 나타낼 네트워크로 object embedding 정보를 사용해 꼭지점의 위치를 예측
        - $(x_0,y_0,x_1,y_1)$, [0,1]로 Normalize된 상태로 좌상우하 위치
    - Mask Regression network
      - 해당 사각형에 이미지를 그려버리면 배경과의 조화가 이루어 질수 없으므로 투명도를 위한 마스킹 정보 생성 (조금 의문?)
      - ![Table1](/assets/post/201202/table7.png)
    - 생성된 마스크 $(1 \times M \times M)$ 을 embedding$(D)$ 에 곱해 $(D \times M \times M)$ 의 텐서를 생성 후 BoxRegression 정보와 결합하여 $(D \times M \times M)$ 의 최종 Scene Layout 생성
- Cascaded Refinement network (Generator)
  - 앞에서 생성된 Scene Layout에 의존한 의미지를 생성해야하는데 이것을 위해 CRN을 사용
  - 각 모듈마다 해상도가 두배씩 점차적으로 증가시키며 Scene Layout 정보(down sampling)와 이전 모듈에서의 출력을 channel wise 방식으로 연결
    - (여기서의 문제는 채널이 고정?)
    - Scene layout 정보는 모듈마다 맞는 사이즈로 downsampling 해서 사용
    - 각 모듈에서의 출력은 다음 입력으로 들어가기전 upsampling됨
    - 첫번째 모듈에서는 입력으로 Gaussian noise 사용
    - 최종 모듈 출력 이후 2의 마지막 컨볼루션 레이어 통과
- Discriminator
  - 두쌍의 Discriminator 사용 $(D_{img}, D_{obj})$
  - 둘 모두 기본적으로 Advasarial Loss를 사용
    - ${L}_{GAN} = \underset{x\sim p_{real}}\mathbb{E}logD(x) +  \underset{x\sim p_{fake}}\mathbb{E}log(1-D(x))$
  - $D_{img}$의 경우 Patch-based discriminator를 사용하여 이미지의 품질에 대해 보장
  - $D_{obj}$의 경우 각 객체가 해당 객체를 제대로 식별하였는지를 보장
    - 고정크기로 crop 및 rescaled된 이미지를 입력으로 사용
    - Auxiliary Classifier를 사용해 오브젝트의 카테고리까지 확인
- Training
  - 학습간에 6가지 loss에 대해 weighted sum을 사용하여 최종 loss를 계산
    - $L_{box} = \sum_{i=1}^n\|\|b_i - \hat{b_i}\|\|_1$ : 박스간 L1 loss
    - $L_{mask}$ : predict mask에 대한 pixelwise cross-entropy loss
    - $L_{pix} = \|\|I-\hat{I}\|\|_1$ : pixelwise L1 loss
    - $L_{GAN}^{img}$ : Patch based Discriminator loss
    - $L_{GAN}^{obj}$ : Object Discriminator loss
    - $L_{AC}^{obj}$ : Classification loss
  - 세부설정
    - Adam 사용(lr : 0.0001)
    - 1M epochs, 32 batch size
    - P100을 사용하여 3일정도 소요
    - 각 미니배치에 대해 CRN부터 학습 후 Discriminator 업데이트
    - GCN에선 ReLU, CRN과 discriminator는 LeakyReLU 및 batch normalization 사용

# 4. Experiments
  - ![Table1](/assets/post/201202/figure5.png)
    - 솔직히 해상도가 워낙구려서 좀 못알아보겠다
    - 일단 단순히 Scene 그래프만으로 같은 유형의 오브젝트가 여러개 있는것도 표현 가능한 것을 보임
    - 생성된 이미지가 Scene Graph에 매우 의존적인것도 보임
    - 맨 아래 GT Layout은 예측된 Layout이 아닌 Ground-Truth의 Layout을 사용한것
  - ![Table1](/assets/post/201202/figure6.png)
    - 점차적으로 그래프의 복잡도를 올리며 생성한 결과인데 그래프에 따라 object의 위치가 결정되는 것을 명확히 보여준다
  - ![Table1](/assets/post/201202/table1.png)
    - 여러가지 설정별 Inception Score 비교표

  - 사소한 실험이 몇개더있는 데 별 의미가 없어보여 빼버렸...

  - 
# 5. Conclusion
  - 결론은 Scene Graph에서 이미지로의 End-to-End 모델을 만들었 다는것.
  - 텍스트가 아닌 구조화된 그래프에서 생성시 명시적인 관계에 의한 객체들을 생성 할 수 있고 객체가 많은 복잡한 구조도 표현이 가능하다