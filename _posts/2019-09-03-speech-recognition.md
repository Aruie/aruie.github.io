---
layout: post
title:  "ASR(Automatic Speech Recognition) 정리"
tag : tip
date : 2019-09-03 10:05:05 +0900
comments: true
---


## 전통적인 음성인식
전통적
전처리 - 특징추출 - 특징 to 음소 - 음소 to 단어(발음사전) - 단어 to 문장(문법모델)

현재 
음성전처리 - 특징추출 - 특징 to 문장 (사운드 모델)\

사운드모델
입력 : 푸리에 (20ms인데 10ms씩 겹치게)
출력 : character
mel 필터 (사람이 잘 들리는 부분에 대해 적용하는 필터)
입출력 길이가 불일치

DNN : 매 프레임마다 정답을 설정


CTC( Connectionist Temporal Classification) : 
 - 아무것도 안나오는 _ black를 추가
 - 블랭크 매우 반복시 띄어쓰기 사용
 - 블랭크없이 반복된문자 합침
 - 미분이 가능하다
 - 경계가 애매할때 사용
 - 계산법
   - 가능한 패스를전부 계산 (동적프로그래밍)
   - 조건부 독립의 문제(Conditional Independent Problem) : 덕분에 spelling이 엄청나게 틀림

Gram-CTC
 - 가능한 조합에 n-gram을 추가함 
 - 사실 발음은 단일발음이 아닌 bi-gram 이다
 - 한글에서도 충분히 가능할 듯

이 CTC가 생기면서 End-To-End 가 가능해졌음

Seq to Seq With Attention
 - 입력 전체를보고 출격을 결정
 - Conditional Independence 해결
 - 비대칭문제 해결

Attention의 문제점
 - 계산량이 많고, 실시간 출력이 불가능( 어텍션은 문장이 끝나야함)
 - 빔서치가 어려움
 - 훈련이 매우 오래걸림

LAS (listen, attend and spell, 2015)
 - 시퀸스의 구간을 줄이는 방식으로 연산량을 줄임
 - 처음으로 어텐션을 추가함

Monotonic Attention (2017)
 - 짧은 구간으로 시퀸스를 나눠서 실행

CNN의 침범
 - 방향, 시간으로 지역성 확보
 - 스펙트로그램을 한칸씩 이동하여 여러 채널을 만들어서 연산량을 줄임
 - 맥스풀링으로 프레임감소

DeepSpeech 2
 - 스펙트로그램을 CNN을 몇레이어 돌라고 양방향 RNN을 돌리고 FCN 후 CTC 실행
 - 휴먼퍼포먼스를 넘어간 첫번째 사례인데 데이터가 신빙성이 조금 떨어짐

Convolutional CTC (텐센트, 2017)
 - RNN부분을 완전히 없애버림
 - Residual Block 사용
 - 성능 매우 좋음

고려할점
 - 실시간처리와 일괄처리
 - 기기, 서버
 - 음향모델, 언어모델
 - 전처리가 엄청나게 많이 필요

전처리 내역
 - 방향탐지 (Beam forming)
 - 음성탐지 (voice activity dectection)
 - 정적탐지 (silence removal)
 - 노이즈제거 (noise reduction)
 - 세기조절 (gain control)
 - 반향제거 (echo cancellation)
 - 사람분리 (cocktail party problem)

후처리 (경험적 보정?)
 - 후보를 만들고 평가(beam search)
 - 





다본느낌
 - 전처리 및 증식부분에 대해 찾아봐야겟
 - 







End-to-End 이 현재 트렌드

# Sound Model 



