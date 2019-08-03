---
layout: post
title:  "파이썬에서 OpenCV 사용"
tag: "Library & Framework"
date : 2019-07-12 10:05:05 +0900
comments: true
---


# OpenCV 

컴퓨터 비전을 위한 오픈소스 라이브러리이다
여러 언어에서도 다 사용가능하고 강력하다하니 배워보자

# 설치
```
$ pip install opencv-python
```
끗

# 불러오기
```
import cv2
```
끗

참고로 따로 창이 생성되어 작업이되는게 많아 jupyter 환경에서는 좀 문제가 많이생기니
VSCode 나 파이참같은 환경을 추천
(물론 피해가는 방법도 있는데 이런것은 추천하지않는다 언제 막힐지 몰라서...)

# 세부 함수

## 이미지 함수들

```
# 이미지 불러오는 함수
image = cv2.imread('이미지 파일')

# 이미지 보여주는 함수
# 참고로 주피터는 사용 불가 plt.imshow 추천
cv2.imshow('Test', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

```


## 영상 함수들

```
# 영상 불러오는 함수
vidcap = cv2.VideoCapture('비디오 파일')

# 영상 한프레임 받아오기
# 자동으로 다음 프레임상태로 넘어감
# 더이상 영상이없으면 ret가 False 출력
# 루프문으로 적당히 돌려주면 된다
# frame 은 ndarray 형태로 바로 사용가능
ret, frame = vidcap.read()

# 메모리 해제
vidcap.release() 
```


# 카메라 함수들

```
# 카메라 연결하는 함수
vidcap = cv2.VideoCapture( camera_num )

# 영상 설정
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, width)
```


