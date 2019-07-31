---
layout: post
title:  "제네레이터를 써보자"
categories: keras
date : 2019-06-28 10:05:05 +0900
comments: true
---


# 제네레이터가 뭘까

나중에해야지  
이게더 어려워

## keras.preprocessing.image.ImageDataGenerator



케라스에서 제공하는 이미지 제네레이터이다  
```
ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)
```

길다............................  
하나씩 다 파보는건 레퍼런스가서 보시고

rotation_range : 회전 제한 각도  (0~180) 
width_shift_range, height_shift_range : 상하좌우 이동가능비율  
horizontal_flip : 좌우반전여부   
vertical_flip = 상하반전여부 
zoom_range : 확대축소비율 (0~1)

이정도면 되겠네요 적당한 숫자를 넣어 줍시다







## 주로 같이 사용하는 함수
```
plt.imshow()
```
이미지 프린트

```
PIL.Image.open
```
이미지를 연다. jpg도 가능   
불러오는 파일에 따라 다른타입으로 저장되는데 그냥 img로 퉁침  

```
img.crop((x1,y1,x2,y2))
```
대망의 자르기 

```
img.resize(size=(x,y), box =(x1, y, x2, y2) )
```
대망의 사이즈 바꾸기인데 box 에 위에 crop에 해당하는 좌표를 넣어주면 두작업을 한번에 할 수 있다  
한마디로 크롭 무쓸모? 속도차이가 나려나  

## 참고
```
os.path.join(path, image_name)
```
이미지 이름과 패스를 합치는 함수  
그냥 약간의 편의성 증진