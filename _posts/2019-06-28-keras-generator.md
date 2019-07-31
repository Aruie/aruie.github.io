---
layout: post
title:  "제네레이터를 써보자"
categories: keras
date : 2019-06-28 10:05:05 +0900
comments: true
---


# 제네레이터가 뭘까

데이터가 필요한데 데이터가 부족하다? 흔한 현상이죠  
하지만 이미지에선 그 틀을 조금 부술수 있는 기법이 존재하죠  
바로 이미지 증식!  
고양이사진을 3픽셀 옆으로 민다고 강아지 사진이 되진 않는다는거죠  
회전 확대 이동 반전등 여러가지를 사용해도 우리 눈엔 고양이로 보인다는것.  
하지만 픽셀 데이터 상으로는 완전히 다른 데이터가 나오므로 이것을 이용해 데이터도 늘릴 겸 위치이동이나 회전에 대한 학습도 하게 되는거죠  

하지만 안그래도 용량 많이먹는 이미지를 전부 다 돌리고 늘리고 한것들을 전부 저장한다면 하드든 메모리든 펑 하고 터져버리겟죠?  
그래서 메모리상엔 기본 이미지만 올려놓고 학습시 배치단위로 입력될 때 살짝 변조해서 데이터를 입력해주는 방식으로 큰 메모리 손실없이 많은 데이터를 학습하게 하는 생성기입니다.

는 사실 그냥 쓰면됩니다  
그럼 써봅시다

## keras.preprocessing.image.ImageDataGenerator

케라스에서 제공하는 이미지 제네레이터인데요  
이래서 케라스를 쓴다 할정도로 편합니다.. 이거 직접 구현하러면 세상귀찬...  

```
ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)
```

길죠............................  
하나씩 다 파보는건 레퍼런스가서 보시고

rotation_range : 회전 제한 각도  (0~180)  
width_shift_range, height_shift_range : 상하좌우 이동가능비율  
horizontal_flip, vertical_flip : 좌우상하반전여부  
zoom_range : 확대축소비율 (0~1)  

이정도면 되겠네요 적당한 숫자를 넣어 줍시다

```
datagen = ImageDataGenerator(rescale =  1./255,          # 0과 1사이로 변경
                             zoom_range = 0.2,           # 확대축소 20%
                             width_shift_range = 0.2,    # 좌우이동 20%
                             height_shift_range = 0.2,   # 상하이동 20%
                             rotation_range = 30,        # 회전각도 30도 이내
                             horizental_flip = True)     # 좌우반전 True
```

이런식으로 제네레이터를 생성해 주시면 됩니다  
끗 이면 좋겟지만 아직 좀남았어요  
이제 생성규칙 하나 만들어줬을 뿐이죠  

진짜 제네레이터를 만드는건 3가지 함수가 있습니다

```
# 먼저 데이터프레임에서 불러오는 함수입니다

generator = datagen.flow_from_dataframe(X_train,                # 데이터프레임
                                        directory = './image',  # 데이터 위치
                                        x_col = 'img_file',     # 파일위치 열이름
                                        y_col = 'class',        # 클래스 열이름
                                        target_size = (224,224),    # 이미지 사이즈
                                        color_mode= 'rgb',          # 이미지 채널수
                                        class_mode= 'categorical',  # Y값 변화방법
                                        batch_size= 32,         # 배치사이즈
                                        Shuffle = True,         # 랜덤 여부
                                        seed = 42,              # 랜덤엔 시드
                                        interpolation= 'nearest')   # 이미지변경시 보완방법

generator = datagen.flow_from_directory('./image'       # 파일위치
                                        classes = []    # 리스트
                            )


```





## 참고용 주로 같이 사용하는 함수
```
plt.imshow()
```
이미지 프린트  
잘잘렸는지 확인은 해봐야겟죠?  

```
PIL.Image.open()
```
이미지를 여는 함수에요 뒤엔 주소가 들어가야겠죠. jpg도 가능합니다. 사실 불러오는 파일에 따라 다른타입으로 저장되는데 그냥 img로 퉁침  

```
Image.crop((x1,y1,x2,y2))
```
대망의 자르기   
원하는 크기만큼의 박스로 사진을 잘라줍니다


```
Image.resize(size=(x,y), box =(x1, y, x2, y2) )
```
대망의 사이즈 바꾸기인데 box 에 위에 crop에 해당하는 좌표를 넣어주면 두작업을 한번에 할 수 있다  
한마디로 크롭 무쓸모? 속도차이가 나려나  


## 참고
```
os.path.join(path, image_name)
```
이미지 이름과 패스를 합치는 함수  
그냥 약간의 편의성 증진을 위한건데  
소스마다 안보이는곳이없....  
역시 사소한게 중요한것인듯  