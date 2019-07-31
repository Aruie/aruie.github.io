---
id : p0001
layout: post
title:  "윈도우에서 WSL을 이용한 딥러닝 개발환경 세팅"
categories: environment
date : 2019-06-24 10:05:05 +0900
comments: true
---



# WSL 

저번에 포스팅 한적이 있는데 윈도우에서 리눅스를 사용하게 해주는 프로그램
가상머신에 비해 가볍고 좋다고한다
기본적으로 폴더 공유도 가능하고 
자세한 기능은 저도 모릅니다 그냥 써요 ㅠㅠ

사용법은 

제어판 - 프로그램 - Windows 기능 켜기/끄기 - Linux용 Windows 하위 시스템 체크 - 재부팅 

![01](/assets/p0001/p0001_01.png)

간단하네요

그리고 Microsoft Store 에서 원하는 리눅스를 검색
(아직 Centos 는 없답니다 ㅠㅠ)  
저는 우분투를 골랐네요 그냥 설치 누르시면 알아서 깔려요

그럼 끗

실행해보면 리눅스 쉘 화면이뜨는데 폴더가 완전히 다른데요
윈도우에서의 폴더에 접근하러면 /mnt 폴더에 들어가보면
자신의 드라이브를 볼 수 있네요

자신의 기본드라이브 (/home/<사용자명>) 에서
```
$ ln -s /mnt/<자신의 작업드라이브> <바로가기명>
$ cd <바로가기명>
```
으로 바로가기를 생성하면 편하게 접근 가능하죠

# Python

제 고민은 여기서 발생했죠
기본적으로 VSCode를 사용하니 윈도우가 편한데
기본적으로 파이썬이나 git은 리눅스가 편리하고
심지어 리눅스에서만 작동하는 라이브러리도 나오고
이걸 양쪽에서 쓰려니 양쪽에 파이썬을 다깔아야하고
패키지도 중복설치가 되겠고
심지어 Cuda 처럼 몇기가 단위의 프로그램도 중복으로 깔려야 한다는거...?

그래서 윈도우에서 이 리눅스에 깔린 파이썬 인터프리터를 사용할 순 없을까 해서 찾던중 발견했네요
(참조 주소는 아래에 표시)
VSCode의 베타버전인 VSCode-insiders에 이 기능이 있다는것! 언젠간 정식버전도 지원하겠죠?

일단 설치를 해봅시다

# VSCode insider

[https://code.visualstudio.com/insiders/](https://code.visualstudio.com/insiders/)

그냥 들어가서 윈도우용으로 설치하시면됩니다 


설치 후   
![](/assets/p0001/p0001_02.png)  
이 버튼을 누르시면 추가 기능 설치가 가능한데요
여기서  
Remote Development  
이것을 설치하시면 됩니다

그리고 리눅스와 VSCode를 전부 꺼줍니다

이제 다시 리눅스를 켜주신 후 (껏다 키지 않으면 명령을 못찼습니다)
```
$ code-insiders .
```
를 쳐주시면(뒤에 점 빼먹으면 안됩니다 ㅠ) 뭔가를 설치 후 실행하죠  
![](/assets/p0001/p0001_03.png)  
잘보시면 왼쪽 아래 구석부분이 이렇게 변한게 보이면 성공!

![](/assets/p0001/p0001_02.png)  
이제 다시 이걸 클릭하시고 Python을 쳐주시면  
그냥 install 대신 install on WSL 이란 버튼이 뜰겁니다  
이걸 설치해주시면 리로드가 필요하다고 뜨고 눌러주시면 됩니다  

이제 기본환경 완성!
이제 아무거나 .py 파일을 하나 만드시고
1+1 같은 거 하나 써주신담에 
Shift + Enter 를 누르시면 되는데

도중에 오른쪽아래에 인터프리터니  
쉬프트엔터를 어떤걸로 사용하겟느니 여러가지뜹니다  
자세한건 너무빨리넘겨서.. 다음번 해볼때 수정할께요  
전부 OK 해주시면 이런환경이 완성됩니다  

![](/assets/p0001/p0001_04.png)  

참고로 지금 작업하는 포스트도 이 환경에서 작업하는...  
(물론 지금은 오른쪽이 마크다운 뷰어로 나오고있죠)  
아무튼 소스코드, 오른쪽엔 쥬피터 노트북 및 변수  
그리고 아래쪽엔 터미널이 뜨는데  
git 이나 pip 등의 명령어를 사용하기 아주 좋은 ㅠ  
맨날 창 몇개씩 키고하다가 확 줄어드는 신세계를 경험하네요 

참고로 이환경에선   
셀을 나눌땐 아래 문구를 사용
```
#%% 
```

Shift + Enter 는 셀실행 및 아래셀로 넘어가기(없을땐 생성)
블록지정후 Shift + Enter 는 블록부분만 실행  
Ctrl + Enter 는 셀실행 및 커서이동 없음  
그리고 F5 를 누를경우 파일단위 실행도 가능합니다(디버깅도 가능)



이정도가 되겠네요


참고로 WSL이 아닌 SSH로도 외부접속을 하여 가능한듯 한데  
이건 아직 안해봐서 해보고 알려드릴게요


# Tensor-flow

자 이제 기본세팅이 끝났으니 텐서플로우를 가봐야겠죠?  
일단 텐서플로우 GPU를 위해 해야할 일들은  
[https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu)  
여기에 참 잘 나와있죠?

시키는대로 해봅시다  
그런데 참고로 전 2.0 베타를 설치할거에요  
정식버전 언제나올건지 원....

## Cuda

절대로 텐서플로우 홈페이지에서 설치하라고 하는 버전으로 설치하셔야합니다
자신의 환경 잘보시고 꼭 맞춰서...
이거 하다가 안돼서 고생하는분들이 많아서

참고로 이상하게도 현재 우분투 18.04에서 네트워크설치가 안되는 현상이  있으니 안되면 로컬로 설치합시다

deb[local] 버전을 받으시면 되는데요
Cuda가 용량이커서 시간이좀걸려요
다운로드도 그렇고 설치도 그렇고...

![](/assets/p0001/05.png)

받으시면 위에 순서대로 실행하는데
1번 실행 후 2번은 중간에 <version> 이 비어있으니  
1번 결과창에 어떤 실행을 하라고 합니다 그걸 치는거에요  
3번 4번은 연속으로 실행하시면 됩니다  
4번이 저어어엉말 오래걸려요 ㅠㅠ 거의 1시간은 걸리는듯...  
한숨 주무시고 오심이...


## NVidia Driver

```
$ sudo apt-get install --no-install-recommends nvidia-driver-410


sudo apt purge nvidia*
```
넘어갔음

## cuDNN

[https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download)

참고로 로그인을 해야 다운이 가능하니 가입을...
cuDNN Library for Linux 를 받읍시다
물론 CUDA 버전과 잘맞춰서...

전 받으면 파일 확장자가 이상한걸로 바뀌는데 그냥 다시 tgz로 바꾸면됩니다
```
$ tar xvfz cudnn-10.0-linux-x64-v7.6.1.34.tgz

$ sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include 
$ sudo cp -P $cuda/lib64/libcudnn* /usr/local/cuda/lib64 
$ sudo chmod a+r /usr/$local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
압축 풀어 주시공 파일을 저 위치로 이동 시켜줍니다
사실 PATH 연결된데면 다 상관없는거같은데 시키는대로 합시다
```
$ sudo apt-get install libcupti-dev
```

이건 이따가

```
$ vi ~/.bashrc
$ code-insiders ~/.bashrc
```
편한 편집기 쓰세요  
근데 위에부터 쭉했다면 아래께 당연히 편하겠죠?
vi에디터 쓰면서 진짜 암울했는데 기쁘네요

```
export PATH=/usr/local/cuda/bin:$PATH 
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64" 
export CUDA_HOME=/usr/local/cuda
```
이거 내용에 추가해주시고

```
$ source ~/.bashrc
```
내용을 바꿧으면 적용해야죠  
그리고 한번 껏다킵시다  
컴퓨터말고 그냥 쉘만 껏다 키면 됩니다

이제 다시 
```
$ nvcc -- version
```
버전이 잘나온다면 굿

## Tensorflow

이젠 진짜로 텐서플로우를...
```
$ pip3 install tensorflow-gpu==2.0.0-beta1
```
살짝 시간이 걸리긴 하지만 CUDA 정도까진 아닌...  

이제 다시 test.py를 하나 만들어 볼게요




# 참조

텐서플로우 공식 홈페이지  
[https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu)

김경훈님 블로그  
[https://gyeonghunkim.github.io/wsl-%ED%99%98%EA%B2%BD%EC%84%A4%EC%A0%95/install-VSCode-on-WSL/](https://gyeonghunkim.github.io/wsl-%ED%99%98%EA%B2%BD%EC%84%A4%EC%A0%95/install-VSCode-on-WSL/)

goodtogreate 블로그
[https://goodtogreate.tistory.com/entry/TensorFlow-GPU-%EB%B2%84%EC%A0%84-%EC%9A%B0%EB%B6%84%ED%88%AC-1604%EC%97%90-%EC%84%A4%EC%B9%98-%ED%95%98%EA%B8%B0](https://goodtogreate.tistory.com/entry/TensorFlow-GPU-%EB%B2%84%EC%A0%84-%EC%9A%B0%EB%B6%84%ED%88%AC-1604%EC%97%90-%EC%84%A4%EC%B9%98-%ED%95%98%EA%B8%B0)


GPU Support for Deep Learning on Windows Subsystem Linux(WSL) with Conda
[https://earnfs.wordpress.com/2019/02/24/gpu-support-for-deep-learning-on-windows-subsystem-linuxwsl-with-conda/](https://earnfs.wordpress.com/2019/02/24/gpu-support-for-deep-learning-on-windows-subsystem-linuxwsl-with-conda/)

miniconda
http://www.erogol.com/using-windows-wsl-for-deep-learning-development/