---
layout: post
title:  "WSL 에서 Jekyll 설치"
date:   2019-06-12 16:05:05 +0900
categories: jekyll
comments: true
---

#WSL 에서 jekyll 설치

어제 하던 뻘짓은 날려버리고... 기존 윈도우용 루비를 삭제했습니다

이제 편하게 리눅스에서 지킬을 설치해볼게요

일단 루비를 설치해야겠죠

```
$ sudo apt-get update -y && sudo apt-get upgrade -y
$ sudo apt-get install -y build-essential ruby-full

$ sudo gem update-system
$ sudo gem install jekyll bundler
```

참고로 겁나오래걸려요 루비풀 설치...

이렇게하면 뿅하고 될줄 알았는데 또안된다...

다시 스택 오버플로우 ㅠㅠ

```
$ gem sources --remove https://rubygems.org/
$ gem sources -a http://rubygems.org/
$ gem install jekyll
```

왜 s가 들어가 있는지 있다가 없어진거지 아무튼 ㅠㅇㄻㅈㅁㄷㄹ

이러고 다시 위에있던 밑에 두줄을 실행하면 이제 잘됨 ㅠㅠ

참고로 이거 2014년부터 있던 해결법인데 해결좀 해주지...




