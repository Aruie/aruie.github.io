---
layout: post
title:  "Jekyll 다시 시작"
date:   2019-07-31 10:05:05 +0900
categories: jekyll
comments: true
---

# 다시시작

전에 시도하다 포기하고 일단 포스트만 올리면서 나중에 좋은스킨 나오면 해야지 하다가
이제 슬슬 포트폴리오겸 만들어야겠다 싶어 다시 도전....

일단 설정이 문제가 있었을수도있으니

깔끔하게 우분투를 날려버리고... 다시깔았다...

```
$ sudo apt-get update -y && sudo apt-get upgrade -y
$ sudo apt-get install -y build-essential ruby-full
$ sudo gem install jekyll bundler
```

아니 이게왠일

여기까지 아무 에러가 없이 진행되었..

원하던 스킨을 다운받고

드디어 대망의 

```
$ jekyll serve
```

는 될리가 없지.........ㅜㅜ

jekyll new 로 새로 생성하면 잘만되는데

꼭 스킨을 다운받으면 안되는...

역시 또 하루종일 이거설치 저거설치 

이거삭제 저거삭제

열심히 해본 결과 알아낸 해결책은 해당폴더에서

```
$ bundle install
$ bundle exec jekyll serve 
```

이거였다...

문제는 또 parsing 오류가 떳는데

이건 title 'Aru's Blog'  여기에서 에러가..

이래서 특문을 쓸땐 항상조심 ㅠㅠ

결국 

![성공](/assets/post/190731-1.png)

드디어 성공했다 ㅠㅠㅠ 

이제 꾸미기에 들어가 봅시다