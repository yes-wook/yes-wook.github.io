---
layout: post
title: Install pytorch on Windows
comments: true
---

# Pytorch  
Lua언어로 된 딥러닝 프레임워크인 Torch를 페이스북에서 Pythonic하게 만들어서 배포하는 파이썬용 딥러닝 프레임워크  
[Pytoorch official site](http://pytorch.org/)

# 포스팅 이유  
파이토치가 리눅스, 맥만 지원한다...  
나는 윈도우 쓰는데...!  
Tensorflow나 CNTK가 윈도우를 지원하지만, 파이토치가 그렇게 쉽다던데...  
  
해서 찾아보니 설치방법이 의외로 간단하다.  

# 윈도우에서 설치하는 방법  
나는 기존에 Anaconda를 이용하고 있다. 아마 윈도우에서 파이썬이용자 분들의 대다수는 아나콘다를 이용하고 있으리라 믿는다.  
cmd 창에서 
```
conda install -c peterjc123 pytorch  
```  
명령어를 입력하면 된다.  

# 설치가 잘 되었는지 테스트  
``` Python
import torch  
x = torch.rnad(3,3)  
print(x)
```
위 코드를 실행했을때, 3x3 사이즈의 숫자가 생성되면 된다.  
알겠지만 무작위 수를 3x3사이즈로 생성하는 코드니까...  