---
layout: post
title: Nearly everything of information theory
---

그동안 대략적으로 느낌만 알고 있던 정보이론에 대해서, 공부하고 정리할 일이 생겼는데  
생각보다 재밌는 개념이고 정리도 잘 된것 같아서 만든 ppt를 그대로 이미지로 첨부합니다.  

특히, 정보이론에 대한 설명을 베이지안으로 시작하는 경우는 아직 못보았는데 책 [불멸의 이론](https://www.google.com/search?q=%EB%B6%88%EB%A9%B8%EC%9D%98+%EC%9D%B4%EB%A1%A0)의 영향을 받아서 베이지안으로 시작해보았습니다.  
  
정보이론은 정보통신에서 시작한 개념이라 소스코딩등으로 시작하는 경우가 많은데, 저는 그쪽 전공자가 아니다보니 전공자 분들께서 생각하시는 개념과는 약간 다를 수 있습니다. 이 부분은 감안해주시고, 정보량(Information content)의 개념에서 시작하여 정보 엔트로피(Information entropy), 쿨백 라이블러 발산(Kullback-Leibler divergence), 상호정보량(Mutual information) 등 풍부하게 담으려고 노력했습니다. (만 발표자료라 내용이 완전하지 않을 수 있습니다...ㅠㅠ)  
  
p.s 최근 딥러닝의 기본개념중 하나인 교차엔트로피(크로스 엔트로피, Cross entropy)에 대해서 찾다가 알게 되었는데 크로스 엔트로피를 H(p,q)로 표기하는 것은 잘못된 표기입니다. H(p, q)는 p와 q의 결합분포(Joint probability distribution)에 대한 정보엔트로피의 표기이고, 크로스엔트로피는 H_q(p) 가 맞습니다.

![cover](/img/post_img/2018-07-28-information_theory/0001.jpg)  
![about a shannon](/img/post_img/2018-07-28-information_theory/0002.jpg)  
![bayesian](/img/post_img/2018-07-28-information_theory/0003.jpg)  
![information content](/img/post_img/2018-07-28-information_theory/0004.jpg)  
![information content2](/img/post_img/2018-07-28-information_theory/0005.jpg)  
![information content3](/img/post_img/2018-07-28-information_theory/0006.jpg)  
![information entropy](/img/post_img/2018-07-28-information_theory/0007.jpg)  
![information entropy2](/img/post_img/2018-07-28-information_theory/0008.jpg)  
![entropy in physics](/img/post_img/2018-07-28-information_theory/0009.jpg)  
![source coding](/img/post_img/2018-07-28-information_theory/0010.jpg)  
![conditional information entropy](/img/post_img/2018-07-28-information_theory/0011.jpg)  
![mutual information](/img/post_img/2018-07-28-information_theory/0012.jpg)  
![relation between entropy concepts](/img/post_img/2018-07-28-information_theory/0013.jpg)  
![Kullback-Leibler divergence](/img/post_img/2018-07-28-information_theory/0014.jpg)  
![f-divergence](/img/post_img/2018-07-28-information_theory/0015.jpg)  
![Kullback-Leibler divergence for model estimation](/img/post_img/2018-07-28-information_theory/0016.jpg)  
![cross entropy](/img/post_img/2018-07-28-information_theory/0017.jpg)  