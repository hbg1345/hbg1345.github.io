---
layout: post
title: "[논문 리뷰] Attention is all you need"
---
이미 리뷰가 많은 논문이지만 transformer모델을 더 깊게 이해하려면 직접 논문을 보고 정리하는 편이 좋겠다는 생각에 이 논문을 리뷰하기로 하였다.

>### Attention이 내가 필요한 모든 것?

논문의 제목이 상당히 파격적인데, 논문에서 제시한 트랜스포머 모델의 성능 또한 그러했다. 이에 그치지 않고 Attention은 이제 대부분의 인공지능에 사용되는 핵심 메커니즘이 되었다. 
>### Attention

논문에서 설명한 문장의 뜻을 해치지 않도록 번역해보면 다음과 같다.  
어텐션 함수는 query와 (key, value) 쌍을 하나의 결과값으로 매핑한다.  
query, key, value, 결과값은 모두 벡터이다.   
결과값은 value의 가중합으로 결정되는데, 이 때 각 value가중치는 대응하는 key와 query의 유사도다.  

문장을 읽기만 해서는 이해하기가 좀 어려운데, 수식으로 보는 편이 빠르다.  

$$\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$






