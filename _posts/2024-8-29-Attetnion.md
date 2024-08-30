---
layout: post
title: "[논문 리뷰] Attention is all you need"
---
이미 리뷰가 많은 논문이지만 transformer모델을 더 깊게 이해하려면 직접 논문을 보고 정리하는 편이 좋겠다는 생각에 이 논문을 리뷰하기로 하였다.

>### Attention이 내가 필요한 모든 것?

논문의 제목이 상당히 파격적인데, 논문에서 제시한 트랜스포머 모델의 성능 또한 그러했다. Attention은 현재 대부분의 인공지능에 사용될 만큼 중요한 메커니즘이라 할 수 있다.
>### Attention

논문에서는 attention을 다음과 같이 설명한다.  

어텐션 함수는 query와 (key, value) 쌍을 하나의 결과값으로 매핑한다.  
query, key, value, 결과값은 모두 벡터이다.   
결과값은 value의 가중합으로 결정되는데, 이 때 각 value의가중치는 대응하는 key와 query의 유사도다.  

문장을 읽기만 해서는 이해하기가 좀 어려운데, 수식으로 보는 편이 빠르다.  


$$ \operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left({Q K^T}\right) V $$

위 수식에서 Q: Query, K: Key, V: Value 이다.  
모두 행렬의 형태인데, 단순하게 Q를 열이 1개인 matrix, 즉 열 벡터일 때를 생각해보자.

n개의 (key, value)쌍이 있다고 가정하면  
$$ K = (k_1, k_2, ... , k_n),\  V = (v1, v2, ..., vn) $$ 이다. 
