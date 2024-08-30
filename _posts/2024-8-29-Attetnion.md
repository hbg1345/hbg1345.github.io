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

(참고: 앞으로 나오는 벡터는 모두 **열 벡터**이고, 굵은 글씨로 표기되었다.)

위 수식에서 **Q: Query, K: Key, V: Value** 이다.  
모두 행렬의 형태인데, 단순하게 Q를 열이 1개인 행렬, 즉 열 벡터 $$ \mathbf{q} $$ 로 가정하자.

n개의 (key, value)쌍이 있다고 가정하면  

$$ K = (\mathbf{k_1}, \mathbf{k_2}, \ldots , \mathbf{k_n}),\  V = (\mathbf{v_1}, \mathbf{v_2}, \ldots, \mathbf{v_n}) $$ 
이다.

$$ \mathbf{q}K^T = \mathbf{q} \left (\begin{array}{ccc} \mathbf{k_1^T} \\ \mathbf{k_2^T} \\ \vdots \\ \mathbf{k_n^T} \end{array} \right) = \left (\begin{array}{ccc} \mathbf{qk_1^T} \\ \mathbf{qk_2^T} \\ \vdots \\ \mathbf{qk_n^T} \end{array} \right)=\left (\begin{array}{ccc} \mathbf{q \cdot k_1} \\ \mathbf{q \cdot k_2} \\ \vdots \\ \mathbf{q\cdot k_n} \end{array} \right)$$

각 행은 벡터 q와 해당 key벡터의 스칼라 곱으로, 벡터 간의 유사도를 의미한다.  
$$ qK^T$$ vector에 softmax를 적용하면 이제 가중치 벡터 W를 얻을 수 있다.

$$ W = \operatorname{softmax}(\mathbf{q}K^T)=\left (\begin{array}{ccc} w_1 \\ w_2 \\ \vdots \\ w_n \end{array} \right), \sum_{i=1}^n w_i = 1 \\ Attention(Q, K, V) = WV = \left (\begin{array}{ccc} w_1 \\ w_2 \\ \vdots \\ w_n \end{array} \right)V\\ \left (\begin{array}{ccc} w_1 \\ w_2 \\ \vdots \\ w_n \end{array} \right)(\mathbf{v_1},\mathbf{v_2}, \ldots,\mathbf{v_n})=\sum_{i=1}^nw_i\mathbf{v_i}\\ \therefore Attention(Q, K, V)=\sum_{i=1}^nw_i\mathbf{v_i}$$

즉, 어텐션 함수는 이미 갖고 있는 n개의 key와 value의 관계 데이터를 이용하여 새로운 key,즉 q에 해당하는 value를 얻는 방법이라고 할 수 있다.  

>### Dot-product Attention vs Additive Attention

예시로 든 어텐션 함수는 dot product(스칼라 곱)로 key와 query간 유사도를 측정하는 **dot-product attention**이다.  
이 외에도 주로 사용하는 어텐션 함수로 **additive attention**이 있다.  
additive attentnion은 한 층의 hidden layer로 구성된 신경망으로 query vector와 key vector간의 유사도를 예측한다.  

두 어텐션은 이론적으로는 비슷한 성능을 가지지만, 실제로는 행렬 곱 코드가 최적화되어서 dot-product어텐션이 훨씬 빠르고 메모리를 덜 사용한다.  
하지만 additive 어텐션은 key벡터의 차원이 증가할 수록 dot-product 어텐션보다 뛰어난 성능을 보이는데, 그 이유는 softmax함수의 특성에 있다.  

>### Key vector의 차원과 어텐션 성능의 관계?

key벡터의 차원이 커지면, 스칼라 곱이 매우 큰 값을 가질 수 있다. 논문에서는 스칼라 곱이 크면 소프트맥수 함수의 그래디언트가 매우 작아져 모델 업데이트가 제대로 되지 않을 수 있다는 점을 이유로 설명한다.  

조금 더 생각해보면, 소프트맥스 함수는 값에 지수 함수를 적용하고 정규화하기 때문에, 특정 값만 1에 가깝고 나머지는 0에 가까울 수 있다.  
소프트맥스 함수의 그래디언트가 원래 값들의 곱임을 생각하면 그래디언트가 0에 가깝게 나올 수 있다.

>### Scaled Dot-Product Attention

이러한 dot-product attention의 단점을 해결하기 위해 논문에서는 dot product후 key vector의 차원 d_k로 스케일링하는 scaled-dot product attention을사용한다.

$$ \operatorname{Scaled-Attention}(Q, K, V)=\operatorname{softmax}\left({\frac{QK^T}{\sqrt{d_k}}}\right) V $$

