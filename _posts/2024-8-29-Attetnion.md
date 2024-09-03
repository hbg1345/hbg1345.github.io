---
layout: post
title: "[논문 리뷰] Attention is all you need"
---
이미 리뷰가 많은 논문이지만 transformer모델을 더 깊게 이해하려면 직접 논문을 보고 정리하는 편이 좋겠다는 생각에 이 논문을 리뷰하기로 하였다.

>## Attention이 내가 필요한 모든 것?

논문의 제목이 상당히 파격적인데, 논문에서 제시한 트랜스포머 모델의 성능 또한 그러했다. 나온지 수년이 지난 지금도 Attention은 현재 대부분의 인공지능에 포함되는 핵심 메커니즘이다. 
>## Attention

논문에서는 attention을 다음과 같이 설명한다.  

어텐션 함수는 query와 (key, value) 쌍을 하나의 결과값으로 매핑한다.  
query, key, value, 결과값은 모두 벡터이다.   
결과값은 value의 가중합으로 결정되는데, 이 때 각 value의가중치는 대응하는 key와 query의 유사도다.  

문장을 읽기만 해서는 이해하기가 좀 어려운데, 수식으로 보는 편이 빠르다.  


$$ \operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left({Q K^T}\right) V $$

(참고: 앞으로 나오는 벡터는 모두 **행 벡터**이고, 굵은 글씨로 표기되었다.)

위 수식에서 **Q: Query, K: Key, V: Value** 이다.  
모두 행렬의 형태인데, 단순하게 Q를 행이 1개인 행렬, 즉 행 벡터 
$$ \mathbf{q} $$ 
로 가정하자.

n개의 (key, value)쌍이 있다고 가정하면  

$$ K = \left (\begin{array}{ccc} \mathbf{k_1} \\ \mathbf{k_2}\\ \vdots \\  \mathbf{k_n} \end{array} \right ),\  V = \left (\begin{array}{ccc} \mathbf{v_1} \\ \mathbf{v_2}\\ \vdots \\  \mathbf{v_n} \end{array} \right ) $$ 
이다.

$$ \mathbf{q}K^T = \mathbf{q} (\mathbf{k_1^T}, \mathbf{k_2^T} , \ldots, \mathbf{k_n^T}  )= (\mathbf{qk_1^T}, \mathbf{qk_2^T} , \ldots, \mathbf{qk_n^T}  )=(\mathbf{q\cdot k_1}, \mathbf{q\cdot k_2} , \ldots, \mathbf{q\cdot k_n}  )$$

각 행은 벡터 q와 해당 key벡터의 스칼라 곱으로, 벡터 간의 유사도를 의미한다.  
$$ qK^T$$
vector에 softmax를 적용하면 이제 가중치 벡터 W를 얻을 수 있다.

$$ W = \operatorname{softmax}(\mathbf{q}K^T)=( w_1, w_2,  \ldots,  w_n ), \sum_{i=1}^n w_i = 1 \\ Attention(Q, K, V) = WV = ( w_1, w_2,  \ldots,  w_n )V\\ ( w_1, w_2,  \ldots,  w_n )\left (\begin{array}{ccc} \mathbf{v_1} \\ \mathbf{v_2}\\ \vdots \\  \mathbf{v_n} \end{array} \right )\\ \therefore Attention(Q, K, V)=\sum_{i=1}^nw_i\mathbf{v_i}$$

즉, 어텐션 함수는 이미 갖고 있는 n개의 key와 value의 관계 데이터를 이용하여 새로운 key,즉 q에 해당하는 value를 얻는 방법이라고 할 수 있다.  

>## Dot-product Attention vs Additive Attention

예시로 든 어텐션 함수는 dot product(스칼라 곱)로 key와 query간 유사도를 측정하는 **dot-product attention**이다.  
이 외에도 주로 사용하는 어텐션 함수로 **additive attention**이 있다.  
additive attentnion은 한 층의 hidden layer로 구성된 신경망으로 query vector와 key vector간의 유사도를 예측한다.  

두 어텐션은 이론적으로는 비슷한 성능을 가지지만, 실제로는 행렬 곱 코드가 잘 최적화되어서 dot-product어텐션이 훨씬 빠르고 메모리를 덜 사용한다.  
하지만 additive 어텐션은 key벡터의 차원이 증가할 수록 dot-product 어텐션보다 뛰어난 성능을 보이는데, 그 이유는 softmax함수의 특성에 있다.  

>## Key vector의 차원과 어텐션 성능의 관계?

key벡터의 차원이 커지면, 스칼라 곱이 매우 큰 값을 가질 수 있다. 논문에서는 스칼라 곱이 크면 소프트맥수 함수의 그래디언트가 매우 작아져 모델 업데이트가 제대로 되지 않을 수 있다는 점을 이유로 설명한다.  

조금 더 생각해보면, 소프트맥스 함수는 값에 지수 함수를 적용하고 정규화하기 때문에, 특정 값만 1에 가깝고 나머지는 0에 가까울 수 있다.  
소프트맥스 함수의 그래디언트는 원래 값들의 곱의 형태이므로,  0에 가까운 값이 될 수 있다.

>## Scaled Dot-Product Attention

이러한 dot-product attention의 단점을 해결하기 위해 논문에서는 dot product후 key vector의 차원 d_k로 스케일링하는 scaled-dot product attention을사용한다.

$$ \operatorname{Scaled-Attention}(Q, K, V)=\operatorname{softmax}\left({\frac{QK^T}{\sqrt{d_k}}}\right) V $$

>## Multi-Head Attention

Multi-Head(layer) attention은 어텐션을 여러 번 **병렬**로 적용하는 방법이다.  
물론, 같은 Q, K, V에 대한 어텐션 함수의 출력은 항상 같다.  
그래서 각 행렬에 적절한 가중치 행렬을 곱해주어 새로운 Q', K', V'에 대해 어텐션을 적용한다.

예를 들어 h개의 어텐션 헤드가 있다고 하자.  

i번 어텐션 층(헤드)에는 다음과 같이 원래 행렬에 가중치 행렬이 곱해진 새로운 행렬이 입력으로 들어간다.

$$ Q_i=QW_i^{\operatorname{Query},}\ K_i=KW_{i}^{\operatorname{Key}},\  V_i=VW_{i}^{\operatorname{Value}} $$

i번 어텐션 헤드의 출력은 다음과 같다.

$$ \operatorname{Head_i}=\operatorname{Attention(Q_i, K_i, V_i)}  \\ $$

이제 h개의 어텐션 행렬을 concatenate해서 하나의 행렬로 만든다.  

(참고: 여기서 concatenate란 단순히 이어 붙이는 것을 의미한다.  
예를 들어 
$$ \left (\begin{array}{ccc} 1\ 2\ 3 \\ 4\ 5\ 6\end{array}\right)과 \left (\begin{array}{ccc} 7\ 8\ 9 \\  0\ 1 \ 2 \end{array}\right)를\ 이어붙이면  \left (\begin{array}{ccc} 1\ 2\ 3\ 7\ 8\ 9 \\  4\ 5\ 6\ 0\ 1 \ 2 \end{array}\right)이다.$$
)  
이제 마지막으로,   
이어 붙여진 value행렬에 가중치 행렬을 곱한다.

$$ \operatorname{Multi-Head(Q,K,V) = \operatorname{Concat(Head_1, Head_2,\ldots,Head_h)}W^{\operatorname{Out}}}$$

>## 어텐션을 여러 번 하는 이유

가중치 행렬을 곱해서 크기를 작게 조정하는 것은 행 벡터 각각을 새로운 부분 공간으로 프로젝션하는 것과 같다.  
어텐션 헤드마다 프로젝션에 사용하는 가중치가 다르므로 각 헤드의 출력은 서로 다른 부분 공간에서의 어텐션 값이다.  

이렇게 다른 부분 공간에서의 어텐션이 가지는 의미는 각 벡터를 단어 벡터로 생각했을 때 쉽게 이해할 수 있다.

현재 갖고 있는 
key, vlaue 쌍을 (사과, apple), (멜론, melon)라고 하자.  
Query를 수박이라고 했을 때, 얻고자 하는 value는 watermelon이다.  

어텐션을 한번 적용(single-head attention)하면 (수박, 사과), (수박, 멜론)의 유사도를 측정하고, 그 유사도에 비례하게 apple과 melon을 더하여 새로운 value를 얻을 것이다.  

멀티 헤드 어텐션은 우선 수박, 사과, 멜론, apple, melon의 색깔, 모양, 줄무늬 등의 h개의 성분 벡터 각각에 대해 attention을 수행한다. 그 뒤 얻은 h개의 value들; 색깔과 모양, 줄무늬 등을 종합하여 최종value를 출력한다.

즉, 멀티 헤드 어텐션은 다양한 방향에서 객체를 분석한 뒤 그 분석 결과를 종합하여 최종 결과물을 얻는 방법이라고 볼 수 있다.  

(참고: 위 예시는 필자의 주관적인 해석이다.)

>## Self-Attention

셀프 어텐션에서 query, key, value는 모두 동일한 벡터로부터 비롯된다.  
즉, 입력 행렬 X에 적절한 가중치 행렬을 곱해 Q,K,V matrix를 얻는다.  

예를 들어, I am a boy. 라는 문장이 X로 주어진다고 해보자.  
셀프 어텐션을 수행하면 각 쿼리 벡터에 대한 value값이 나온다.  
어텐션 레이어가 적절히 학습되었다면 query 'boy'에 대한 결과값의 대부분은 value 'I'로부터 구성될 것이다.

이렇듯 셀프 어텐션은 각 단어 벡터를 문장 속 단어들 사이의 정보를 포함한 새로운 벡터로 바꾸는 역할을 한다.  
예를 들어 '사과'를 나타내는 임베딩 벡터는 하나지만, 빨간 사과, 파란 사과에 셀프 어텐션을 수행하면 사과에 색 정보가 포함된 벡터를 얻을 수 있다.

> ## Masked Attetntion

Attetnion은 단어 시퀀스가 통째로 입력으로 들어가기 때문에, i번째에 오는 단어를 예측하기 위해서는 i이상의 위치에 있는 단어 시퀀스가 입력으로 전달되지 않도록 해야 한다. Masked Attention은 softmax를 시행하기 전, i이상 위치의 값을 모두
$$ - \infin $$ 
로 설정한다.  
이렇게 하면, 소프트맥스 이후 값이 모두 0이 되어, i이후 단어의 value 벡터들이 attention의 결과에 포함되지 않는다.

> ## Position-wise Feed-Forward Networks

FCN->ReLu->FCN의 형태이다.
정확히는 모르겠지만, 모델을 더 깊게 만들는 역할인 듯하다.

> ## Positional Encoding

어텐션은 각 행이 단어 벡터와 관련된 행렬을 입력으로 받는다.  
이렇게 하면 병렬로 처리가 가능하다는 장점이 있지만, 단어 시퀀스 내에서의 순서 정보를 담지 못하는 단점이 생긴다.  
논문에서는 이를 해결하기 위해 위치 인코딩 레이어를 추가하였다.

수식은 다음과 같다.

$$ \operatorname{PE(pos, 2i)}=sin(\frac{pos}{10^{4 * \frac{2i}{d_{model}}}}) \\ \operatorname{PE(pos, 2i+1)}=cos(\frac{pos}{10^{4 * \frac{2i+1}{d_{model}}}})$$

pos는 시퀀스 내에서 단어의 위치이고, d_model은 단어 벡터의 차원, 2i, 2i+1은 각각 한 단어 벡터 내에서 짝수, 홀수 번째 값을 의미한다.

이 PE가 이런 형태인 이유는 이후 추가하도록 하겠다.

> ## 트랜스포머 구조

<p align="center">
  <img src="../Transformer architecture.png">
</p>

트랜스포머는 인코더-디코더 구조로, 각각 N개의 sublayer들로 구성되어 구성된다.  
우선 입력 행렬은 포지셔널 인코딩과 셀프 어텐션 층을 지난다. 이후 정규화층과 FFN을 거친 뒤 다음 인코더 sublayer로 전달된다.  
디코더 sublayer에서는 인코더의 마지막 sublayer의 출력 행렬 만들어진 key, value matrix와 셀프 어텐션한 query행렬로 어텐션을 수행한다. 이후 FFN에 전달되고 다음 sublayer로 넘어간다 마지막은 softmax를 한 뒤 단어들의 확률을 출력한다.

인코더의 입력을 질문, 디코더의 입력을 현재까지의 응답이라고 해보자.  
인코더의 입력: What is your name?  
디코더의 입력: My name is 

셀프 어텐션은 각 문장 내에서 단어 간 정보를 섞는 효과를 준다.  
디코더의 다음 출력이 민수라고 해보자.
민수임을 예측하기 위해서는, 질문 문장과 현재 응답 문장사이의 관계를 알아야 한다. 이를 위해 인코더의 문장을 key, value로 사용하고 디코더 문장을 query로 어텐션을 수행한다.  

(참고: 필자의 주관적 해석이다.)

p.s.
다음 포스팅은 트랜스포머를 구현하고 간단한 챗봇을 만드는 내용으로 할 예정이다.











