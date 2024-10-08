---
layout: post
title: "[알고리즘] 이분 탐색(Binary Search)에 대하여"
---
첫 게시글인 만큼 가벼운 주제로 시작해보고자 한다.  

이분 탐색은 탐색 알고리즘의 한 종류이다.

>## 탐색 알고리즘이란?  

주어진 값들 중에서 내가 원하는 값을 찾게 해주는 것이 탐색 알고리즘이다.  

로컬 컴퓨터에서 원하는 파일을 찾거나, 인터넷에서 검색하는 행위 등에 탐색 알고리즘이 쓰이는 만큼, 중요한 알고리즘이라 할 수 있다.

여러 탐색 알고리즘 중 가장 간단한 것은 완전 탐색 알고리즘일 것이다.  
> ## 완전 탐색 알고리즘  

모든 원소를 하나씩 보며 내가 찾고 있는 원소인지 확인하는 방법이다.  
예를 들어, 사람 10명의 키가 다음과 같다고 하자.


|사람|1|2|3|4|5|6|7|8|9|10|
|-|-|-|-|-|-|-|-|-|-|-|
|키|175|181|172|158|155|169|183|188|165|181|


키가 172cm인 사람을 찾고자 할 때, 왼쪽부터 순서대로 보면서 확인하는 방법이 완전 탐색 알고리즘이다. (물론 오른쪽부터 확인해도 상관 없다.)  
c++ 코드로 작성하면 아래와 같다.
~~~cpp
int height[10] =
{ 175, 181, 172, 158, 155, 169, 183, 188, 165, 181 };
int target_height = 172;
for (int i = 0; i < 10; i++)
    if (height[i] == target_height) {
        printf("%d번 째 사람의 키가 %dcm이다.", i,
            target_height);
        break;
    }
~~~
완전 탐색 알고리즘은 이렇게 정말 간단하지만, 치명적인 단점이 있다.  
바로 시간 복잡도이다.
>## 시간복잡도란?  

알고리즘의 성능을 나타내는 지표로, 주로 big-O표기법을 사용하여 나타낸다.  
O(f(n))의 형태로, n은 입력 데이터의 개수를 의미한다.
  
완전 탐색 알고리즘의 시간 복잡도는 O(n)으로, 데이터 개수에 선형적으로 비례하는 시간이 걸린다. 위 예시에서, 9번 째 원소인 165cm를 찾기 위해서는 9번의 탐색이 필요하다. 예시 데이터는 수가 적어서 괜찮지만 데이터의 수가 많아지면 한번 탐색할 때 시간이 너무 오래 걸리 수 있다.

>## 컴퓨터가 빨라져서 괜찮지 않나?  

라는 생각을 할 수 있지만, 컴퓨터 이용자의 수도 늘어났기 때문에, 원활한 응답을 위해서 다량의 쿼리를 짧은 시간 내에 처리할 수 있는 알고리즘이 필요하다.  

> ## 이분 탐색 알고리즘  

이분 탐색은 O(logn)의 시간 복잡도를 갖기 때문에 정말 효율이 좋다.  
n = 2^10 = 1024개의 데이터를 대략 10번의 탐색으로 처리한다는 점에서 약 1000번을 탐색해야 하는 완전 탐색에 비해 100배 빠른 성능을 보인다.  
n이 크면 클수록 이분 탐색이 유용함을 알 수 있다.  

> ## O(logn)? 어떻게? 
 
이분 탐색은 우선 정렬된 데이터를 가정한다.  
위의 사람 10명의 키 예시를 재사용하면 다음과 같다.  

|사람|1|2|3|4|5|6|7|8|9|10
|-|-|-|-|-|-|-|-|-|-|-|
|키|155|158|165|169|172|175|181|181|183|188

이분 탐색 알고리즘은 한번 탐색할 때마다 탐색 범위 &#91;low, high&#41;을 절반으로 감소시킨다.  &#40;탐색 범위는 찾고 있는 원소가 있을 수 있는 위치 범위이다.  &#41;

예를 들어, 위의 배열에서 172를 찾는다고 가정하자.  
처음 low=1, high=11 이다.   
이제 해당 구간의 중앙에 위치한 (1 + 11)/2 = 6번째 원소를 확인한다.  

|사람|<span style="color:blue">1|2|3|4|5|<span style="color:red">6</span>|7|8|9|10|<span style="color:blue">11
|-|-|-|-|-|-|-|-|-|-|-|-|
|키|155|158|165|169|<span style="color:green"> 172</span>|175|181|181|183|188|-

6번째 원소인 175는 찾는 값인 172보다 크고, 데이터가 오름차순이므로 6번째 이후로는 172가 없다.
이를 바탕으로 새로운 탐색 범위는 &#91;1, 6&#41;이 된다.

|사람|<span style="color:blue">1|2|<span style="color:red">3</span>|4|5|<span style="color:blue">6|x|x|x|x|x
|-|-|-|-|-|-|-|-|-|-|-|-|
|키|155|158|165|169|<span style="color:green"> 172</span>|175|181|181|183|188|-

이제 다시 (1+6)/2 = 3번 째 원소인 165는 172 이하이므로 3번째 미만, 즉 2번째 이하로는 172가 없다. 따라서 새로운 탐색 범위는 &#91;3, 6&#41;이 된다.

|사람|x|x|<span style="color:blue">3|<span style="color:red">4|5|<span style="color:blue">6|x|x|x|x|x
|-|-|-|-|-|-|-|-|-|-|-|-|
|키|155|158|165|169|<span style="color:green"> 172</span>|175|181|181|183|188|-

또 (3 + 6)/2 = 4번 째 원소인 169는 172 이하이므로, 4번째 미만, 즉 3번 째 이하로는 172가 없다. 새로운 탐색 범위는 &#91;4, 6&#41;이다.

|사람|x|x|x|<span style="color:blue">4|<span style="color:red">5|<span style="color:blue">6|x|x|x|x|x|
|-|-|-|-|-|-|-|-|-|-|-|-|
|키|155|158|165|169|<span style="color:green"> 172</span>|175|181|181|183|188|-|

마지막으로 (4+6)/2 = 5번 째 원소인 172는 172이하이므로, 5번째 미만, 즉 4번 째 이하로는 172가 없다. 새로운 탐색 범위는 &#91;5, 6&#41;이다.

|사람|x|x|x|x|<span style="color:blueviolet">5|<span style="color:blue">6|x|x|x|x|x
|-|-|-|-|-|-|-|-|-|-|-|-|
|키|155|158|165|169|<span style="color:green"> 172</span>|175|181|181|183|188|-

탐색 범위가 하나로 좁혀졌으므로 탐색을 종료한다.

위 과정을 C++코드로 구현하면 다음과 같다.

``` cpp
int height[11] =
{ 0, 175, 181, 172, 158, 155, 169, 183, 188, 165, 181 };
sort(height, height + 11); // 이분 탐색을 위한 데이터 정렬

int target_height = 172;
int low = 1;
int high = 11;
while (high - low > 1 /*탐색 구간의 길이가 1이면 종료*/)  {
    int mid = (low + high) / 2; // 구간의 중앙
    if (height[mid] > target_height) 
        high = mid; // mid이상을 후보에서 제거
    else low = mid; // mid미만을 후보에서 제거
}
if (height[low] == target_height)
    printf("%d번 째 사람의 키가 %dcm이다.", low,
        target_height);
else printf("%d번 째 사람의 키가 %dcm로 %dcm이상 중 가장 앞에 있는 원소다.",
    low, height[low], target_height); 
```

흥미가 있다면 target_height를 173으로 설정하고 코드를 실행해보길 바라며, 다음 포스팅은 그와 관련된 것으로 할 생각이다.

이분 탐색은 정말 빠른 시간 복잡도를 가지지만,   
데이터를 정렬해야 한다는 단점이 있다.  
정렬의 시간 복잡도는 보통 O(nlogn)으로 O(n)의 완전탐색보다 느리다.  

그럼에도 이분 탐색은 데이터를 한번 정렬해 놓기만 하면 검색 쿼리를 log시간에 처리할 수 있으니, 쿼리가 많은 경우라면 이분 탐색 쪽이 훨씬 빠르다.  
(이 문장은 약간의 오류가 있다. 이것도 다음 포스팅에서 다뤄보려고 한다.)

 
 > ## TMI  

 첫 블로그 글을 써보았는데, 마크 다운 문법에 좀 익숙해진 느낌이다.

