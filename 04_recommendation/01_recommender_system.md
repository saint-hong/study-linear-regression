# 추천 시스템

## 1. 추천 시스템
- `추천 시스템 Recommender System` : 사용자(user)가 선호하는 상품(Item)을 예측하는 시스템
    - 사용자가 아이템에 대해서 평가한 점수를 기반으로 예측한다.
    - **사용자 아이디와 상품 아이디라는 두 개의 카테고리 입력과 평점 출력을 갖는 예측 시스템**
- `추천시스템은 회귀모형을 따른다.`
    - user와 item 이라는 카테고리 독립변수와 rate 라는 실수형 종속변수를 토대로 평점을 예측하는 회귀모형을 따른다.

### 1-1. 평점 데이터
- `사용자 ID, 상품 ID, 평점으로 이루어진 행렬`
    - 독립변수 : user, item
    - 종속변수 : rate
- `R matrix`
    - x축 : item
    - y축 : user
    - value : rate
    - pivot_tabel(["user", "item"]).unstack()
    - user와 item을 이중 인덱스로 만들고, unstack() 을 사용하여 item을 열로 변환
- 평점 데이터가 일부에만 있다. : sparse 행렬
    - user가 모든 item에 대해서 평점을 매긴 것이 아니므로 평점이 비어 있게 된다.
    - 비어 있는 곳을 어떻게 채우냐에 따라서 예측 방법이 달라질 수 있다.
    
## 2. 추천시스템 알고리즘
- 추천 시스템은 두 개의 카테고리 값 입력으로부터 하나의 실수 값을 출력하는 **회귀모형**
    - 회귀분석과 유사하지만 여러가지 방법으로 성능을 증가시킬 수 있다.

### 2-1. 추천시스템 모형
- 1) 베이스라인 모형 (Baseline Model)
- 2) 협업 필터링 모형 (Collaborative Filtering)
    - 2-1) 이웃기반 협업 필터링 (Neighborhood Models)
        - 사용자 기반 필터링 (user-based CF)
        - 아이템 기반 필터링 (item-based CF)
    - 2-2) 잠재 요인 모델 (Latent Factor Models)
        - 행렬 분해 (Matrix Factorization)
        - 특잇값 분해 (Singular Value Decomposition, SVD)

### 2-2. 최적화
- 오차함수를 최소화하기 위한 최적화 알고리즘
- `확률적 경사 하강법 Stochastic Gradient Descent, SGD ` 클래스의 인수
    - reg : 정규화 가중치, 디폴트 0.02 : 모형의 과최적화 방지
    - learning_rate : 최적화 스텝사이즈, 디폴트 0.005 : 스텝사이즈에 따라서 모형의 성능이 달라질 수 있다.
    - n_epochs : 최적화 반복 횟수, 디폴트 20
- `교대 최소 자승법 Alternating Least Squares` 클래스의 인수
    - reg_i : 상품에 대한 정규화 가중치, 디폴트 10 : b(i)의 값
    - reg_u : 사용자에 대한 정규화 가중치, 디폴트 15 : b(u)의 값
    - n_epochs : 최적화 반복 횟수, 디폴트 10

### 2-3. 추천성능 평가기준
- `루트오차제곱평균 Root Mean Squared Error, RMSE`
    - $\text{RMSE} = \sqrt{\dfrac{1}{\vert \hat{R} \vert} \sum_{\hat{r}(u,i)\in \hat{R}} (r(u, i) - \hat{r}(u, i))^2}$
    - 원래 평점과 예측 평점의 차이를 제곱한 후 테스트 데이터 셋으로 나눈 값에 루트 변환한다.
    - $\hat{R}$ 은 테스트 데이터셋
- `오차절대값평균 Mean Absolute Error, MAE`
    - $\text{MAE} = \dfrac{1}{\vert \hat{R} \vert} \sum_{\hat{r}(u,i) \in \hat{R}} \vert r(u, i) - \hat{r}(u, i) \vert$
    - 원래 평점과 예측 평점의 차이에 절대값 적용한 후 테스트 데이터 셋으로 나눈 값
- `일치쌍의 비율 Fraction of Concordant Pairs, FCP`
    - $\text{FCP} = \dfrac{\text{number of concordant paris}}{\text{number of discordant pairs}}$
    - concordant pairs : 회귀분석에서 i, j번째 데이터에 대한 yi, yj와 예측한 값 $\hat{y}_i, \hat{y}_j$ 사이의 **증가 방향**이 같은 경우
        - $\text{sign}(y_i - y_j) = \text{sign}(\hat{y}_i - \hat{y}_j)$
        
## 3. 모형

### 3-1. 베이스라인 모형
- `베이스라인 모형 Baseline Model` : 사용자 아이디 u, 상품 아이디 i, 두 개의 카테고리값으로부터 평점인 r(u, i)의 예측값 $\hat{r}(u, i)$ 를 예측하는 회귀모형이다.
    - r(u, i) : u 사용자가 i 아이템에 매긴 평점
    - $\hat{r}(u, i)$ : 모형이 u 사용자의 i 아이템에 대한 평점을 예측한 값
    - $\hat{r}(u, i) = \mu + b(u) + b(i)$
    - mu : 전체 평점의 평균값
    - b(u) : 동일한 사용자에 의한 평점 조정값
    - b(i) : 동일한 아이템에 의한 평점 조정값
- `오차함수` : 베이스라인 모형의 모수추정은 오차함수를 최소화하는 방향으로 구해진다.
    - $\underset{u,i \in R_{train}}{\sum} (r(u, i) - \hat{r}(u, i))^2$
    - R_train 은 실제 데이터, 즉 u, i 는 실제 데이터에 포함된 데이터
- `정규화`
    - 과최적화를 방지하기 위한 방법 : 학습 데이터에 과도하게 최적화 되는 현상. 검증 데이터에서는 성능이 떨어진다.
    - $\underset{u,i \in R_{train}}{\sum} (r(u,i)-\hat{r}(u, i))^2 + \lambda (b(u)^2 + b(i)^2)$
    - $\hat{r}(u, i)^2$의 계산에서 나오는 b(u), b(i) 값을 상쇄 시켜준다.


### 3-2. Collaborative Filtering
- `협업필터링 Collaborative Filtering, CF` : 평점 행렬이 가진 특정한 패턴을 찾아서 이것을 평점 예측에 사용하는 방법
    - Neighborhood 모형 : 사용자나 아이템 기준으로 평점의 유사성을 찾는 방법
        - user-based CF
        - item-based CF
    - Latent Factor 모형 : 행렬의 수치적 특징을 이용하는 방법
        - Matrix Factorization 
        - SVD
    
### 3-3. Neighborhood
- `이웃협업필터링 Neighborhood Models` : Memory-based CF 라고도 한다. 해당 사용자와 유사한(similar) 사용자에 대해 가중치를 준다.
    - 특정 사용자의 평점을 예측하는 것과 다르다.
    - A 사용자가 item1, item2, item3 에 대해서 2.3, 3.5, 1.7 이라는 평점을 줬을때, 이와 유사한 다른 사용자에 대해서 가중치를 준다.
- `user-based CF` : 해당 사용자와 유사한 사용자를 찾는 방법, 평점 행렬 R에서 유사한 사용자 행 벡터를 찾고 이를 기반으로 빈 데이터를 예측하는 방식
    - 사용자 P의 평점 데이터에서 비어 있는 상품을 채워넣기 위해서 P와 유사한 평점을 준 다른 사용자들을 기반으로 가장 적당한 값을 채워 넣어 P의 평점을 완성한다.
- `item-based CF` : 특정한 상품에 대해 사용자가 준 점수를 찾는 방법, 평점 행렬 R에서 상품 열 벡터의 유사성을 찾고 이 상품들의 평점 정보로 해당 상품의 빈 평점을 예측하는 방법
    - 어떤 상품 J의 평점이 비어 있을때 유사성을 가진 다른 상품들의 평점 정보를 바탕으로 J의 비어있는 평점을 예측한다.
- 사용자기반, 아이템기반 둘다 유사한 다른 대상들의 평점정보를 바탕으로 비어있는 평점을 채워넣는 방식이다.

### 3-4. Latent Factor
- `잠재요인 모형 Latent Factor Model` : 사용자, 상품에 대한 대규모 데이터에서 몇가지의 특성을 벡터로 간략화(approximate)할 수 있다는 가정에 기반한다.
    - PCA(주성분 분석 Principle Component Analysis)를 사용하여 긴 특성벡터를 소수의 차원으로 차원 축소할 수 있는 것과 같다.
    - 사용자의 특성도 차원 축소 할 수 있다.
- `어떤 사용자의 특정 상품에 대한 평점은 사용자의 특성과 상품의 특성의 내적과 같다.`
    - A 사용자의 영화 장르에 대한 평점 : 액션, 로맨스, 느와르에 장르에 대한 각각의 점수
        - $p(u)^T = (-1, 2, 3)$
    - B 영화의 장르적 특성에 대한 평점 : 액션요소, 로맨스요소, 느와르요소에 대한 각각의 점수
        - $q(i)^T = (2, 1, 1)$
    - 평점 $r(u, i) = q(i)^Tp(u) = -1\dot2 + 2\dot1 + 3\dot+1$
- 즉 사용자가 어떤 대상에 대해 매긴 평점들과 어떤 상품이 이 대상들을 어느정도 요소로 지녔지에 대한 평점들을 내적하면, 특정사용자의 특정 상품에 대한 평점을 계산할 수 있다.
- 베이스라인 모형의 예측값 $\hat{r}(u, i)$와는 다른 의미이다.

### 3-5. Matrix Factorization
- `행렬인수분해 모델 Matrix Factorization` : 모든 사용자와 모든 상품에 대해 아래의 오차함수를 최소화하는 요인 벡터를 찾는다. 즉 P, Q 행렬을 찾는다.
    - $R \approx PQ^T$
    - $R \in \text{R}^{m \times n}$ : m 사용자와 n 상품의 **평점 행렬**
    - $P \in \text{R}^{m \times k}$ : m 사용자와 k 요인의 **관계 행렬**
    - $Q \in \text{R}^{n \times k}$ : n 상품과 k 요인의 **관계 행렬**
    
### 3-6. SVD
- `특잇값분해 모델 Singular Value Decomposition, SVD` : 행렬의 특잇값 분해를 사용하여 Matrix Factorization(행렬 인수분해) 문제를 푸는 한 방법. 
     - (m, n) 크기의 행렬을 특잇값분해 한 후 공분산행렬의 가장 큰 k개의 특이값을 사용하여 원래 행렬과 크기가 같고 원소가 유사한 새로운 특이행렬을 만든다.
     - $R = U \Sigma V^T$ 에서 $\hat{U} \hat{\Sigma} \hat{V}^T = \hat{R}$ 로 만든다.
- $\Sigma$ 행렬의 가장 큰 k개의 값을 사용하여 새로운 행렬을 만든다.
    - $\hat{U}$는 **U에서** 가장 큰 k개의 특이값에 대응하는 성분으로 만든 mxk의 행렬
    - $\hat{\Sigma}$는 가장 값이 큰 k개의 특이값에 대응하는 성분으로 만든 kxk의 행렬
    - $\hat{V}$는 **V에서** 가장 큰 k개의 특이값에 대응하는 성분으로 만든 kxn의 행렬
- 평점 행렬은 빈 데이터가 많은 sparse 행렬이므로 SVD를 적용하기 어렵다.
- 따라서 관계행렬 P, Q는 다음과 같은 오차함수를 최소화하여 구한다.
    - $\sum_{u,i \in R_{train}} (r(u, i) - \hat{r}(u, i))^2 + \lambda \left( b(i)^2 + b(u)^2 + {\Vert q(i) \Vert}^2 + {\Vert p(u) \Vert}^2 \right)$
    - $\hat{r}(u, i) = \mu + b(u) + b(i) + q(i)^Tp(u)$
    
### 3-7. NMF
- `음수 미포함 행렬분해 Non-Negative Matrix Factorization, NMF` : 음수를 포함하지 않는 행렬 X를 음수를 포함하지 않는 행렬 W와 H의 곱으로 분해하는 알고리즘
    - $X = WH$
    
## 4. 유사도
- Neighborhood Models 는 유사한 사용자나 유사한 아이템에 대한 가중치를 적용하는 방식이다.
    - 어떤 사용자와 유사한 사용자들, 어떤 상품과 평점이 유사한 상품들을 찾기위해 유사도를 계산한다.

### 4-1. 유사도 계산
- 사용자 특성 벡터(평점 행렬의 행 벡터), 상품 특성 벡터(평점 행렬의 열 벡터)의 유사도를 비교하는 기준
    - **평균제곱차이 유사도 Mean Squared Difference Similarity**
    - **코사인 유사도 Cosine Similarity**
    - **피어슨 유사도 Pearson Similarity**
    - **피어슨-베이스라인 유사도 Pearson-Baseline Similarity**
    
### 4-2. MSD
- `평균제곱차이 유사도 Mean Squared Difference, MSD` : 유클리드 공간에서 거리 제곱에 비례하는 값과 같다.
    - msd 값을 구한 후 역수를 적용한다. msd 값이 0이 될 수 있으므로 1을 더해준다.
- 사용자에 대한 msd 유사도
    - $\text{msd sim}(u, v) = \dfrac{1}{msd(u,v) + 1}$
    - $msd(u, v) = \dfrac{1}{\vert I_{uv} \vert} \cdot \sum_{i \in I_{uv}} (r(u,i) - r(v, i))^2$
- 상품에 대한 msd 유사도
    - $\text{msd sim}(i, j) = \dfrac{1}{msd(i,j) + 1}$
    - $msd(i, j) = \dfrac{1}{\vert U_{ij} \vert} \cdot \sum_{u \in U_{ij}} (r(u,i) - r(u, j))^2$
- $U_{ij}$ : 상품 i와 상품 j를 모두 평가한 **사용자의 집합**
- $\vert U_{ij} \vert$ : 상품 i와 상품 j를 모두 평가한 **사용자의 수**




/usr/bin/bash: +: command not found

















































