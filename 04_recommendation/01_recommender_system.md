# 추천 시스템
- `모형`
    - 1) 베이스라인 모형 (Baseline Model) : 추천 시스템의 기본 모형
    - 2) 협업 필터링 모형 (Collaborative Filtering)
        - 2-1) 이웃기반 협업 필터링 (Neighborhood Models)
            - 사용자 기반 필터링 user-based CF
            - 아이템 기반 필터링 item-based CF
        - 2-2) 잠재 요인 모델 (Latent Factor Models)
            - 행렬 분해 Matrix Factorization
            - 특잇값 분해 Singular Value Decomposition, SVD
            - 음수 미포함 행렬분해 Non-Negative Matrix Factorization, NMF
- `최적화` : 베이스라인 모형의 오차함수를 최소화하기 위한 방법
    - **확률적 경사 하강법 Stochastic Gradient Descent, SGD**
    - **교대 최소 자승법 Alternating Least Squares, ALS**
- `추천시스템 성능 평가기준`        
    - **루트오차제곱평균 Root Mean Squared Error, RMSE**
    - **오차절대값평균 Mean Absolute Error, MAE**
    - **일치쌍의 비율 Fraction of Concordant Pairs, FCP**
- `유사도` : 이웃기반협업필터링 모델에서 사용됨
    - **평균제곱차이 유사도 Mean Squared Difference Similarity, MSD**
    - **코사인 유사도 Cosine Similarity**
    - **피어슨 유사도 Pearson Similarity**
    - **피어슨-베이스라인 유사도 Pearson-Baseline Similarity**
- `KNN 가중치 예측 방법` : 이웃기반협업필터링 모델에서 유사도에 따른 가중치 예측
    - **KNNBasic**
    - **KNNWithMeans**
    - **KNNBaseline**
- `Python`
    - Baseline 모형 : 베이스라인 모형은 최적화 방법을 인수로 설정가능
        - **surprise.Baseline(bsl_options)** 
    - KNN 모형 : 이웃협업필터링 모형은 유사도 방법을 인수로 설정가능
        - **surprise.KNNBasic(sim_options)** 
        - **surprise.KNNWithMeans(sim_options)**
        - **surprise.KNNBaseline(sim_options)**
    - SVD 모형 : Latent Factor 모형의 하나. 차원 축소기능 있음.
        - **surprise.SVD(n_factors=100)**
    - NMF 모형 : Latent Factor 모형의 하나. 차원 축소기능 있음.
        - **surprise.NMF(n_factors=100)**
    - 최적화 : 딕셔너리 타입, Baseline 모형의 인수로 사용가능
        - **bal_options  = {"method":"als", "n_epochs":5, "reg_u":12, "reg_i":5}**
    - 유사도 : 딕녀너리 타입, KNN 모형의 인수로 사용가능
        - **sim_options = {"name" : "msd"}**
    - 모형 성능 평가 : rmse, mae, fcp 선태 가능
        - **surprise.accuracy.rmse(pred, verbose=True)**
    - 교차검증 : surprise 패키지의 model_selection 서브패키지에서 사용 가능
        - **from surprise.model_selection import cross_validate**
        - **cross_validate(model, data, maesures=["rmse", "mae", "fcp"])**
    
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
- 1) 베이스라인 모형 (Baseline Model) : 추천 시스템의 기본 모형
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

#### python
- 딕셔너리 안에 최적화 방법의 인수와 값을 저장한 후 최적화 모형의 인수로 입력한다.
    - bal_options  = {"method":"als", "n_epochs":5, "reg_u":12, "reg_i":5}
    - surprise.Baseline(bls_options)

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
        
#### python
- surprise 패키지의 accuracy 서브패키지의 클래스로 사용가능하다.
    - **surprise.accuracy.rmse(pred, verbose=True)**
        
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
    
#### python
- surprise 패키지의 BaselineOnly() 객체를 사용한다.
    - **model = surprise.BaselineOnly()**
    - 인수로 최적화 옵션을 사용할 수 있다.
- 모형 예측
    - **pred = model.test(test_set)**
- 성능 평가
    - **surprise.accuracy.rmse(pred, verbse=True)**
- cross_validate() 를 사용하면 교차검증 모델에 쉽게 적용가능하다.
    - **cross_validate(surprise.BaselineOnly(), data)**

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

#### python
- surprise 패키지에서 가중치 예측 방법에 따라서 모형을 선택할 수 있다. 인수로 유사도 방법을 설정할 수 있다.
    - **surprise.KNNBasic(sim_options)**
    - **surprise.KNNWithMeans(sim_options)**
    - **surprise.KNNBaseline(sim_options)**
    

### 3-4. Latent Factor
- `잠재요인 모형 Latent Factor Model` : 사용자, 상품에 대한 대규모 데이터에서 몇가지의 특성을 벡터로 간략화(approximate)할 수 있다는 가정에 기반한다.
    - PCA(주성분 분석 Principle Component Analysis)를 사용하여 긴 특성벡터를 소수의 차원으로 차원 축소할 수 있는 것과 같다.
    - 사용자의 특성도 차원 축소 할 수 있다.
- `어떤 사용자의 특정 상품에 대한 평점은 사용자의 특성과 상품의 특성의 내적과 같다.`
    - A 사용자의 영화 장르에 대한 평점 : 액션, 로맨스, 느와르에 장르에 대한 각각의 점수
        - $p(u)^T = (-1, 2, 3)$
    - B 영화의 장르적 특성에 대한 평점 : 액션요소, 로맨스요소, 느와르요소에 대한 각각의 점수
        - $q(i)^T = (2, 1, 1)$
    - 평점 $r(u, i) = q(i)^Tp(u) = -1 \dot 2 + 2 \dot 1 + 3 \dot 1$
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
    
#### pyrhon
- surprise 패키지에서 사용가능. 축소할 차원의 갯수를 인수로 설정 가능
    - **surprise.SVD(n_factors=100)**
    
### 3-7. NMF
- `음수 미포함 행렬분해 Non-Negative Matrix Factorization, NMF` : 음수를 포함하지 않는 행렬 X를 음수를 포함하지 않는 행렬 W와 H의 곱으로 분해하는 알고리즘
    - $X = WH$
    
#### pyrhon
- surprise 패키지에서 사용가능. 축소할 차원의 갯수를 인수로 설정 가능
    - **surprise.NMF(n_factors=100)**    
    
## 4. 유사도
- Neighborhood Models 는 유사한 사용자나 유사한 아이템에 대한 가중치를 적용하는 방식이다.
    - 어떤 사용자와 유사한 사용자들, 어떤 상품과 평점이 유사한 상품들을 찾기위해 유사도를 계산한다.

### 4-1. 유사도 계산
- 사용자 특성 벡터(평점 행렬의 행 벡터), 상품 특성 벡터(평점 행렬의 열 벡터)의 유사도를 비교하는 기준
    - **평균제곱차이 유사도 Mean Squared Difference Similarity**
    - **코사인 유사도 Cosine Similarity**
    - **피어슨 유사도 Pearson Similarity**
    - **피어슨-베이스라인 유사도 Pearson-Baseline Similarity**

#### pyrhon
- 딕셔너리 타입에 저장하여 KNN 모형의 인수로 사용가능.
    - **sim_options = {"name" : "msd"}**
    - **surprise.KNNBasic(sim_options)**

### 4-2. MSD
- `평균제곱차이 유사도 Mean Squared Difference, MSD` : 유클리드 공간에서 거리 제곱에 비례하는 값과 같다.
    - msd 값을 구한 후 역수를 적용한다. msd 값이 0이 될 수 있으므로 1을 더해준다.
- `사용자에 대한 msd 유사도` : 상품 i에 대한 사용자 u와 v의 평점의 차이를 제곱한 후, 사용자 u와 v가 공통으로 평가한 상품의 수로 나누어 준 값. 
    - $\text{msd sim}(u, v) = \dfrac{1}{msd(u,v) + 1}$
    - $msd(u, v) = \dfrac{1}{\vert I_{uv} \vert} \cdot \sum_{i \in I_{uv}} (r(u,i) - r(v, i))^2$
    - $I_{uv}$ : 사용자 u와 사용자 v에 의해 평가된 **상품의 집합**
    - $\vert I_{uv} \vert$ : 사용자 u와 사용자 v에 의해 평가된 **상품의 수**
- `상품에 대한 msd 유사도` : 사용자 u가 매긴 상품 i와 상품 j의 평점의 차이를 제곱한 후, 상품 i와 j를 공통으로 평가한 사용자의 수로 나누어 준 값.
    - $\text{msd sim}(i, j) = \dfrac{1}{msd(i,j) + 1}$
    - $msd(i, j) = \dfrac{1}{\vert U_{ij} \vert} \cdot \sum_{u \in U_{ij}} (r(u,i) - r(u, j))^2$
    - $U_{ij}$ : 상품 i와 상품 j를 모두 평가한 **사용자의 집합**
    - $\vert U_{ij} \vert$ : 상품 i와 상품 j를 모두 평가한 **사용자의 수**
    
### 4-3. Cosine Similarity
- `코사인 유사도 Cosine Similarity` : 두 특성 벡터의 각도에 대한 코사인 값. 두 벡터의 각도 $\theta$가 0이면 코사인 유사도는 1이고, $\theta$가 1이면 코사인 유사도는 0이다.
    - $x \cdot y = {\vert x \vert} {\vert y \vert} \text{cos}\theta$
    - $\text{cosine similarity} = \text{cos}\theta = \dfrac{x \cdot y}{{\vert x \vert} {\vert y \vert}}$
    - 벡터의 내적 공식으로부터 도출
    - **두 벡터가 같은 방향을 가리킬 수록, 즉 닮을 수록 코사인 유사도 값은 1에 가까워진다.**
- `사용자 u, v간의 코사인 유사도`
    - $\text{cosine sim}(u, v) = \dfrac{\underset{i \in I_{uv}}{\sum} r(u,i) \cdot r(v,i)}{\sqrt{\underset{i \in I_{uv}}{\sum} r(u,i)^2} \cdot \sqrt{\underset{i \in I_{uv}}{\sum} r(v,i)^2}}$
- `상품 i, j간의 코사인 유사도`
    - $\text{cosine sim}(i, j) = \dfrac{\underset{u \in U_{ij}}{\sum} r(u,i) \cdot r(u,j)}{\sqrt{\underset{u \in U_{ij}}{\sum} r(u,i)^2} \cdot \sqrt{\underset{u \in U_{ij}}{\sum} r(u,j)^2}}$
    
### 4-4. Pearson Similarity
- `피어슨 유사도 Pearson Similarity` : 두 벡터의 상관계수(pearson correlation coefficient) 값을 의미한다.
- `표본상관계수 sample correlation coefficient`
    - $r_{xy} = \dfrac{S_{x,y}}{\sqrt{S_x^2 \cdot S_y^2}} = \dfrac{\text{x,y의 공분산}}{\sqrt{\text{x의 분산} \cdot \text{y의 분산}}}$
    - $\text{sample covariance} = S_{x,y} = \dfrac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})$
- `사용자 u와 v 간의 피어슨 유사도`
    - $\text{pearson sim}(u, v) = \dfrac{\underset{i \in I_{uv}}{\sum} (r(u,i) - \mu(u)) \cdot (r(v,i) - \mu(v))}{\sqrt{\underset{i \in I_{uv}}{\sum} (r(u,i) - \mu(u))^2} \cdot \sqrt{\underset{i \in I_{uv}}{\sum} (r(v,i) - \mu(v))^2}}$
    - $\mu(u), \mu(v)$ : 사용자 u와 v가 매긴 평점 평균
- `상품 i와 j간의 피어슨 유사도`
    - $\text{pearson sim}(i, j) = \dfrac{\underset{u \in U_{ij}}{\sum} (r(u,i) - \mu(i)) \cdot (r(u,j) - \mu(j))}{\sqrt{\underset{u \in U_{ij}}{\sum} (r(u,i) - \mu(i))^2} \cdot \sqrt{\underset{u \in U_{ij}}{\sum} (r(u,j) - \mu(j))^2}}$
    - $\mu(i), \mu(j)$ : 상품 i와 j가 받은 평점 평균
    
### 4-5. Pearson-Baseline Similarity
- `피어슨-베이스라인 유사도 ` : 피어슨 유사도와 같이 상관계수를 구하지만, 각 벡터의 기댓값을 단순 평균이 아니라 베이스라인 모형(Baseline Model)에서 예측한 값을 사용한는 방법
    - **Baseline Model** : 추천시스템의 기본모형. 사용자 아이디 u와 상품 아이디 i, 두 카테고리 입력변수와 평점인 실수 종속변수 r(u, i)의 예측치 평점 $\hat{r}(u, i)$을 예측하는 회귀분석모형. 최적화를 통해 오차함수를 최소화하는 방식으로 계산된다.
- `사용자 u, v 간의 msd(mean squared difference sim)` 
    - $\text{pearson baseline sim}(u, v) = \hat{\rho}_{uv} = \dfrac{\underset{i \in I_{uv}}{\sum} (r(u,i) - b(u,i)) \cdot (r(v,i) - b(v,i))}{\sqrt{\underset{i \in I_{uv}}{\sum} (r(u,i) - b(u,i))^2} \cdot \sqrt{\underset{i \in I_{uv}}{\sum} (r(v,i) - b(v,i))^2}}$
    - 피어슨 유사도와 구조는 같다. u와 v의 평점 평균 대신에 베이스라인 모형의 평점 조정값 b(u,i)와 b(v,i)가 사용된다.
- `상품 i와 j 간의 msd(mean squared difference sim)`
    - $\text{pearson baseline sim}(i, j) = \hat{\rho}_{ij} = \dfrac{\underset{u \in U_{ij}}{\sum} (r(u,i) - b(u,i)) \cdot (r(u,j) - b(u,j))}{\sqrt{\underset{u \in U_{ij}}{\sum} (r(u,i) - b(u,i))^2} \cdot \sqrt{\underset{u \in U_{ij}}{\sum} (r(u,j) - b(u,j))^2}}$
    - 피어슨 유사도와 구조는 같다. i와 j의 평점 평균 대신에 베이스라인 모형의 평점 조정값 b(u,i)와 b(u,j)가 사용된다.
- `shrinkage 정규화` : 피어슨-베이스라인 모형을 두 사용자간 또는 두 상품간의 평점 중 같은 평점 원소의 갯수를 이용하여 정규화하는 방법
    - $\text{pearson baseline shrunksim}(u,v) = \dfrac{{\vert I_{uv} \vert} - 1}{{\vert I_{uv} \vert} - 1 + \text{shrinkage}} \cdot \hat{\rho}_{uv}$
    - $\text{pearson baseline shrunk sim}(i,j) = \dfrac{{\vert U_{ij} \vert} - 1}{{\vert U_{ij} \vert} - 1 + \text{shrinkage}} \cdot \hat{\rho}_{ij}$
    
## 5. KNN 가중치 예측 방법
- `KNN 기반 예측 방법 K Nearest Neighbors` : 이웃기반협업필터링 Neighborhood Filtering 방법의 가중치 예측 방법에 해당한다. 평점을 구하고자 하는 사용자(또는 상품)의 유사도를 구한 뒤 유사도가 큰 k개의 사용자(또는 상품) 벡터를 사용하여 가중 평균을 구해서 가중치를 예측하는 방법.
- `surprise 패키지의 KNN 기반 가중치 예측 방법`
    - **KNNBasic**
    - **KNNWithMeans**
    - **KNNBaseline**
    
#### python
- surprise 패키지에서 사용가능. 인수로 유사도를 설정 할 수 있다.
    - **surprise.KNNBasic(sim_options)**
    
### 5-1. KNNBasic    
- `KNNBasic` : 평점들을 단순히 가중 평균하는 방식. $N^k$는 유사도가 가장 큰 k개의 벡터의 집합
- `사용자간 유사도를 기반으로한 가중치의 예측 평점`    
    - $\hat{r}(u,v) = \dfrac{\underset{v \in N_i^k (u)}{\sum} \text{sim}(u,v) \cdot r(v,i)}{\underset{v \in N_i^k (u)}{\sum} \text{sim} \text{sim}(u,v)}$
    - $v \in N_i^k (u)$ : 유사도가 가장 큰 k개의 상품 i 벡터의 집합 중 사용자 u의 벡터에 포함되는 사용자 v의 벡터 
- `상품간 유사도를 기반으로한 가중치의 예측 평점`
    - $\hat{r}(i,j) = \dfrac{\underset{j \in N_u^k (i)}{\sum} \text{sim}(i,j) \cdot r(u,j)}{\underset{j \in N_u^k (j)}{\sum} \text{sim} \text{sim}(i,j)}$
    - $i \in N_u^k (j)$ : 유사도가 가장 큰 k개의 사용자 u 벡터의 집합 중 상품 j의 벡터에 포함되는 사용자 i의 벡터
    
### 5-2. KNNWithMeans
- `KNNWithMeans` : 평점들을 평균값 기준으로 가중 평균해준다.
- `사용자간 유사도를 기반으로한 가중치의 예측 평점`
    - $\hat{r}(u,i) = \mu(u) + \dfrac{\underset{v \in N_i^k (u)}{\sum} \text{sim}(u,v) \cdot (r(v,i)-\mu(v))}{\underset{v \in N_i^k (u)}{\sum} \text{sim} \text{sim}(u,v)}$
- `상품간 유사도를 기반으로한 가중치의 예측 평점`
    - $\hat{r}(u,i) = \mu(i) + \dfrac{\underset{j \in N_u^k (i)}{\sum} \text{sim}(i,j) \cdot (r(u,j) - \mu(j))}{\underset{j \in N_u^k (j)}{\sum} \text{sim} \text{sim}(i,j)}$
    
### 5-3. KNNBaseline
- `KNNBaseline` : 평점들을 베이스라인 모형의 값 기준으로 가중 평균한다.
- `사용자간 유사도를 기반으로한 가중치의 예측 평점`
    - $\hat{r}(u,i) = b(u,i) + \dfrac{\underset{v \in N_i^k (u)}{\sum} \text{sim}(u,v) \cdot (r(v,i)-b(v, i))}{\underset{v \in N_i^k (u)}{\sum} \text{sim} \text{sim}(u,v)}$
- `상품간 유사도를 기반으로한 가중치의 예측 평점`
    - $\hat{r}(u,i) = b(u,i) + \dfrac{\underset{j \in N_u^k (i)}{\sum} \text{sim}(i,j) \cdot (r(u,j) - b(u,j))}{\underset{j \in N_u^k (j)}{\sum} \text{sim} \text{sim}(i,j)}$    















































