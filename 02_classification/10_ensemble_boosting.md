# 부스팅 방법

## 1. 특징
- 여러개의 약한 분류기(weak lerner)가 순차적으로 학습을 진행하는 방식
- 이전 분류기가 예측을 틀린 데이터에 대해 다음 분류기가 가중치를 인가해서 학습을 진행
- 약한 분류기들의 이러한 학습과정을 통해 강한 분류기가 된다.
    - weak learner -> output 1, output 2 , ... -> strong learner
- 예측 성능이 뛰어나 앙상블 학습에 많이 사용된다.
    - GradientBoosting, 
    - XGBoost(extra gradient boost), 
    - LightGBM(Light gradient Boost) 등
- `Boostrap Aggregating : bagging 방식`
    - parallel : 동시에 병렬적으로 결과를 얻음
    - 여러 분류기가 동시에 학습, 예측
- `boosting 방식`
    - sequential : 순차적으로 진행됨 
    - 여러 분류기가 순차적으로 학습, 예측, 이전 분류기의 결과를 이어받아 다음 분류기가 결과 예측

## 2. 이론

- `모형 결합 model combining` : 앙상블 방법론 ensemble mdehods, 여러개의 예측 모형을 결합하여 더 나은 성능을 예측하는 방법
    - 취합방법 aggregation
        - 다수결 방법 majority voting
        - 배깅 방법 bagging
        - 랜던포레스트 random forest
    - 부스팅 방법 boosting
        - 에이다 부스트 ada boost
        - 그래디언트 부스트 gradient boost
- `부스트 boost` : 하나의 모형에서 시작하여 모형 집합에 포함할 개별 모형을 하나씩 추가하는 방법
    - voting, bagging, RF 모형 등은 개별 모형의 갯수가 미리 정해진다.
- `위원회 commitee` : 모형의 집합, C
    - m개의 개별모형을 포함하는 위원회, 모형 집합 : $C_m$
    - 개별 모형 : 약 분류기 (weak classifier) : k
    - $C_1 =  \left\{ k_{1} \right} $ 
    - $C_m=C_{m-1} \cup k_m = \left\{ k_1, k_2, \cdots, k_m \right\}$
    - 이전 위원회에 개별모형 k_m이 추가 된다.
- `k_m의 선택 방법`
    - $C_{m-1}$의 성능을 보완하는 것을 선택한다.
- `Cm의 최종 결정 방법 : m개의 모형을 결합한 Cm의 분류 예측`
    - 개별 모형의 예측값들에 가중치 $\alpha$를 가중선형조합한 값을 판별함수로 사용
    - 이 판별함수에 따라서 Cm의 예측값이 결정된다.
    - $C_m (x_i) = \text{sign} (\alpha_1 k_1 (x_i) + \cdots + \alpha_{m} k_m (x_i)$
    - $\alpha$는 개별모형에 대한 가중치
- `부스트 방법은 이진분류에만 사용할 수 있다.`
    - $y= -1 \;\text{or}\; 1$
    - 정상-스펨, 환자-비환자, 개-고양이 등의 이진분류에만 적용가능
    - 이러한 이진분류의 특성에 따라서 개별모형의 가중치 등을 계산한다.

## 3. 에이다부스트
- `에이다부스트 ada boost` : 적응 부스트(adaptive boost). k_m을 선택하는 방법으로 학습 데이터의 i번째 데이터에 가중치 wi를 주고 분류 모형이 틀리게 예측한 데이터들의 가중치를 합한 값을 손실함수 L로 사용한다. 이 손실함수 L을 최소화하는 모형 k_m을 선택한다.
- `손실함수 L_m`    
    - $L_m = \sum_{i=1}^{N} w_{m,i} I(k_m(x_i) \neq y_i)$
    - I는 지시함수(indicator function) : 개별모형의 예측값과 종속변수의 값이 다르면 1, 같으면 0을 반환한다.
    - 즉 에이다부스트의 손실함수는 예측이 틀린 데이터의 가중치들을 모두 더한 값과 같다.
    - $w_i$는 i번째 데이터에 대한 가중치
- `개별모형의 가중치 alpha_m`
    - $\epsilon_m = \dfrac{\sum_{i=1}^{N} w_{m,i} I(k_m(x_i) \neq y_i)}{\sum_{i=1}^{N} w_{m,i}}$
        - 개별모형의 예측 틀린 데이터의 가중치의 합 / 전체 데이터의 가중치의 합
        - 즉 k_m 후보 모형의 틀린 예측의 데이터의 가중치의 비중
        - 후보 모형별로 예측을 틀리게 한 비중이 다르다.
    - $\alpha_m = \dfrac{1}{2} \text{log} \left( \dfrac{1-\epsilon_m}{\epsilon_m} \right)$
        - alpha는 모형에 대한 가중치, w는 데이터에 대한 가중치
- `Adaboost 진행 과정`
    - positive, negative 들 중에서 positive 를 분류하는 결정경계-1 설정
    - 결정경계-1 에서 잘 못 분류된 것들 중 pos 인데 neg 로 분류한 것들(틀린 예측)에 가중치 부여(boosting)해서 결정경계-2 설정
    - 결정경계-2 에서 잘 못 분류된 것들 중 neg 인데 pos 로 분류한 것들(틀린 예측)에 가중치를 부여해서 결정경계-3 설정
    - 이렇게 생성한 결정경계들을 합쳐서 정확도를 높임

### 3-1. 데이터 샘플에 대한 가중치 w
- 데이터에 대한 가중치 w_i는 최초에는 모든 데이터에 같은 균일한 값이지만, 위원회가 증가할때마다 값이 변한다.

       
    - <img src="https://latex.codecogs.com/gif.latex?w_%7Bm%2C%20i%7D%20%3D%20w_%7Bm-1%2C%20i%7D%20exp%28-y_i%20C_%7Bm-1%7D%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%5C%3Bw_%7Bm-1%2Ci%7D%20e%5E%7B-1%7D%20%5C%3B%5C%3B%20if%20%5C%3B%20C_%7Bm-1%7D%20%3D%20y_i%20%5C%5C%20%5C%3Bw_%7Bm-1%2Ci%7D%20e%20%5C%3B%5C%3B%5C%3B%20if%20%5C%3B%20C_%7Bm-1%7D%20%5Cneq%20y_i%20%5Cend%7Bmatrix%7D%5Cright.">


    - 예측이 맞은 경우 가중치 값은 작아지고, 예측이 틀린 경우 가중치 값은 크게 확대(boosting) 된다.
    - m번째 개별 모형의 모든 후보에 대해 이 손실함수 L을 적용하여 값이 가장 작은 후보 를 선택한다.

### 3-2. 에이다부스팅의 손실함수
- 손실함수 L_m    
    - $L_m = \sum_{i=1}^{N} exp(-y_i C_m (x_i))$
    - $C_m(x_i) = \sum_{j=1}^{m} \alpha_j k_j (x_i) = C_{m-1}(x_i) + \alpha_m k_m(x_i)$
        - 개별모형의 결과와 가중치의 선형조합형태
    - C_m 식을 손실함수 L에 대입하여 정리

    - <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20L_m%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20exp%28-y_i%20C_m%20%28x_i%29%29%20%5C%5C%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20exp%28-y_i%20C_%7Bm-1%7D%28x_i%29%20-%20%5Calpha_m%20y_i%20k_m%20%28x_i%29%29%20%5C%5C%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20exp%28-y_i%20C_%7Bm-1%7D%28x_i%29%29%20exp%28-%20%5Calpha_m%20y_i%20k_m%20%28x_i%29%29%5C%5C%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20w_%7Bm%2Ci%7D%20exp%28-%20%5Calpha_m%20y_i%20k_m%20%28x_i%29%29%20%5Cend%7Baligned%7D">


    - 종속변수 y_i와 개별모형 k_m(x_i)의 값은 1 또는 -1 만 가질 수 있으므로,
    
    - <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20L_m%20%26%3D%20e%5E%7B-%5Calpha_m%7D%20%5Csum_%7Bk_m%28x_i%29%20%3D%20y_i%7D%20w_%7Bm%2Ci%7D%20&plus;%20e%5E%7B%5Calpha_m%7D%20%5Csum_%7Bk_m%28x_i%29%20%5Cneq%20y_i%7D%20w_%7Bm%2Ci%7D%5C%5C%20%26%3D%20%28e%5E%7B%5Calpha_m%7D%20-%20e%5E%7B-%5Calpha_m%7D%29%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20w_%7Bm%2Ci%7D%20I%28k_m%28x_i%29%20%5Cneq%20y_i%29%20&plus;%20e%5E%7B-%5Calpha_m%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20w_%7Bm%2Ci%7D%20%5Cend%7Baligned%7D">
    
    - **L_m을 최소화하는 alpha_m**
    - $\dfrac{d L_m}{d \alpha_m} = 0$
    
### 3-3. 에이다 부스트 모형의 정규화
- 모형의 과최적화를 방지하기 위해 학습 속도(learning rate)를 조정하여 정규화를 한다.
    - $C_m = C_{m-1} + \mu \alpha_m k_m$
    - $\mu$ : learning rate
    - 필요한 멤버의 수를 강제로 증가시켜서 과최적화를 막는 역할을 한다.
- learning rate 값이 1보다 작으면 새로운 멤버(k_m)의 가중치 $\alpha_m$를 강제로 낮춘다.
- learning rate 값이 1보다 크면 성능은 크게 저하된다.

## 4. 그래디언트 부스트
- `그래디언트 부스트 gradient boost` : 변분법(calculus of variations)을 사용한 모형
    - 변분법 : 범함수의 미분 : 입력 함수가 변화할 때 출력이 어떻게 달라지는지 계산
    - 범함수 functional : 함수를 입력받아 실수를 출력하는 함수, 기댓값, 엔트로피 등이 범함수에 해당함
- `그래디언트 디센트 gradient descent` : 경사하강법
    - $x_m = x_{m-1} - \alpha_m \dfrac{df}{dx}$
    - 함수의 현재 위치에서 다음 위치를 정하기 위한 방법 : 점차 최소값에 다가간다.
    - 스텝사이즈(=alpha=mu)에 따라서 수렴이 아닌 발산을 하기도 한다.
- `그래디언트 부스트 모형`    
    - $C_m = C_{m-1} - \alpha_m \dfrac{\delta L(y, C_{m-1})}{\delta C_{m-1}} = C_{m-1} + \alpha_m k_m$
    - $L(y, C_{m-1})$ : 손실 범함수
    - 손실 범함수 값을 최소화하는 개별 분류 모형 k_m을 찾는다.
- **그래디언트 부스트 방법은 회귀와 분류 문제에 상관없이 회귀 분석 방법을 사용한다.**
    - 의사결정 회귀나무(decision tree regression model) 모형을 주로 사용한다.
- GBM (Gradient Boosting Machine)
    - AdaBoost 기법과 비슷함
    - 가중치를 업데이트할 때 경사하강법 gredient descent 를 사용한다.

### 4-1. 그래디언트 부스트 모형의 계산 방식
- `계산 방식`
    - $- \dfrac{\delta L(y, C_m)}{\delta C_m}$ 를 목표값으로 개별 멤버 모형 k_m을 찾는다.
    - $(y - (C_{m-1} + \alpha_m k_m))^2$ 를 최소화하는 스텝사이즈 $\alpha_m$을 찾는다.
        - y와 예측값의 차이의 크기
    - $C_m = C_{m-1} + \alpha_m k_m$ 최종 모형을 갱신한다.
        - 이 과정을 반복하여 개별 모형과 가중치를 계산한다.
- `손실 범함수가 오차 제곱 형태인 경우`
    - $L(y, C_{m-1}) = \dfrac{1}{2} (y - C_{m-1})^2$
- `범함수의 미분은 실제 목표값 y와 C_{m-1}의 차이, 즉 잔차가 된다.`
    - $- \dfrac{d L(y, C_m)}{d C_m} = y - C_{m-1}$

### 4-2. python
- from sklearn.ensemble import GradientBoostingClassifier
    
## 5. XGBoost,  라이브러리
- `XGBoost (Extream Gradient Boosting)` : 그래디언트 부스팅 머신을 보완한 분류 모형
- `그래디언트 부스팅` : Friedman의 논문 "Greedy Function Approximation" 에서 "A Gradient Boosting Machine(GBM)"에서 유래됨
    - 위원회 방식 : 매 실행마다 손실함수를 최소화하는 m번째 약분류기를 추가하는 방식.이 과정에서 데이터샘플의 가중치와 약분류기의 가중치를 이전 위원회로부터 이어받는데 틀린 예측의 가중치를 확대하고, 맞은 예측의 가중치를 작게하는 방식으로 현재 모델의 가중치를 찾는다.  
    - 범함수 형태의 손실함수(L(y, C_m))를 최소하화하는 k_m을 찾기 위해 변분법을 사용하는 방식
    - 회귀와 분류문제 모두 회귀모형을 사용하며, "의사결정 회귀나무"를 주로 사용한다.
- `xgb 장점`
    - GBM에서 pc의 파워를 효율적으로 사용하기 위한 다양한 기법에 채택되어 빠른 속도와 효율을 가진다.

### 5-1. 이론적 배경
- `지도학습의 수학적 모형에 기반한다.`
    - $\hat{y} = \sum_{j} \theta_j, x_{ij}$  
- 회귀, 분류, 순위 지정과 같은 문제에 적용할 수 있다.
    - XGBLogisticRegressor
    - XGBClassifier
    - XGBRanking
- 목적함수의 구성 : 학습 데이터의 손실함수 + 정규화 항
    - $\text{obj}(\theta) = L(\theta) + \Omega(\theta)$
- 손실함수 L: 훈련 데이터에 대한 모형의 예측 정확도를 평균제곱오차(mse)로 측정함
    - $L(\theta) = \sum_{i} = (y_i - \hat{y}_2)^2$
    - 로지스틱 손실함수가 사용되기도 함
- 정규화 항 : 모델의 복잡성을 제어하여 과적합을 방지하는 역할
    - 학습 데이터에 대해 모형의 정규화 항의 갯수와 손실함수의 크기가 적당해야 한다.
    - 적당한 모형을 찾기 위해 편향-분산 트레이드 오프를 사용한다.
- `의사 결정 나무에 기반한 선택 과정을 따른다.`
    - 트리 앙상블 모델의 특징 : 회귀와 분류 문제에 사용할 수 있다. (CART)
    - DT는 노드에 결정값만 포함되지만 CART 모델은 결정값(점수)이 각 리프와 연결되어 있는 특징이 있다. 더 풍부한 해석이 가능하다.
    - 앙상블 트리는 개별 트리들의 결과값을 합산해주므로 DT 보다 더 유의미하다.
- `랜덤포레스트`
    - DT를 개별모형으로 모형결합하는 방식
    - 각 노드에서 모든 독립변수를 비교하는 것이 아니라, 랜덤하게 독립변수의 특징차원을 감소시키다.
    - 이 독립변수들을 비교한 후 노드를 분리할 최적의 변수를 선택한다. 개별 모형(DT)사이의 상관관계가 줄어들어 모형 성능의 변동이 감소한다. 즉 모형의 성능이 안정화 된다.
    - Extream Randomized Trees 모형은 노드 분리시 랜덤하게 독립변수의 차원을 줄이는 것이 아니라 아예 랜덤하게 독립변수를 선택한다.
- `앙상블 나무 모형`
    - 개별나무 전체를 평가하는 것이 아니라 각 단계마다 새로운 개별나무를 추가해나가는 방식
    - $\hat{y}_i^{t} = \sum_{k=1}^{t} f_k(x_i) = \hat{y}_i^{t-1} + f_t(x_i)$
    - 목적함수 : $\begin{aligned}
\text{obj}^{(t)} 
&= \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t)}) + \sum_{i=1}^{t} \omega (f_i) \\
&= \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t (x_i)) + \omega (f_i) + constant 
\end{aligned}$
    - 목저감수의 손실함수를 MSE를 사용하면 수식정리가 간단하지만, 로지스틱 손실함수를 사용하면 테일러 확장을 적용해야하므로 정리가 쉽지 않다.
- `XGBoost 모형의 목적함수` : 로그손실 값을 손실함수로 사용
    - $\sum_{i=1}^{n} [g_i f_t(x_i) + \dfrac{1}{2} h_i f_t^{2} (x_i)] + \omega (f_t)$
    - 이 목적함수를 최소화하는 새로운 개별 모형을 선택한다.
    - **XGBoost 모형은 로지스틱 회귀모형과 쌍별랭킹을 포함한 손실함수를 최적화할 수 있다.**
- 정규화 가중치 정의 : 복잡도에 대한 가중치
    - $\omega (f) = \gamma T + \dfrac{1}{2} \lambda \sum_{j=1}^{T} w_j^{2}$
    - $\omega$ : 노드(잎)에 대한 점수 벡터 : 가중치
    - T : 노드(잎)의 갯수
    - 일반적으로 나무 모형은 불순도(엔트로피) 개선에 중점을 두고 복잡성 제어는 휴리스틱 방법(사용자의 방법)에 맡긴다.
    - **XGBoost 모델은 이 복잡성 제어를 공식적으로 사용하여 더 나은 모델을 얻을 수 있는 방법이다.**
- 목적함수에 정규화 가중치 식을 추가하여 정리하면 정보획득량을 구할 수 있다.
    - gain = 1/2 [왼쪽 노드 점수 + 오른쪽 노드 점수 - 원래 노드 점수] + 정규화가중치

### 5-2. 파라미터 튜닝
- `편향-분산 법칙`
   - XGBoost의 파라미터는 편향-분산 트레이드 오프와 관련되어 있다.
   - 모델이 복잡해질 수록(max depth가 더 깊어질 수록) 훈련 데이터에 적합한 모형이 되어 편향이 줄어 든다.
   - 그러나 복잡해진 모델을 맞추기 위해서는 데이터가 더 필요하게 된다.
- `과적합 통제 방법`
   - 훈련 데이터에 대한 성능은 높지만 검증 데이터에 대한 성능이 낮을 경우 2가지 방법 사용할 수 있다.
   - 모델 복잡성을 직접 제어하는 방법
       - max_depth, min_child_weight, gamma
   - 훈련데이터에 임의성을 추가하여 노이즈에 강화시키는 방법 
       - subsample, colsample_bytree
       - eta(스텝사이즈), num_round
- `훈련 속도 증가`
    - tree_method, hist, gpu_hist
- `불균형 데이터 처리`
    - 광고 로그 데이터와 같이 데이터가 불균형인 경우
    - AUC 값과 관련하여 양수가중치와 음수가중치 설정
        - scale_pos_weight
    - 정확한 예측 확률과 관련하여
        - max_delta_step

### 5-3. 여러가지 XGBoost 모형
- 각 클래스마다 파라미터와 속성값이 거의 같다.
- 패키지 임포트
    - import xgboost
- 분류 모형 클래스
    - xgb = xgboost.XGBClassifier()
- Booster 모형 클래스
    - xgb_booster = xgboost.Booster()
- 교차검증 클래스
    - xgb_cv = xgboost.cv()
- XGBoost 용 scikit learn의 래퍼 함수 클래스
    - xgboost.XGBRegressor()
- XGBoost 랜덤 포레스트 회귀 나무 
    - xgboost.XGBRFRegressor()
- XGBoost 랜덤 포레스트 분류 모형
    - xgboost.XGBRFClssifier()
    
### 5-4. 파라미터
- n_estimators : 약 분류기의 갯수
- max_depth : 나무의 깊이
- max_leaves : 노드의 최대갯수
- grow_policy : 나무의 크기 정책 
    - 0 : 노드에서 가까운 노드로 분할
    - 1 : 손실의 변화가 가장 큰 노드에서 분할함
- learning_rate : 학습률
- booster : 부스터 방법 : gbtree, gblinear, dart 
- n_jobs : 병렬 스레드 수, 그리드 서치를 사용할 때 스레드를 병렬화하는 방식
- min_child_weight : 하위 노드에 필요한 가중치의 최소합
- early_stopping_rounds : 훈련을 조기에 중지하도록 설정

### 5-5. 속성값
- best_iteration : 훈련을 조기중지한 반복값
- best_score : 훈련을 조기중지한 최고의 성능값
- coeff_ : 계수값
- feature_importances_ : 독립변수 중요도

## 6. LightGBM 라이브러리
- LightGBM (Light Gradient Bosst Machine) : XGBoost 보다 빠른 속도를 갖는다.

### python
- import lightgbm
- lgbm = lightgbm.LGBMClassifier()
























