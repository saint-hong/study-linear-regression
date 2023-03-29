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
    
## 5. XGBoost, LightGBM 라이브러리

### 5-1. XGBoost
- XGBoost (eXtra Gradient Boost) : GBM에서 pc의 파워를 효율적으로 사용하기 위한 다양한 기법에 채택되어 빠른 속도와 효율을 가진다.
    - import xgboost
    - xgb = xgboost.XGBClassifier()

### 5-2. LightGBM
- LightGBM (Light Gradient Bosst Machine) : XGBoost 보다 빠른 속도를 갖는다.
    - import lightgbm
    - lgbm = lightgbm.LGBMClassifier()


