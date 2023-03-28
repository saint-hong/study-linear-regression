# 부스팅 방법
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

## 1. 에이다부스트
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

### 1-1. 데이터 샘플에 대한 가중치 w
- 데이터에 대한 가중치 w_i는 최초에는 모든 데이터에 같은 균일한 값이지만, 위원회가 증가할때마다 값이 변한다.        
    - $w_{m, i} = w_{m-1, i} exp(-y_i C_{m-1}) = 
    \left{\begin{matrix}
\;w_{m-1,i} e^{-1} \;\; if \; C_{m-1} = y_{i} \\
\;w_{m-1,i} e \;\;\; if \; C_{m-1} \neq y_{i}
\end{matrix}right.$
    - 예측이 맞은 경우 가중치 값은 작아지고, 예측이 틀린 경우 가중치 값은 크게 확대(boosting) 된다.
    - m번째 개별 모형의 모든 후보에 대해 이 손실함수 L을 적용하여 값이 가장 작은 후보 를 선택한다.

### 1-2. 에이다부스팅의 손실함수
- 손실함수 L_m    
    - $L_m = \sum_{i=1}^{N} exp(-y_i C_m (x_i))$
    - $C_m(x_i) = \sum_{j=1}^{m} \alpha_j k_j (x_i) = C_{m-1}(x_i) + \alpha_m k_m(x_i)$
        - 개별모형의 결과와 가중치의 선형조합형태
    - C_m 식을 손실함수 L에 대입하여 정리
    - $\begin{aligned} L_m
&= \sum_{i=1}^{N} exp(-y_i C_m (x_i)) \\ 
&= \sum_{i=1}^{N} exp(-y_i C_{m-1}(x_i) - \alpha_m y_i k_m (x_i)) \\
&= \sum_{i=1}^{N} exp(-y_i C_{m-1}(x_i)) exp(- \alpha_m y_i k_m (x_i))\\
&= \sum_{i=1}^{N} w_{m,i} exp(- \alpha_m y_i k_m (x_i))
\end{aligned}$
    - 종속변수 y_i와 개별모형 k_m(x_i)의 값은 1 또는 -1 만 가질 수 있으므로,
    - $\begin{aligned} L_m
&= e^{-\alpha_m} \sum_{k_m(x_i) = y_i} w_{m,i} + e^{\alpha_m} \sum_{k_m(x_i) \neq y_i} w_{m,i}\\
&= (e^{\alpha_m} - e^{-\alpha_m}) \sum_{i=1}^{N} w_{m,i} I(k_m(x_i) \neq y_i) + e^{-\alpha_m} \sum_{i=1}^{N} w_{m,i}
\end{aligned}$
    - **L_m을 최소화하는 alpha_m**
    - $\dfrac{d L_m}{d \alpha_m} = 0$
    
### 1-3. 에이다 부스트 모형의 정규화
- 모형의 과최적화를 방지하기 위해 학습 속도(learning rate)를 조정하여 정규화를 한다.
    - $C_m = C_{m-1} + \mu \alpha_m k_m$
    - $\mu$ : learning rate
    - 필요한 멤버의 수를 강제로 증가시켜서 과최적화를 막는 역할을 한다.
- learning rate 값이 1보다 작으면 새로운 멤버(k_m)의 가중치 $\alpha_m$를 강제로 낮춘다.
- learning rate 값이 1보다 크면 성능은 크게 저하된다.

## 2. 그래디언트 부스트
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

### 2-1. 그래디언트 부스트 모형의 계산 방식
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
