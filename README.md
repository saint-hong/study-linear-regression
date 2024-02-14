# study-machine-learning
Description and Summary of the Machine Learning

## 1. Linear Regression
- 선형회귀분석 모형
    - 상수항 결합
    - 최소자승법(OLS)
    - 직교방정식
- 스케일링
    - 조건수(condition number)
    - 다중공선성(multicollinearity)
- 범주형 독립변수
    - 풀랭크 방식(full-rank)
    - 축소랭크 방식(reduced-rank)
    - 상호작용(interaction)
- 부분회귀
    - 새로운 독립변수의 추가
    - 프리슈-워-로벨 정리(FWL)
    - CCPR
- 확률론적 선형 회귀모형
    - 부트스트래핑
    - 확률론적 선형 회귀모형의 4가지 가정
    - 최대가능도법(MLE)과 선형 회귀분석
    - 회귀계수의 표준오차(std err)
    - 단일계수 t-검정(single coefficient t-test)
    - 회귀분석 F-검정(regression F-test)
- 회귀분석의 기하학
    - 회귀 벡터공간
    - 잔차행렬(residual matrix)과 투영행렬(projection matrix)
    - 잔차행렬과 투영행렬의 성질
    - 종속변수와 예측값, 잔차의 관계
- 레버리지와 아웃라이어
    - 레버리지의 특성
    - 레버리지의 영향
    - 아웃라이어
    - 표준화 잔차
    - 쿡스디스턴스(cook's distance)
    - 폭스아웃라이어 추천값(fox outlier recommendation)
- 분산분석과 모형 성능
    - 분산분석(ANOVA)
    - TSS, ESS, RSS
    - 분산관계식과 의미
    - 결정계수(coefficient od determination)
    - 분산분석표
    - 회귀분석 F-검정과 분산분석의 관계
    - F-검정을 사용한 모형비교, 변수중요도 비교
    - 조정결정계수(adjustments R-squared)
    - 정보량규준(information criterion), AIC, BIC
- 모형진단과 수정
    - 정규성 검정
    - 잔차와 독립변수의 비상관관계
    - 이분산성(heteroskedastic)
    - 오차의 독립성 검정
    - 비선형 변형
- 기저함수 모형과 과최적화
    - 비선형 모형
    - 기저함수, 다항회귀
    - 과최적화(overfitting)
- 교차검증
    - 과최적화(overfitting)
    - 학습/검증 데이터 분리
    - k-폴드 교차검증
    - 평가점수
- 다중공선성과 변수 선택
    - 다중공선성(multicolinearity)
    - 독립변수의 풀랭크 조건
    - 과최적화 방지 방법
    - 변수선택법(VIF)
- 정규화 선형회귀
    - 정규화 선형회귀방법
    - Ridge 모형
    - Lasso 모형
    - Elastic Net 모형
    - 최적 정규화(optimal regularization)
    - 검증성능 곡선(validation curve)
    - 다항회귀의 차수 결정

## 2. Classification
- 분류 기초
    - 분류용 예제 데이터
    - 분류용 가상 데이터 생성
    - 분류모형들
    - 분류성능평가
- 확률적 모형(probabillistic model)
    - 생성 모형(generative model)
        - LDA/QDA
        - 나이브베이즈(naivebayes)
    - 판별 모형(discriminant model)
        - 로지스틱 회귀(logistic regression)
        - 의사결정 나무(decision tree)
- 판별함수 모형(discriminant function model)
    - 퍼셉트론(perceptron)
    - 서포트벡터머신(SVM)
- 모형결합(ensemble)
    - 취합 방법
    - majority voting
    - bagging
    - 랜덤 포레스트(random forest)
    - 부스팅(boosting)
        - adaboost
        - gradient boost
        - XGboost
        - LightGBM
- 모형최적화
    - hyperparameter tunning
    - grid search
    - pipeline

## 3. Clustering
- 군집화
- kmeans
- 디비스캔 군집화
- 계층적 군집화
- affinity propagation

## 4. Recommendation
- surprise
    - 베이스라인 모형
    - collaborative filtering
    - neighborhood models
        - user based CF
        - item based CF
    - latent factor models
        - matrix factorization
        - SVD
- 추천성능 평가기준
- 유사도(simirarity)

## 5. Time Series
- fb prophet

## 6. 머신러닝 알고리즘 정리

### 전처리
- 간략한 설명, 여러가지 종류 위주로 정리, pca 는 자세하게
- Scaler : 수치형 데이터의 차이를 맞춰주는 작업
    - MinMaxScaler, StandardScaler, LogScaler 등
- PCA : principle component analysis : 차원축소 (중요)
- mpimg : 이미지 처리 도구, 이미지 분석 때 필요함
- Regular Expression : 정규표현식 : 문자열 데이터를 분석하는데 필요한 도구
- Label Encoder : 라벨인코더 : 문자열 데이터를 수치형 데이터로 변환해주는 도구
- SMOTE oversampling : 데이터의 불균형을 맞추기 위한 도구, 데이터를 생성해준다.
- tgdm : 반복문의 실행 과정을 시각화해서 보여주는 도구

### 모델
- 이론 + 옵션 + 사용법
- Decision Tree : 의사결정 나무
- Cost Function : 선형회귀모델의 분류 방법
- Logistic Regression : 로지스틱 함수를 사용한 회귀모델의 분류 방법
    - cost function 과 연관되어 있음
    - Decision Boundary
    - 다변수 방정식
- Ensemble : 여러개의 분류기를 생성하고 분류기들의 예측값을 결합하여 더 정확한 최종 예측을 도출하는 기법
    - 단일 분류기보다 신뢰성이 높은 예측값을 얻고자 함
    - voting
    - bagging - bootstrapping
    - 하드보팅, 소프트보팅
    - Random Forest Classifier
- Boosting : 앙상블 기법 중 부스팅 방식을 사용한 모델들
    - Adaboost
    - Gradientboosting : GBM
    - XGBoost
    - LightGBM
- kNN : k Nearest Neighbor : 가까운 이웃 데이터 분류 모델, k 는 가까운 거리
- FBProphet : 페이스북에서 제작한 시계열 데이터 분석 모델
- Natural language processing : 자연어 처리
    - nltk : natural language toolkit
    - KoNLPy : Korean natural language processing in Python : 한국어 정보처리 파이썬 패키지
    - wordcloud
- MultiNomial Naive Bayes Classifier : 나이브 베이즈 분류
    - 베이즈 정리르 적용한 확률 분류기
- Suppoert Vector Machine : SVM, 서포트 벡터 머신

### 모델 성능 향상 도구
- Hyperparameter Tunning : 모델 성능 향상을 위한 설정값 조절
    - GridSearchCV : 교차검증의 종류, 모델의 설정값을 일괄적으로 조절하는 도구
- PipeLine : 모델을 만드는 과정의 여러 단계들의 순서를 연결해주는 도구

### 모델평가
- Cross validation : 교차검증
    - 훈련용 데이터를 5개로 나누고 1개의 테스트데이터와 4개의 훈련데이터로 세분화하여 모델의 성능의 정확도를 높이는 검증과정
    - k-Fold cross validation
    - stratified cross validation
- Model Verification : 모델 평가
    - 정확도, 오차행렬, 정밀도, 재현율, FPR-TPR, F1score, AUC score, ROC curve
    - threshold 값에 따른 평가지표의 변화
    - classification report, confusion matrix
