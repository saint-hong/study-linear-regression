## 모델링 프로세스

### feature engineering과 hyper parameter tunning
- 데이터 원본을 기준으로 간단한 전처리와 feature engineering을 적용하여 우선 성능테스트를 한다.
   - 훈련데이터 원본이 크게 바뀌지 않는 한 모델 성능은 어느정도 제한된 범위에 있다.
- 점차 전처리와 feature engineering의 여러가지 방법들을 조합하여 성능 테스트를 한다.
   - 일반적인 feature engineering 방법외에도 카테고리 변수를 어떤 것으로 선택하느냐에 따라서도 성능이 달라질 수 있다.
- 성능 변화를 모니터링한 후 하이퍼 파라미터 튜닝을 통해 모델을 고도화한다.
   - 모형 최적화 자체가 매우 많은 시간을 필요로 하기 때문에, 앞에서 전처리와 엔지니어링을 통해 데이터의 품질을 높여야 한다.
   - 모수 추정 원리에 따른 대표적인 모델들을 우선 테스트하고 높은 성능의 모델이 나오면 이것과 같은 계열의 다른 모델들을 추가로 테스트 해보는 것도 좋다.


### 1. 데이터 분석
- 선형회귀, 분류, 시계열 인지에 대한 판단
- 독립변수와 종속변수의 특징 분석 : 종류, 타입, 분포, 상관관계 등
   - 데이터가 불균형인지? ---> 불균형이면 비대칭 데이터 처리
   - 차원이 높은지? ---> 차원축소 기능을 써야하는지?
   - ** 수치형, 범주형 독립변수 분석 : 모델링에서 가장 중요한 부분이기도 하다. feature engineering을 하다보면 feature의 dtype이 바뀔 수도 있다. **
- 모델링을 통해서 얻고자 하는 결과에 대한 기대를 충분히 분석한다.

### 2. 데이터 전처리
- 분석모델의 인풋 데이터로 사용할 수 있도록 데이터 전처리
   - dtype 변환
   - 수치형, 범주형 변수의 유니크값 중 1개인 것의 처리
- 분석모델의 기능에 맞춰서 데이터 전처리
- 독립변수의 특징에 맞는 전처리기 사용
- 이미지, 자연어, 수치형 데이터에 맞는 전처리
- feature engineering
   - 결측치 처리 : 통계값, MICE 사용
   - 스케일링
   - 더미변수화 : 범주형 데이터의 숫자형 데이터로 변환
   - 바이닝 : 연속하는 숫자형 데이터에서 특정 구간을 나눔
   - 그룹핑 : 고윳값이 많은 경우 특정 그룹단위로 구분
   - 로그변환 : 데이터가 왜곡 된 경우 균형을 맞춰줌
   - 텍스트 추출 : 텍스트 데이터에서 특정한 데이터를 추출
   - 날짜 추출 : 날짜 데이터에서 날짜 유형 추출
   - 새로운 변수 생성

### 3. 모델 선택
- 모형별 원리와 특징, 성능에 기반하여 모형 취합
- 데이터 분석에서 나온 인사이트를 기반으로 모형 선택
- 전처리기, 파이프라인, CV에 대한 계획

### 4. 모델링
- ML/DL 모형
   - 확률적 생성모형 : LDA/QDA, Naive Bayes(GaussianNB, BernoulliNB, MultinomialNB, complementNB, categoricalNB)
   - 확률적 판별모형 : LogisticRegression, DecisionTree
   - 판별함수 모형 : Perceptron(MLP(neural network models)), SVC, DeepLearnning
   - SGDClassifier, GaussianProcessClassifier
- Ensemble
   - 취합 : VotingClassifier, BaggingClassifier, RandomForestClassifier
   - 부스팅 : adaboost, gradientboost, XGBoost, LightGBM
- Preprocessing
   - ColumnTransformer() : 컬럼별 다른 전처리를 할 수 있는 기능
   - make_column_selector() : CoulumTransformer 내부에서 feature의 dtype을 구분하는 기능으로 사용할 수 있음
   - numeric : StandardScaler(), MinMaxScaler
   - categorical : OneHotEncoder(), OrdinalEncoder()
      - 범주형 독립변수를 어떻게 처리하느냐에 따라서 모델의 성능이 달라 질 수 있다.
      - OneHotEncoding 방식은 무의미한 feature를 많이 만들 수 있다.
- Pipeline
   - 파이프라인의 구조를 잘 만들면, 하이퍼 파라미터 튜닝을 할때 파이프라인의 인수들을 사용하여 여러 가지 조건들을 쉽게 변경할 수 있다.
      - 모델 변경, 전처리기 변경, feature engineering 방법 변경 등
- KFold
   - KFold, RepeatedStratifiedKFold, 중첩 CV
   - kf.split() method : 구분된 Fold 별 train, test 데이터 인덱스 값을 사용할 수 있다.
- CV
   - validation curve
   - GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
- 이외에 다양한 모델들 적용

### 5. 모델 평가
- 분석 타입별 평가지표
   - acc, recall, precision, f1, AUC, Classification Report
   - 로그손실값 : 교차엔트로피값
   - 1-acc : 모델의 오분류값
- PDP, ICE : 독립변수와 종속변수의 의존성
- OOB estimator : SGD 계열의 모형 성능 평가에 사용 (RF 계열 ensemble 모형)
   - 트리가 분화할 때 사용하는 훈련 데이터 샘플의 조합을 바꾸는 방식
   - RF 모형인 경우 : 변수 중요도, 최적의 트리수, 변수선택 등의 이점이 있음
- Statistic evaluation : 모형 성능값을 통계적 방법으로 비교
   - 빈도주의적 접근
   - 베이지안 추정
- 상위 성능의 표준편차의 범위안에서 균형있는 모델 평가(예컨데 PCA의 차원축소 갯수)

### 6. 모형 최적화

#### 교차검증
- KFold
   - 교차검증에 사용할 분할 데이터를 만든다.
   - shuffle params: 데이터를 분할하기 전에 데이터를 섞을지 말지 결정
   - kf.split() method : 구분된 Fold 별 train, test 데이터 인덱스 값을 사용할 수 있다.
- RepeatedStratifiedKFold
   - n_split, n_repeats로 n x n 번의 cv를 실행한다.
   - shuffle은 사용할 수 없다.   
- cross_val_score
   -
- GridSearchCV
   - 파이프 라인을 사용하여 다양한 parameter 조합으로 교차검증을 수행할 수 있다.
   - 이로부터 모델 fitting 결과에 대한 다양한 속성값을 사용할 수 있다.
- RandomizedSearchCV
   - parameter의 샘플을 특정한 분포(distribution)에서 샘플링한다.
   - param_grid가 아니라 param_distributions에 파라미터 설정 dict를 넣는다.
   - {"alpha": stats.expon(scale=1), "gamma": stats.loguniform(1e-1, le3)}
   - GSCV보다 속도가 빠르지만 성능은 낮다.
- HalvingCV
   - 실험 회차별 데이터 샘플의 수가 증가하고, 상위성능 절반이 후보(parameter 조합)가 되어 점점 줄어드는 방식
   - HalvingGridSearchCV
   - HalvingRandomSearchCV   

#### 특징선택
- 독립변수가 많은 경우 종속변수와의 상관관계나 성능이 높은 독립변수를 선택하여 성능을 높일 수 있다.
   - 변수 선택으로 반드시 성능이 높아지는 것은 아니다.
   - 성능이 높아진다고 하더라도, 다른 지표를 활용하여 사용해야한다.
- SVM-ANOVA (단일 변수 선택 방법의 하나)
   - 분류성능을 향상시키기 위해서 feature select를 할 수 있다.
   - from sklearn.feature_selection import SelectPercentile, f_classif
   - percentile 값으로 (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100) 을 설정하고
   - cross_val_score로 fitting 하면서 높은 score를 찾을 수 있다.
   - iris data에 36개의 feature를 추가한 후 SelectPercentile(precentile=) 을 사용하면 40개의 feature 중에서 percentile 에 해당하는 갯수만큼 feature를 선택한 후 cv를 수행한다.
   - pipeline에 넣어서 사용가능
   - SelectPercentile(f_classif, percentile) 파라미터를 사용한다.
   - https://scikit-learn.org/stable/auto_examples/svm/plot_svm_anova.html#sphx-glr-auto-examples-svm-plot-svm-anova-py
- 분산 선택법
   - VarianceThreshold() : feature의 분산값을 계산하여 thr을 기준으로 제거
- 단일 변수 선택
   - SelectKBest() : 모델의 성능 통계 함수를 사용하여 상관관계나 성능이 높은 독립변수를 선택
   - SelectPercentile() : 설정한 최고 점수 비율을 제외한 모든 feature를 제거한다.
- 모델 기반 선택
   - SelectFromModel() : coef_, feature_importances_, l1 penalty의 가중치 값을 기준으로 선택
      - DT, RF, LR, LinearSVC, Lasso 등 위의 값을 사용할 수 있는 모형을 사용할 수 있다.
- 순차적 선택
   - SequentialFeatureSelector() : forward, backward 방향에 따라서 feature 순차적으로 늘리거나 줄이면서 cv score가 높은 것을 선택
- pipeline에 넣어서 모델링 할 수 있다.
   - make_pipeline(StandardScaler(), SequentialFeatureSelector(), SVC())


### 7. 최종 모델 선택과 해석
- 최종 모델을 선택한 후 성능에 대한 해석이 가능해야 한다. 따라서 전체 모델링 과정에서 시도한 방법들과 일어난 사건들의 특징점들을 모델의 해석을 위해 잘 정리하는 것이 좋다.

## 모델링 팁

### 1. 분류 문제의 클래스 전처리
- 범주형 클래스들을 해당여부에 따라서 1과 0으로 변환 : label_binarize
   - roc_curve()를 사용할 떄 종속변수의 값을 binarize 변환하여 사용해야 한다.
   - from sklearn.preprocessing import label_binarize
   - label_binarize(y, classes=(0, 1, 2, 3))
   - fpr, tpr, thr = roc_curve(bin_y[:, i], clf.predict_proba(X_test)[:, 1])
   - [[0, 1, 2]] => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
- thr을 기준으로 큰값과 작은값을 이진값으로 변환 : Binarizer
   - from sklearn.perprocessing import Binarizer
   - Binarizer(threshold=7).fit_transform(X)
   - [0, 8, 9, 3, 6] => [0, 1, 1, 0, 0]

### 2. 로지스틱회귀모델 (Logistic Regression)
- 소규모 데이터는 libliner, 대규모 데이터는 saga solver가 성능이 좋은 편
- multi_class 인수는 multinomial이 ovr 보다 성능이 좋은 편
   - solver="liblinear"는 ovr 설정
- max_iter는 solver의 최적화 방법이 수렴하는데 걸리는 반복횟수
   - 이 설정에 따라서 성능이 변할 수 있다.
- solver 별로 penalty 설정값이 다르다.
   - penalty는 모형을 정규화 시키는 설정값
- C 는 penalty 설정값의 범위 : l1_min_c 패키지를 사용하여 C의 범위를 만들 수 있다.
   - from sklearn.svm import l1_min_c
   - l1_min_c(X, y, loss="log) * np.logspace(0, 10, 16)
- 모형의 속성값으로 coef_ 사용가능
   - C에 따른 coef_의 값 변화

### 3. 선형판별분석모형, 이차판별분석모형 (LDA/QDA)
- 조건부확률의 계산을 위해 베이즈 정리를 사용하며, 가능도를 입력변수 x가 실수인 다변수정규분포라고 가정한다.
   - QDA는 이 다변수 정규분포의 공분산이 모든 k에 대해 다르다고 가정
   - LDA는 이 다변수 정규분포의 공분산이 모든 k에 대해 같다고 가정
- 다중 클래스 분류 문제에서 잘 적용 된다고 알려짐
- 하이퍼 파라미터 조정이 단순한 편이다.
- LDA는 shrinkage 파라미터를 사용하여 공분산행렬을 설정할 수 있다.
   - shirinkage는 공분산 행렬을 추정하는데 사용되는 입력인수(parameter)
      - 데이터의 수(sample)가 독립변수(feature)의 수보다 작을 때 공분산 행렬을 추정하기 위함
      - 'auto', 0, 1, 0<= x <= 1 값을 사용할 수 있다.
   - sklearn.covarinace.OAS 추정기 만들고 이 값을 사용할 수도 있다.
      - OAS 추정기를 사용한 경우 공분산행렬을 더 잘 추정할 수 있다.
   - solver가 lsqr, eigen일때 사용가능
   - 독립변수가 데이터 샘플보다 더 많을 때 추정 공분산 행렬을 개선해준다.
- 공분산 행렬을 찾는 방식으로 covariance_estimator 인수를 설정할 수 있다.
   - from sklearn.covariance import OAS
   - oas = OAS(store_precision=False, assume_centered=False)
   - LinearDiscriminantAnalysis(solver="lsqr", covariance_estimato=oas).fit(X, y)
- eigen solver는 공분산 행렬을 계산해야하므로 독립변수가 많은 경우 시간이 오래 걸릴 수 있다.
- lsqr solver는 분류에만 작동하는 알고리즘이다.
- svd solver는 (특잇값분해) 공분산행렬의 계산에 의존하지 않으므로 독립변수가 많은 경우 적합할 수 있다.

### 4. 나이브베이즈 (Naive Bayes)
- 종류
   - GaussianNB : 가우시안 NB : 정수형 데이터
   - BernoulliNB : 베르누이 NB : 이진형 데이터
   - MultinomialNB : 다항분포 NB : 실수형 데이터
   - complementNB : 보완 NB : MNB의 불균형 데이터 세트 보완
      - 텍스트 분류에서 MNB보다 성능이 더 좋은 경향
   - categoricalNB : 카테고리 NB : 범주형 데이터
- 데이터 세트의 타입에 따라서 다른 NB 모형을 사용하여 분석한 후, 모형의 likelihood 값을 곱하여 전체 가능도를 구할 수 있다.
   - (GNB likelihood * MNB linelihood * BNB likelihood) * prior proba
- 정수형 데이터의 경우 특정값을 기준으로 이진 데이터로 변환하여 BNB로 분석하면 성능이 더 좋을 수 있다.
   - 이미지의 경우
- 다른 분류기들보다 성능과 속도가 좋은편이다.
   - 성능 지표는 좋지만, 클래스별 확률값은 신뢰하기 어려울 수 있다.
- NB 최적화 논문
   - 실제 데이터의 feature는 서로 의존성이 있으나, NB 모형의 성능이 좋은이유는 특정 feature간의 의존성이 전체 의존성 분포에 따라서 상쇄되는 효과가 있기때문이라는 점.
- alpha 파라미터
   - 스무딩 적용방법 : 모수가 0 또는 1과 같이 극단적인 값이 나오지 않도록 0.5에 가깝게 조정하는 방식
   - 라플라스 스무딩, 애드윈 스무딩

### 5. 의사 결정 나무 (Decision Treee)
- max_depth가 클 수록 의사결정이 더욱 복잡해지고 모델이 더 적합해진다. (fitter model)
   - depth는 자식 노드의 층의 수
   - 즉 각 자식 노드마다 분할기준을 찾고, 이 분할 기준에 의해 다음 자식노드로 데이터를 분할하여 넘겨준다.
- 과최적화가 발생할 수 있다.
   - 지나치게 복잡한 형태로 모형이 만들어질 수 있다.
   - 파라미터 조절을 통해서 해결할 수 있다.
   - ensemble 모형을 사용하여 데이터 세트를 다변화하여 학습시킬 수 있다.
- 분할기준은 엔트로피 값이 낮은 방향, 질서도가 높아지는 기준을 찾는다.
- 데이터의 전처리를 하지 않아도 된다.
   - 결측값이 있어도 분류가 가능하다.
   - 데이터 균형을 맞춰야 한다. (불균형 데이터이면 편향된 트리가 만들어 진다.)
- 과최적화를 방지하기 위해 파라미터 설정을 해줘야 한다.
   - max_depth, min_samples_leaf, min_smaples_split
- 데이터에 따라 성능의 변화가 클 수 있다.
   - 데이터가 조금만 바뀌어도 성능이 변화한다.
- Tree 계열의 모델들은 비슷한 특징을 갖는다.   

### 6. 모형결합 (ensemble) 취합방법
- 모형결합 = 앙상블, 취합방법 = aggregation
- 모형결합을 하여 성능을 개선시키는 방법
   - 시간은 늘어나지만 단일 모형보다 성능 분산 감소 ---> 과최적화 방지
   - 개별 모형보다 모형결합의 성능이 더 높음
- 취합 aggregation
   - majority, bagging, RF
- 부스팅 boosting
   - adaboost, gradientboost, xgboost, lightgbm
- 다수결 방법
   - from sklearn.ensemble import VotingClassifier
   - VotingClassifier(estimators=[("lr", model_1), ("qda", model_2)], voting="soft", weights)
   - 다른 형태의 모형끼리도 결합 할 수 있다.
   - hard voting : 개별 모형의 결과 기준 다수의 결과값을 최종 반환
   - soft voting : 개별 모형의 조건부 확률의 합에 가중치를 곱한 값 중 큰 것을 반환
   - 개별모형의 갯수 N이 늘어날 수 록 성능이 커진다.
- 배깅 방법
   - from sklearn.ensemble import BaggingClassifier
   - BaggingClassifier(estimator=model_1, n_estimators=100, bootstrap="bagging", max_samples=0.8)
   - 같은 확률모형을 결합하고, 랜덤한 데이터 세트를 훈련하여 다른 결과를 출력하는 다수의 모형을 만든다.
   - train 데이터 선택 방법 : pasting, bagging, random subspaces, random patches
   - test 데이터 선택 방법 : OOB
- Random Forest
   - 의사 결정나무를 기본 모형으로 사용하는 취합 방법
   - VotingClassifier, BaggingClassifier는 여러 모형을 사용할 수 있지만, RF는 DT만 사용 가능
   - 훈련 데이터 세트의 독립변수의 갯수를 랜덤하게 줄인다.
   - DT의 과최적화를 방지하는 대안 방법
   - RandomforestClassifier, ExtraTreesClassifier
   - ExtraT는 독립변수를 선택하는 방법의 무작위성을 극단적으로 적용한 것
      - 최적의 분할기준을 찾기 위해 더 무작위성이 높은 형태
- RF 파라미터 튜닝 규칙
   - n_estimators : 클 수록 시간이 길어지고, 특정 나무 수가 넘어가면 성능이 더 좋아지지 않는다.
   - max_features : 노드 분할시 사용되는 독립변수의 갯수 : 분류문제에서는 "sqrt", 회귀에서는 1.0 값 사용 일반적
   - max_depth=None, min_samples_split=2 (트리를 완전히 개별화함)
   - bootstrap : RF는 True, EXT는 False

### 7. 모형결합 (ensemble) 부스팅 방법
- 이진 분류에 적용 가능
- Cm에서 M개의 약분류기의 조건부 확률을 가중선형조합한 값을 판별함수로 사용한다.
   - 이 판별함수를 최적화 한다.
- adaboost
   - 손실함수 L_m을 최소화하는 모형 선택
   - i번째 훈련 데이터에 w_i 가중치를 적용하고, 각 분류기에서 틀리게 예측한 w_m,i 를 합한다.
   - 개별 모형의 가중치는 틀린 데이터의 확대된 가중치의 영향을 받게 된다.
   - w_m,i 는 처음에는 모두 같지만 위원회가 늘어날 수록 달라진다.
   - 손실함수는 지수함수 등을 사용하여 틀린 데이터의 가중치는 크게, 맞은 데이터의 가중치는 축소시킨다.
   - 틀린 데이터는 다음 위원회의 데이터 세트에서 여러번 선택될 가능성이 더 높아지게 된다.
   - 이 과정에서 새로운 약분류기 후보는 손실함수를 최소화하는 모형이 선택된다.
   - 예측이 틀린문제들의 가중치가 점차 커지게 되므로 정규화가 필요하다.
   - 개별 모형에 learning_rate을 곱하여 정규화를 한다.
- gradientboost
   - 변분법을 사용한 모형으로 손실 범함수 L(y, C_m-1)을 최소화하는 개별모형 k_m을 찾는다.
   - 경사하강법 식에 이 손실 범함수를 대입하여 풀 수 있다.
   - 회귀/분류문제에 모두 회귀 모형을 사용하며 의사결정 회귀나무를 많이 사용한다.
   - GradientBoostingClassifier, HistGradientBoostingClassifier
   - 데이터 샘플수가 수만개가 넘는 경우 HGBC가 GBC보다 빠르다.
   - DTC 특징과 같이 결측 데이터가 있어도 전처리 필요없이 사용가능하다.
   - learning_rate가 낮을 수록 많은 n_estimators가 필요하다.
- HistGradientClassifier, HistGradientRegressor
   - 히스토그램 기반의 clf : 트리를 구축하는 과정에서 학습 데이터를 구분할 때 실수가 아닌 정수값을 사용
   - n_estimator 대신 max_iter 값을 사용한다.
   - categorical_features를 설정하면 범주형 독립변수의 전처리를 한다.

### 8. 서포트 벡터 머신 (SVM)
- 서포트 벡터 머신의 판별함수는 마진을 최대화하기 위한 손실함수 L을 사용하며, 샘플 x와 서포트 벡터와의 코사인 유사도중에 큰 쪽으로 결과를 반환한다.
- 최종 목적함수는 L에 슬랙변수를 추가하여 선형판별경계로 완전히 분리되지 않는 벡터들의 분류를 쉽게 할 수 있게 된다.
   - 슬랙변수 엡실론은 0보다 크거나 같은 값이다.
   - 슬랙변수의 크기를 제어하기 위한 상수 C에 의해서 서포트 벡터의 갯수가 달라지게 된다.
   - C의 값이 작아질 수록 서포트벡터의 갯수가 많아진다. 즉 분류가 용이해 진다.
   - 노이즈가 많은 데이터에서는 C의 값을 줄인다.
- kernel SVM
   - 선형판별경계로 분류되지 않는 XOR 문제 등을 풀기 위한 방법으로 기저함수 벡터를 사용하여 다양한 형태의 커널함수를 적용할 수 있다.
   - kernel={"linear", "poly", "rbf", "sigmoid", my_custom()}
   - 각각의 커널들은 다양한 기저함수들로 분리될 수 있다.
- 분류, 회귀, 이상치 탐지등에 사용 된다.
- 고차원 데이터 샘플에서 효과적이다.
   - features > samples
- 다중 클래스 문제
   - decision_function_shape={"ovo", "ovr"}
- 매서드 predict_proba() 보다 decision_function() 값이 더 나을 수 있다.
- 클래스의 값을 [0, 1] 또는 [1, -1] 로 scaling 하여 사용한다.
- SVM의 중요한 파라미터
   - gamma : 개별 데이터 하나가 훈련에 미치는 영향을 조절, 클 수록 다른 데이터 샘플들에 영향을 받는다.
   - C : 작으면 의사 결정 표면을 매끄럽게 만든다, 크면 모든 데이터를 정확하게 분류하려고 한다.
      - 슬랙변수 조정값 : 작을 수록 서포트 벡터의 갯수가 늘어난다. 선형경계면에 가까운 데이터들이 추가된다.
   - gamma와 C는 반비례관계로 두 파라미터의 크기에 따라서 성능이 달라 질 수 있다.
   - kernel : {"linear", "poly", "rbf, "sigmoid", my_custom} 
      - rbf kernel이 성능이 좋은 것 같다.
      - kernel 함수를 직접 만들고 함수 형태로 설정할 수 있다.
   - class_weight : 클래스의 불균형이 있는 경우 특정 클래스에 대한 가중치를 설정할 수 있다. 
      - class_weight={1: 100} : 클래스 1에 가중치를 100을 적용한다.
   - clf.fit(sample_weight=sample_weight) :  fit 매서드의 인수로 sample_weight 값을 설정할 수 있다. 
      - 개별 데이터 하나하나에 대한 가중치를 적용할 수 있다. 
      - class_weight, sample_weight 설정을 통해 판별경계면의 오류를 일부 바로 잡을 수 있다. 
   - decision_function_shape : multi-class 데이터인 경우 "ovo", "ovr" 설정을 할 수 있다. 
- feature select
   - feature의 조합을 분위수로 나누어 테스트 할 수 있다. 

### 빅데이터 모형의 누적 fitting
- 빅데이터를 분할하고 분할한 데이터를 학습한 모형을 누적하여 fitting 할 수 있다.
   - **사전 확률 분포를 설정 할 수 있는 생성 모형**
   - **시작 가중치를 설정할 수 있는 모형**
- 반복문에서 학습 데이터를 구분하고 훈련한 모형을 누적하여 fitting 한다.

### feature engineering 
- 차원축소 : PCA, t-nse
- 피쳐 엔지니어링 방법들 :
   - null data fill
   - log transform
   - binning
   - outlier handling
   - scaling
   - groupping
   - categorical data embading
   - feature interaction
   - feature select
   - create new feature
