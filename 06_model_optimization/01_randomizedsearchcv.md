# RandomSearchCV를 사용한 하이퍼파라미터 튜닝 및 시각화
- 파이프라인을 사용한 텍스트 분류의 과정에서 하이퍼파라미터 튜닝을 시각화 할 수 있다.
- RandomizedSearchCV : vect와 clf 분류기의 하이퍼파라미터를 조정할 수 있다.
    - 하이퍼파라미터의 조합을 제어할 수 있다.
    
### 과정
- 데이터 임포트
- 파이프라인 생성
    - **pipeline = Pipeline()**
- 하이퍼파리미터 그리드 생성
    - **parameter_grid = {}**
    - random search cv에서 사용
- RandomizedSearchCV 설정
    - **random_search = RandmoizedSearchCV()**
    - estimator=pipeline : 파이프라인 연결
    - param_distributions=parameter_grid : 하이퍼파라미터 그리드
    - n_iter : 조합의 반복수
    - n_jobs : CPU 병렬 연결 수, 계산 시간과 관련있음
    - random_state : 난수 발생값
    - verbose : 피팅 설명 출력
- 하이퍼파라미터 튜닝
    - **random_search.fit(data_train.data, data_train.target)**
- 튜닝 결과 확인
    - **random_search.best_estimator_.get_params()** : 최적의 parameter 값 반환
    - vect와 clf의 여러가지 파라미터 값이 출력된다.
    - 하이퍼파라미터 그리드에서 설정하지 않은 파라미터는 디폴트 값
- 성능 측정
    - **accuracy = random_search.score(data_test.data, data_test.target)**
    - 검증 데이터를 사용하여 정확도 측정
    - 훈련 데이터를 사용한 정확도는 **random_search.best_score_** 에 저장되어 있음
- 성능 결과 데이터 프레임으로 변환
    - 모든 성능 측정 결과는 **random_search.cv_results_** 에 저장되어 있음
    - **pd.DataFrame(random_search.cv_results_)**
    - 데이터 프레임의 컬럼은 파라미터 이름으로 되어 있으므로, 긴 컬럼은 정리를 하여 가독성을 높여준다.
    - **param_name.rsplit("__", 1)[1]**
- 측정시간-성능값 그래프
    - plotly를 사용하여 오차막대 그래프로 시각화
    - x축 : 측정시간, y축 : 성능값
    - 측정시간과 성능값으로 이루어진 클러스터를 찾을 수 있다.
    - **px.scatter()**
- 하이퍼파라미터 조합의 병렬좌표플롯
    - plotly를 사용하여 하이퍼파라미터간의 조합에 따른 성능 변화를 시각화
    - 어떤 파라미터가 성능에 영향을 주는지 파악할 수 있다.
    - **px.parallel_coordinates()**


### 1. 데이터 임포트
- 20news groups 데이터 사용
    - categoreis 인수 설정하여 원하는 특징 벡터만 선택가능
    - test set, train set 을 각각 다른 변수에 저장

```python
from sklearn.datasets import fetch_20newsgroups

categories = ["alt.atheism", "talk.religion.misc"]

data_train = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True,
    random_state=42, remove=("headers", "footers", "quotes"))

data_test = fetch_20newsgroups(
    subset="test", categories=categories, shuffle=True,
    random_state=42, remove=("header", "footer", "quotes"))

print(f"Loading 20 newsgroups datasets for {len(data_train.target_names)} categories :")
print(data_train.target_names)
print(f"{len(data_train.data)} documents")

>>> print

Loading 20 newsgroups datasets for 2 categories :
['alt.atheism', 'talk.religion.misc']
857 documents
```

### 2. 하이퍼파라미터 튜닝을 위한 파이프라인 생성
- TfidfVectorizer : text feature vectorizer : 텍스트 특징 벡터화기
- ComplementNB : 분류기
- Pipeline : 파이프라인

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer()),
        ("clf", ComplementNB()),
    ]
)
pipeline

>>> print

Pipeline(steps=[('vect', TfidfVectorizer()), ('clf', ComplementNB())])
```

### 3. 하이퍼파라미터의 그리드 설정
- RandomizedSearchCV의 param_distributions 인수에 사용할 설정값
- 하이퍼 파라미터 튜닝 특징
    - RandomizedSearchCV : 무작위 조합의 수를 제어한다. 
    - GridSearchCV : 계산 비용이 많이 든다. 모든 조합을 탐색한다.
- n_iter : 무작위 조합 수 제어 인수
    - 그리드에서 가능한 조합의 수보다 클 경우 이미 탐색한 조합을 반복한다.
- pipeline의 vect와 clf의 최상의 매개변수 조합을 조회한다.    

```python
parameter_grid = {
    "vect__max_df" : (0.2, 0.4, 0.6, 0.8, 1.0),
    "vect__min_df" : (1, 3, 5, 10),
    "vect__ngram_range" : ((1, 1), (1, 2)), #unigram or bigram
    "vect__norm" : ("l1", "l2"),
    "clf__alpha" : np.logspace(-6, 6, 13)
}

parameter_grid

>>> print

{'vect__max_df': (0.2, 0.4, 0.6, 0.8, 1.0),
 'vect__min_df': (1, 3, 5, 10),
 'vect__ngram_range': ((1, 1), (1, 2)),
 'vect__norm': ('l1', 'l2'),
 'clf__alpha': array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06])}
```


### 4. RandomizedSearchCV 설정
- n_iter : 설정에 따라서 조합의 제어가 달라진다.
    - 불필요한 조합은 제거한다는 의미인 것 같다.
- n_jobs : 계산량 증가를 낮추기위해 CPU를 병렬화하여 사용하도록 설정 할 수 있다.    

```python
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=pipeline, # vect : tfidf, clf : complementNB
    param_distributions=parameter_grid,
    n_iter=40,
    random_state=0,
    n_jobs=2,
    verbose=1
)

random_search

print("Perfoming grid search...")
print("Hyperparameters to be evaluated : ")
pprint(parameter_grid)

>>> print

Perfoming grid search...
Hyperparameters to be evaluated :
{'clf__alpha': array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]),
 'vect__max_df': (0.2, 0.4, 0.6, 0.8, 1.0),
 'vect__min_df': (1, 3, 5, 10),
 'vect__ngram_range': ((1, 1), (1, 2)),
 'vect__norm': ('l1', 'l2')}
```

### 5. 하이퍼 파라미터 튜닝
- RandomizedSearchCV
    - estimator=pipeline : vect와 clf 모형 객체 연결
    - param_distributions=parameter_grid : 모형의 파라미터 값 탐색

```python
from time import time

t0 = time()
random_search.fit(data_train.data, data_train.target)
print(f"Done in {time() - t0:.3f}s")

>>> print

Fitting 5 folds for each of 40 candidates, totalling 200 fits
Done in 27.184s
```

### 5-1. 튜닝 결과값 확인
- best_estimators_.get_params()에는 dict안에 vect와 clf의 여러가지 하이퍼 파라미터 값이 저장 되어 있다.
- dict의 key 값을 사용하여 parameter_grid에 해당하는 파라미터 값을 반환한다.

```python
best_parameters = random_search.best_estimator_.get_params()
best_parameters
```





























































































