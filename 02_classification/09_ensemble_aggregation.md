# 모형 결합
- `모형 결합 model combining` : 앙상블 방법론(ensemble methods)이라고도 한다. 여러개의 예측 모형을 결합하여 더 나은 성능의 예측을 하는 방법
- `모형 결합의 장점`
    - 계산량은 증가한다.
    - 단일 모형 사용때보다 성능의 분산이 감소한다. 과최적화를 방지할 수 있다. 
    - 개별 모형이 성능이 안좋을 경우 결합 모형의 성능이 더 향상 된다. 
- `모형 결합의 종류`
    - 취합 aggregation : 사용할 모형의 집합이 이미 결정되어 있다. 
        - 다수결 (Majority Voting)
        - 배깅 (Bagging)
        - 랜던포레스트 (Random Forests)
    - 부스팅 boosting : 사용할 모형을 점진적으로 늘려간다.
        - 에이다 부스트 (Ada Boost)
        - 그레디언트 부스트 (Gradient Boost)

    
## 1. 취합 방법 : 다수결 방법
- 취합 방법 aggregation, 다수결 방법 Majority Voting
- 기본적인 모형 결합 방법 : 완전히 다른 모형도 결합 할 수 있다. 
    - hard voting : 단순 투표, 개별 모형의 결과 기준
    - soft voting : 가중치 투표, 개별 모형의 조건부 확률의 합 기준
- `모형 결합을 사용한 성능의 향상` : 이항분포의 pmf의 합과 같다.
    - $\sum_{k > \frac{N}{2}}^{N} \binom{N}{k} p^{k}(1-p)^{N-k}$
    - p : 모형이 정답을 출력할 확률, N : 모형의 갯수
- **독립적인 모형의 수가 많을 수록 성능 향상이 일어날 가능성이 높다.**

### python
- scikit-learn의 ensemble 서브패키지 사용
- `VotingClassifier의 인수`
    - estimators : 개별 모형 목록, 리스트나 named parameter 형식으로 입력
        - 개별 모델 : vc.set_params(lr="drop")
    - voting : 문자열, {hard voting, soft voting} 중에서 선택. 디폴트 hard
        - hard : predict 반환 : 예측 클래스를 반환한다.
        - soft : predict_proba 반환 : 클래스별 조건부 확률을 반환한다. 
    - weights : 사용자 가중치 리스트
    - n_jobs : 병렬 작업 시행 여부
    - flatten_transform : 반환 행렬의 형태
    - verbose : 모수추정 과정 프린트
- 개별 모형보다 성능이 좋은 편이다.
- 성능의 분산이 작다. 과최적화가 방지된다.

## 2. 취합 방법 : 배깅
- `배깅 bagging` : 트레이닝 데이터를 랜덤하게 선택해서 다수결 모형에 적용하는 방식
    - 동일한 모형과 모형의 모수를 사용하는 대신, 부트스트래핑(bootstrapping) 방식 사용
- `트레이닝 데이터 선택 방식`    
    - Bagging : 중복사용(replacement)하는 경우
    - Pasting : 중복사용 안하는 경우
    - Random Subspaces : 데이터가 아닌 다차원의 독립변수에서 일부 차원을 선택하는 경우
    - Random Patches : 데이터 샘플과 독립변수 차원 모두 일부만 랜덤하게 사용하는 경우
- `OOB out-of-bag` : 성능 평가시 사용하는 트레이닝 데이터가 아닌 검증용 데이터

### Python
- scikit-learn의 ensemble 서브패키지 사용
- `BaggingClassifier의 인수`
    - base_estimator : 기본 모형
    - n_estimators : 모형 결합 갯수, 디폴트 10
        - 100이면 base_estimator 모형을 100개 결합한 것과 같음
    - bootstrap : 데이터 중복 사용 여부, 디폴트 True(중복사용)
    - max_samples : 선택할 데이터 샘플의 수 혹은 비율, 디폴트 1.0
    - bootstrap_features : 특징벡터 차원의 중복 사용 여부, 디폴트 False(중복 안함)
    - max_features : 다차원 독립 변수 중 선택할 차원의 수 혹은 비율, 디폴트 1.0
    - oob_score : 일반화 오류를 막기위해 외부 샘플 사용 여부, bootstrap=True 인 경우
        - out of bag : 현재 데이터가 아닌 다른 데이터를 의미하는 용어
    - warm_start : True 이면 이전 모델의 속성을 사용함, False이면 새로운 모델 생성
        - 일부 매개변수만 적용됨
        - 그리스 서치에서 이전 매개변수값을 재사용할 수 있음
    - n_jobs : 병렬 처리 여부
    - random_state : 원본 데이터의 무작위 샘플링
    
## 3. 랜덤포레스트
- `랜덤포레스트 Random Forest` : 의사결정나무를 개별 모형으로 사용하는 모형 결합 방법.
    - 데이터의 특징차원의 일부만 선택하여 사용
- 노드 분리 방법 
    - 독립변수 차원을 랜덤하게 감소시킨 후 이 중에서 기준 독립변수를 선택한다.
    - 모든 독립변수의 기준을 비교하여 최선의 독립변수를 찾는 것이 아니다.
    - 개별 모형들 사이의 상관관계가 줄어들어 모형 성능의 변동이 감소하는 효과가 있다.
- `익스트림리 랜덤트리 Extremley Randomized Trees` : 각 노드마다 랜덤하게 독립변수를 선택한다.
- meta estimator 메타 추정기의 한 종류
    - 평균화를 사용하여 예측 정확도를 개선하고 과적합을 제어한다. 
- `장점`
    - 각 독립변수의 중요도를 계산할 수 있다. feature importance
    - 포레스트의 모든 노드에서 어떤 독립변수를 사용했고, 그 노드에서 얻은 정보획득량(information gain)을 구할 수 있다.
    - 각각의 독립변수가 얻은 정보획득량의 평균을 비교하면 어떤 독립변수가 중요한지 비교할 수 있다.

### Python
- scikit-learn의 ensemble 서브패키지 사용
- `RandomForestClassifier의 인수`
    - n_estimators : 나무의 갯수
    - criterion : 노드 분류 기준 측정 방법, {"gini", "entropy", "log_loss"}
    - max_depth : None이면 min_samples_split 미만이 될때까지 노드가 분류 된다.
    - min_samples_split : 노드를 분할하기 위한 최소 샘플수
    - min_samples_leaf : 양쪽 자식 노드에 있어야할 최소 샘플수 (즉 노드의 데이터 집합의 갯수)
    - min_weight_fraction_leaf : 자식 노드에 있어야 하는 가중치(입력 샘플의) 합계의 최소 비율
    - max_features : 노드 분할 기준을 찾을 때 사용할 독립변수의 최대 수, {"sqrt", "log2", None}
    - max_leaf_nodes : 나무를 성장시키는 기본 크기값
    - min_impurity_decrease : 노드 분할 불순도 기준값, 이 기준값 보다 크거나 같은 값이면 노드가 분할 된다.
    - bootstrap : 부트스트랩 사용 여부
    - oob_score : 외부 데이터 샘플 사용 여부
    - n_jobs : 병렬 처리 여부
    - random_state : 원본 데이터의 무작위 샘플링
    - verbose : 모수 추정 과정 프린트
    - warm_start : 이전 모델의 속성값을 재사용 할지 여부
    - class_weight : 클래스 값별 가중치
    - ccp_alpha : 비용 함수에 관한 복잡성 매개변수
    - max_samples : 각 추정기 훈련에 사용할 데이터 샘플의 수
- `ExtraTreesClassifier의 인수`
    - RandomForestClassifier의 인수와 같음

