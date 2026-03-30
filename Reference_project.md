# CatBoost Starter - [LB 0.60]

### 의료 데이터 분석 참고 프로젝트 제안서

Info.

EEG 신호를 분석하여 6가지 주요 뇌 활동(발작, LPD, GPD, LRDA, GRDA, 기타)을 자동 분류하는 머신러닝 모델 구축.

GOAL: 시계열 파형 데이터인 EEG를 정도의 따라 분류

DATA: HMS - Harmful Brain Activity Classification

### 참고 프로젝트 선정 이유

본 프로젝트에서 개발자는 EEG 데이터를 활용하여 발작의 정도를 자동으로 분류하는 머신러닝/딥러닝 모델을 개발하고자 함. 발작의 정도를 여러 단계로 분류하는 모델을 구축하여 환자의 상태를 보다 정밀하게 모니터링할 수 있는 향후 프로젝트에 많은 도움이 될 것이라 판단하여 선정함.

## 원본 데이터 | Harmful Brain Activity Classification

<img width="560" height="508" alt="image" src="https://github.com/user-attachments/assets/e3faa07e-8bfa-4288-9b8a-6f2c6b3bdcaf" />

원본 데이터 출처: Harvard Medical School

데이터 크기 및 구성
- Train Set: 약 106,800개의 행(row)으로 구성된 메타데이터(train.csv)와 수만 개의 EEG/Spectrogram 개별 파일
- Test Set: 1개 (대회 규정)

EEG data(**독립 변수 x**): Fp1, F3, C3, P3, Fz, Cz, Pz, Fp2, F4, C4, P4, F7, T3, T5, O1, F8, T4, T6, O2, EKG (총 20개 채널)

| Column Name | 설명 | 예시 | 역할 |
|-------------|------|------|------|
| eeg_id | EEG 전체 신호 파일 ID | 1628180742 | train_eegs/{eeg_id}.parquet 파일과 연결 |
| eeg_sub_id | EEG 파일 내 세그먼트 번호 | 0, 1, 2 | 하나의 EEG를 여러 샘플로 나눈 인덱스 |
| eeg_label_offset_seconds | EEG에서 라벨이 적용되는 시작 시간 (초) | 353733 | 해당 위치에서 일정 구간 EEG 추출 |
| spectrogram_id | Spectrogram 파일 ID | 127492639 | train_spectrograms/{id}.parquet 연결 |
| spectrogram_sub_id | Spectrogram 세그먼트 번호 | 0 | spectrogram segmentation 인덱스 |
| spectrogram_label_offset_seconds | Spectrogram에서 라벨 시작 위치 | 0.0 | spectrogram 기준 offset |
| label_id | 라벨 ID | 353733 | 라벨 식별용 ID |
| patient_id | 환자 ID | 42516 | 같은 환자의 여러 EEG 존재 |
| expert_consensus | 전문가 최종 판단 라벨 | Seizure | 모델이 예측해야 할 target |
| seizure_vote | Seizure 투표 수 | 3 | 전문가 투표 수 |
| lpd_vote | LPD 투표 수 | 0 | 전문가 투표 수 |
| gpd_vote | GPD 투표 수 | 0 | 전문가 투표 수 |
| lrda_vote | LRDA 투표 수 | 0 | 전문가 투표 수 |
| grda_vote | GRDA 투표 수 | 0 | 전문가 투표 수 |
| other_vote | 기타 라벨 투표 수 | 0 | 전문가 투표 수 |

_filtered spectogream_: 위 19개 채널을 그대로 쓰지 않고, 4개의 영역(LL, LP, RP, RR)으로 묶어 이미지화. 각 영역은 128(주파수) x 256(시간) 크기의 행렬(Matrix) 데이터가 되어 모델의 입력값이 됨.

6가지 뇌 활동(**종속 변수 y**): 

- seizure_vote (발작)

- lpd_vote (좌측 주기적 이당성 방전)

- gpd_vote (일반화된 주기적 방전)

- lrda_vote (좌측 리드미컬 델타 활동)

- grda_vote (일반화된 리드미컬 델타 활동)

- other_vote (기타)

## 데이터 전처리

### 결측치 처리 및 피처 엔지니어링

결측치: EEG 신호 내에 결측치가 발생한 경우, 해당 신호의 평균값(np.nanmean)으로 채우거나 결측치가 너무 많을 경우 해당 구간을 0으로 처리하여 모델 학습의 불안정성을 방지

피처 엔지니어링:

1. 스펙트로그램 통계:

10분(Full) 윈도우와 중심부 20초 윈도우에서 각각 400개 주파수 대역의 평균과 최소값을 추출.

2. EEG 기반 스펙트로그램 특징:

Raw EEG 데이터를 직접 스펙트로그램으로 변환(STFT 등 활용)하여 새로운 이미지 특징을 생성. 이를 다시 평균, 최소, 최대, 표준편차의 4가지 통계치로 압축하여 모델에 입력함.

3. 데이터 리샘플링:

학습 효율을 위해 중복되는 eeg_id 중 첫 번째 관측치나 대표값만을 추출하여 데이터 크기를 약 17,000개 정도로 경량화하여 학습 속도 증대.

# 모델링 분석

### 사용된 알고리즘: CatBoost (Gradient Boosting Decision Tree)

여러 개의 의사결정 나무(Decision Tree)를 순차적으로 학습시켜 오차를 보정하는 그래디언트 부스팅 방식.

## 알고리즘 선정 이유

### 왜 CatBoost (Gradient Boosting Decision Tree)인가?

1. 시계열 데이터의 통계적 요약 (Tabular Data로의 변환)

이 대회의 데이터는 복잡한 시계열(EEG)이지만, 개발자는 이를 주파수별 통계치(평균, 최솟값 등)로 요약하여 정형 데이터(Tabular Data) 형태로 변환함. 정형 데이터 분류 작업에서는 현재까지도 XGBoost, LightGBM, CatBoost 같은 트리 기반 앙상블 모델이 딥러닝보다 훨씬 빠르고 안정적인 성능을 보임.

2. KL Divergence 손실 함수 최적화

본 프로젝트의 평가지표는 예측 확률 분포와 실제 투표 분포 사이의 차이를 측정하는 KL Divergence. CatBoost는 커스텀 손실 함수를 설정하기 용이하며, 확률값(0~1 사이의 연속형 수치)을 예측하는 회귀 문제로 접근했을 때 매우 정교한 예측이 가능함.

3. 범주형 데이터 및 노이즈에 유용

CatBoost는 대칭 트리를 사용하여 예측 속도가 매우 빠르고 과적합을 방지하는 능력이 탁월함. 뇌파 데이터처럼 노이즈가 많은 데이터셋에서 모델이 너무 세부적인 특징에 집착하지 않게 도움.

## 성능 평가 분석

### 사용된 성능 평가 지표 KL Divergence (Kullback-Leibler Divergence)

뇌파 진단은 전문가들 사이에서도 의견이 갈릴 수 있음. "이건 100% 발작이다"라고 단정하기보다, "발작일 확률이 60%, 다른 증상일 확률이 40%다"라는 식의 불확실성을 모델이 얼마나 잘 학습했는지 평가하기 위함.

- 전문가들이 투표한 6개 클래스의 확률 분포(ex: Seizure 0.5, LPD 0.1, ...)와 모델이 예측한 확률 분포 사이의 거리 계산.

- 두 분포가 완벽하게 일치하면 값은 0이 됨. 즉 예측이 틀릴수록 값이 커지는 손실 개념의 지표.

### 한계 극복을 위한 시도 및 최적화 전략

1. 데이터 중복 제거 및 중앙값 추출

동일한 eeg_id 내에 수많은 중복 샘플이 있어 학습이 느리고 특정 환자 데이터에 과적합됨. 따라서 eeg_id에서 투표 결과가 가장 명확하거나 시간상 중앙에 위치한 샘플 하나만 선택하여 데이터의 질을 높임.

2. 하이퍼파라미터 튜닝

CatBoost 모델의 성능을 극대화하기 위해 다음과 같은 파라미터를 조정

- iterations: 충분한 학습을 위해 반복 횟수를 최적화.

- learning_rate: 너무 빠르지도 느리지도 않은 최적의 학습률을 설정하여 지역 최적점(Local Minimum)에 빠지는 것을 방지.

- loss_function: 모델이 직접 KL Divergence를 최적화하도록 설정하거나, 그와 유사한 동작을 하는 Regression 기반 손실 함수를 채택.

3. Group K-Fold Cross Validation

같은 환자의 데이터가 Train과 Validation에 섞이면 모델이 환자의 개인적 특성을 외워버리는 'Data Leakage' 발생. 따라서 patient_id를 기준으로 그룹을 나누어, 학습에 사용되지 않은 새로운 환자의 뇌파를 얼마나 잘 맞히는지 검증. 이를 통해 리더보드 점수와 실제 검증 점수 사이의 간극을 줄임.

## 최종 성능 요약

본 프로젝트는 전문가 투표 분포와 모델 예측 분포 사이의 차이를 측정하는 KL Divergence 지표에서 0.60이라는 우수한 점수를 기록하며, 복잡한 딥러닝 없이도 효율적인 피처 엔지니어링만으로 높은 분류 정확도와 일반화 성능을 입증한 모델임. 이후 많은 프로젝트들이 이 모델의 예측값을 토대로 이미지나 시계열 같은 딥러닝 모델들의 토대가 되는, 상당히 유의미한 프로젝트임.
