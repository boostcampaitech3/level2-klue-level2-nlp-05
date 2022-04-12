# Boostcamp AI Tech 3기: NLP-05-외않되조

---

# Project: 문장 내 개체간 관계 추출

- Wrap-up Report : [KLUE_NLP_팀 리포트(05조).pdf](https://github.com/boostcampaitech3/level2-klue-level2-nlp-05/files/8474802/KLUE_NLP_.05.pdf)

## Members

| 이름 | Github Profile | 역할 |
| --- | --- | --- |
| 공통 |  | EDA, git |
| 강나경 | angieKang | Roberta+LSTM, entity embedding layer, ko-bigbird |
| 김산 | nasmik419 | entity marker(special token at bert), tapt 시도, 타 모델 kobert 시도, 데이터 검수 |
| 김현지 | TB2715 | 실험 환경 설정, Baseline PyTorch 구조로 수정, MT5 실험 |
| 정민지 | minji2744 | Task Adaptive Pre-Training, Roberta, Bert, Model Architecture, GPT2, Electra, MT5 |
| 최지연 | jeeyeon51 | Back Translation, Roberta, Typed Entity Marker, Multi-Sentence, Curriculum Learninig, RECENT |

## 문제 개요

문장 속에서 단어간에 관계성을 파악하는 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제이며 자연어처리 응용 프로그램에서 중요한 역할을 하고 있습니다. 

이번 대회에서는 문장, 단어에 대한 정보를 통해 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시켜 단어들의 속성과 관계를 파악하며 개념을 학습하도록 하는 것을 목표로 합니다. 

## 프로젝트 수행 절차 및 방법

<img width="2689" alt="프로젝트 수행 절차" src="https://user-images.githubusercontent.com/59854630/163010472-a20ce975-f819-487b-a186-f57f86202cec.png">

### MODELS
- KLUE-BERT-base
    - typed entity marker (special token)
    - TAPT
- KLUE-RoBERTa-large
    - Typed Entity Marker (punct)
    - Multi-Sentence
    - LSTM
- SOTA
    - Curriculum Learning
    - RECENT

## 데이터셋 구조

- 문장 내 subject entity와 object entity는 총 30개의 관계로 정의됩니다.
    
    ![classess](https://user-images.githubusercontent.com/59854630/163010805-d06ffe7d-f642-4244-a03c-217ac1ebe33a.png)
    

## 실험 결과

### 리더보드 (대회 진행)

![리더보드(대회진행)](https://user-images.githubusercontent.com/59854630/163010898-9a47e75d-7ca7-447d-a7e2-c8759c5c18b6.png)

- micro_f1: 74.5286
- auprc: 79.5928

### 리더보드 (최종)

![리더보드(최종)](https://user-images.githubusercontent.com/59854630/163010940-5a396f70-1047-463a-80d8-d603119d2e2e.png)

- micro_f1: 72.9139
- auprc: 78.6011

## Requirements

> Confirmed that it runs on Ubuntu 18.04.5, Python  3.8, and pytorch 1.10.2
> 

필요한 패키지들은 아래 명령어를 입력하여 설치하실 수 있습니다.

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install torchvision==0.11.3
conda install -c conda-forge tensorboard
conda install pandas=1.1.5
pip install opencv-python==4.5.1.48
conda install -c conda-forge scikit-learn=0.24.1
conda install -c conda-forge matplotlib=3.2.1
pip install python-dotenv
pip install wandb

```

## Getting Started

### 1. 코드 구조

- **dataset.py** : 데이터 전처리 및 라벨링, 토크나이징
- **model.py** : BERT, RoBERTa, MT5, Entity Marker에 따른 다양한 모델 정의
- **inference.py** : 학습된 모델로 예측 실행
- **train.py** : loss function 정의, 파라미터들을 조절하여 학습 진행

### 2. 코드 실행 방법

1. 먼저 위의 [requirements](#requirements) 참고해 환경설정을 진행합니다.
2. `./run_train.sh` 명령어를 통해 하이퍼파라미터 옵션을 수정한 모델들을 훈련합니다.
3. `python [inference.py](http://inference.py) --dataset_folder '/opt/ml/dataset/test’ --test_dataset 'test_data.csv’ --mode ‘eval’ --task ‘entity_marker’ --model_name 't5’ --model_class 'MT5ForTypedEntityMarker’ --model_dir ‘/workspace/lv2-klue/src/models’ --output_dir '/workspace/lv2-klue/src/outputs’` 을 사용해 추론된 결과를 `output_dir`에csv로 저장합니다.
