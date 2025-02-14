# 커피 선물 예측 및 투자 시뮬레이션

이 프로젝트는 Yahoo Finance의 커피 선물 데이터를 기반으로 Prophet 모델을 사용하여
예측을 수행하고, 매매 시그널 및 투자 수익률을 시뮬레이션하는 Streamlit 앱입니다.

## 폴더 구조

coffee_prediction_app/ ├── app/ │ ├── init.py │ ├── coffee_model.py # 백엔드: 데이터 처리, 모델 학습 및 시뮬레이션 │ └── coffee_app.py # 프론트엔드: Streamlit UI 구성 ├── data/ │ └── coffee.csv # Yahoo Finance에서 다운로드 받은 데이터 파일 ├── requirements.txt # 프로젝트 의존성 목록 └── README.md # 프로젝트 설명서


## 설치 및 실행 방법

1. 필요한 라이브러리 설치:

   ```bash
   pip install -r requirements.txt
