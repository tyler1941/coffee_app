import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly
import plotly.graph_objects as go

import coffee_model  # 백엔드 로직이 담긴 coffee_model 임포트

# --- Streamlit 페이지 설정 ---
st.set_page_config(
    page_title="Coffee Price Prediction",
    layout="wide",  # 화면을 넓게 사용
)

# --- 간단한 CSS로 헤더/푸터 숨기기 및 인덱스 컬럼 숨기기 ---
st.markdown(
    """
    <style>
    /* 기본 메뉴와 footer 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* DataFrame에서 row index와 blank 컬럼 숨기기 */
    .row_heading.level0 {display:none}
    .blank {display:none}
    </style>
    """,
    unsafe_allow_html=True
)

# --- 페이지 타이틀 / 헤더 ---
st.title("커피 선물 (KC=F) 5분봉 데이터 예측 및 투자 시뮬레이션")
st.write("CSV 데이터를 이용해 Prophet 모델로 예측하고, **최신 15일** 데이터를 **test 데이터**로 활용합니다.")

# 1) 데이터 로드
symbol = "KC=F"
period = "60d"
interval = "5m"

@st.cache_data
def load_and_cache_data(symbol, period, interval):
    return coffee_model.load_data(symbol, period, interval)

df_raw = load_and_cache_data(symbol, period, interval)

# 2) 훈련/테스트 데이터 분할
df_train, df_test = coffee_model.train_test_split(df_raw, test_days=15)

# --- 레이아웃: 2개 컬럼으로 Train/Test 데이터 미리보기 표시 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Train 데이터 미리보기")
    st.dataframe(df_train, use_container_width=True)

with col2:
    st.subheader("Test 데이터 미리보기")
    st.dataframe(df_test, use_container_width=True)

# 3) Prophet 학습 데이터 준비 (Train 데이터 사용)
df_prophet = coffee_model.prepare_prophet_data(df_train)

# 4) Prophet 모델 학습
model = coffee_model.train_prophet_model(df_prophet)

# 5) Test 기간 예측 (training 모델로 test 데이터 예측)
test_periods = len(df_test)
future_test = model.make_future_dataframe(periods=test_periods, freq='5min')
forecast_test = model.predict(future_test)
forecast_test = forecast_test[forecast_test['ds'] > df_prophet['ds'].max()]

with st.expander("Test 데이터 예측 결과", expanded=True):
    st.dataframe(
        forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        use_container_width=True
    )

# 6) 미래 예측 (CSV 최신 데이터 이후 15일 예측)
st.markdown("---")
st.info("**아래는 CSV의 최신 데이터 이후 미래 15일(5분봉) 예측 결과입니다.**")

latest_timestamp = df_raw.index.max()
future_periods_future = 15 * 288  # 15일치 5분봉
forecast_future = model.make_future_dataframe(periods=future_periods_future, freq='5min')
forecast_future = model.predict(forecast_future)
future_forecast = forecast_future[forecast_future['ds'] > latest_timestamp]

with st.expander("미래 예측 결과 (CSV 최신 데이터 이후 15일)", expanded=True):
    st.dataframe(
        future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        use_container_width=True
    )

# 7) 매매 시그널 생성 (미래 데이터 기준)
st.markdown("---")
st.subheader("미래 매매 시그널")
st.write(
    "최신 실제 데이터 이후의 15일치 예측값을 기준으로, "
    "임계치(%) 이상 상승 시 **Buy**, 하락 시 **Sell!!!!!**, 그 외는 **Hold** 신호를 생성합니다."
)

threshold = st.number_input(
    "변동 임계치 (%)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1
)
threshold_decimal = threshold / 100.0

last_actual_price = df_raw['Close'].iloc[-1]
future_forecast_signals = coffee_model.generate_signals(
    forecast_df=future_forecast,
    last_actual_price=last_actual_price,
    threshold_decimal=threshold_decimal
)

st.dataframe(
    future_forecast_signals[['ds', 'yhat', 'price_change', 'signal']],
    use_container_width=True
)

# 8) 투자 수익률 시뮬레이션 (미래 데이터 기준)
st.markdown("---")
st.subheader("투자 수익률 시뮬레이션 (미래 데이터 기준)")

final_amount_15 = coffee_model.simulate_trading(future_forecast_signals, initial_cash=10000)
profit_percentage_15 = ((final_amount_15 - 10000) / 10000) * 100

forecast_30 = model.make_future_dataframe(periods=30 * 288, freq='5min')
forecast_30 = model.predict(forecast_30)
forecast_30 = forecast_30[forecast_30['ds'] > latest_timestamp]
forecast_30 = coffee_model.generate_signals(forecast_30, last_actual_price, threshold_decimal)
final_amount_30 = coffee_model.simulate_trading(forecast_30, initial_cash=10000)
profit_percentage_30 = ((final_amount_30 - 10000) / 10000) * 100

st.write(f"**초기 투자금:** $10,000")
st.write(f"**15일 후 예상 투자금:** ${final_amount_15:,.2f} (수익률: {profit_percentage_15:.2f}%)")
st.write(f"**30일 후 예상 투자금:** ${final_amount_30:,.2f} (수익률: {profit_percentage_30:.2f}%)")
