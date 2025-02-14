import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.set_page_config(
    page_title="Coffee Price Prediction",
    layout="centered",
)

st.title("커피 선물 (KC=F) 5분봉 데이터 예측 및 투자 시뮬레이션")
st.write("Yahoo Finance에서 커피 선물 데이터를 가져와 Prophet으로 예측하고, 매매 시그널을 분석하여 투자 수익을 계산합니다.")

symbol = "KC=F"
period = "60d"
interval = "5m"

@st.cache_data
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df.index = df.index.tz_localize(None)
    return df

df_raw = load_data(symbol, period, interval)

# 데이터 분할: 75% 학습, 25% 테스트
train_size = int(len(df_raw) * 0.75)
df_train = df_raw.iloc[:train_size]
df_test = df_raw.iloc[train_size:]

# Prophet 모델 학습 데이터 준비
# (이후 매매 시그널 생성을 위해 df_prophet 변수명 사용)
df_prophet = df_train[['Close']].reset_index()
df_prophet.columns = ['ds', 'y']
df_prophet.dropna(inplace=True)

# Prophet 모델 학습
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=True
)
model.fit(df_prophet)

# 미래 예측 수행
future_periods = st.slider(
    "예측할 5분봉 개수(1일=288 정도)",
    min_value=50,
    max_value=600,
    value=288,
    step=10
)
future_df = model.make_future_dataframe(periods=future_periods, freq='5min')
forecast = model.predict(future_df)

st.write("**미래 예측 결과 미리보기**")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# --------------------------------------------------------------------------------
# 5. 매매 시그널 생성 (수정된 부분)
# --------------------------------------------------------------------------------
st.subheader("5) 매매 시그널")
st.write(
    "예측 결과와 마지막 실제 관측치(종가)를 비교하여 단순 임계치 기반 매매 시그널을 생성합니다.\n"
    "임계치 (%)를 조절하여 매수(Buy) / 매도(Sell) 신호를 확인할 수 있습니다."
)
# 사용자로부터 임계치 입력 (기본 1%)
threshold = st.number_input("변동 임계치 (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
threshold_decimal = threshold / 100.0

# Prophet의 미래 데이터프레임 중, 실제 관측 이후의 예측만 사용
future_forecast = forecast[forecast['ds'] > df_prophet['ds'].max()]
# 마지막 실제 가격 (가장 최근 관측된 종가)
last_actual_price = df_prophet['y'].iloc[-1]
# 예측 가격 대비 변화율 계산
future_forecast = future_forecast.copy()  # SettingWithCopyWarning 방지
future_forecast['price_change'] = (future_forecast['yhat'] - last_actual_price) / last_actual_price

# 단순 매매 시그널 생성 함수 (문자열 반환)
def get_signal(change):
    if change > threshold_decimal:
        return "Buy"
    elif change < -threshold_decimal:
        return "Sell"
    else:
        return "Hold"

future_forecast['signal'] = future_forecast['price_change'].apply(get_signal)

st.write("**예측을 바탕으로 한 매매 시그널 (미래 데이터)**")
st.dataframe(future_forecast[['ds', 'yhat', 'price_change', 'signal']].tail(10))

# ------------------------------------
# 7. 투자 수익률 시뮬레이션
# ------------------------------------
st.subheader("7) 투자 수익률 시뮬레이션")

# 시뮬레이션 함수 (신호가 "Buy", "Sell", "Hold"로 표기됨)
def simulate_trading(forecast_df, initial_cash=10000):
    cash = initial_cash
    position = 0
    for i in range(len(forecast_df)):
        # 매수 신호 시, 보유 현금으로 전량 매수
        if forecast_df.iloc[i]['signal'] == "Buy" and cash > 0:
            position = cash / forecast_df.iloc[i]['yhat']
            cash = 0
        # 매도 신호 시, 보유 포지션 전량 매도
        elif forecast_df.iloc[i]['signal'] == "Sell" and position > 0:
            cash = position * forecast_df.iloc[i]['yhat']
            position = 0
    final_value = cash if cash > 0 else position * forecast_df.iloc[-1]['yhat']
    return final_value

# 15일 및 30일 예측 데이터프레임 생성
# 15일: 이미 forecast(미래 예측) 중 실제 관측 이후의 데이터인 future_forecast 사용
final_amount_15 = simulate_trading(future_forecast)
profit_percentage_15 = ((final_amount_15 - 10000) / 10000) * 100

# 30일 예측 데이터 생성
future_df_30 = model.make_future_dataframe(periods=30 * 288, freq='5min')
forecast_30 = model.predict(future_df_30)
forecast_30 = forecast_30.copy()
forecast_30['price_change'] = (forecast_30['yhat'] - last_actual_price) / last_actual_price
forecast_30['signal'] = forecast_30['price_change'].apply(get_signal)

final_amount_30 = simulate_trading(forecast_30)
profit_percentage_30 = ((final_amount_30 - 10000) / 10000) * 100

st.write(f"**초기 투자금: $10,000**")
st.write(f"**15일 후 예상 투자금: ${final_amount_15:,.2f}**")
st.write(f"**15일 예상 수익률: {profit_percentage_15:.2f}%**")
st.write("---")
st.write(f"**30일 후 예상 투자금: ${final_amount_30:,.2f}**")
st.write(f"**30일 예상 수익률: {profit_percentage_30:.2f}%**")
