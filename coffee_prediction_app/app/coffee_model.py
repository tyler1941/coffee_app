import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet

def load_data(symbol, period, interval):
    """
    Yahoo Finance로부터 데이터 다운로드 후,
    index의 타임존 제거(tz_localize(None))까지 수행.
    """
    df = yf.download(symbol, period=period, interval=interval)
    df.index = df.index.tz_localize(None)
    return df

def train_test_split(df, test_days=15):
    """
    DataFrame을 training과 testing 데이터로 분리합니다.
    test 데이터는 최신 `test_days`일에 해당하는 데이터(5분봉 기준, 1일 약 288행)로 구성됩니다.
    """
    records_per_day = 288  # 5분봉 기준 1일 약 288행
    test_rows = test_days * records_per_day
    if test_rows >= len(df):
        raise ValueError(f"전체 데이터 행 수가 {test_days}일치(약 {test_rows}행)보다 적습니다.")
    df_train = df.iloc[:-test_rows]
    df_test = df.iloc[-test_rows:]
    return df_train, df_test

def prepare_prophet_data(df_train):
    """
    Prophet 모델 학습에 맞도록
    - (DateTimeIndex → 일반 열)로 변환
    - 'Close' 컬럼을 y로 사용
    - 컬럼명을 ['ds','y'] 형태로 변경
    - 결측 제거
    """
    df_prophet = df_train[['Close']].reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet.dropna(inplace=True)
    return df_prophet

def train_prophet_model(df_prophet):
    """
    Prophet 모델 생성 및 학습.
    """
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    model.fit(df_prophet)
    return model

def make_forecast(model, future_periods, freq='5min'):
    """
    future_periods만큼 미래 예측을 수행.
    freq='5min'으로 5분 간격 예측.
    """
    future_df = model.make_future_dataframe(periods=future_periods, freq=freq)
    forecast = model.predict(future_df)
    return forecast

def generate_signals(forecast_df, last_actual_price, threshold_decimal):
    """
    예측 결과(forecast_df)에 대해
    - 실제 마지막 종가(last_actual_price)와 비교해 변화율 계산
    - 임계치(threshold_decimal) 기준으로 Buy/Sell/Hold 시그널 생성
    """
    future_forecast = forecast_df.copy()
    
    # 예측 값과 실제 마지막 종가 대비 변화율
    future_forecast['price_change'] = (
        (future_forecast['yhat'] - last_actual_price) 
        / last_actual_price
    )

    def get_signal(change):
        if change > threshold_decimal:
            return "Buy"
        elif change < -threshold_decimal:
            return "Sell"
        else:
            return "Hold"

    future_forecast['signal'] = future_forecast['price_change'].apply(get_signal)
    return future_forecast

def simulate_trading(forecast_df, initial_cash=10000):
    """
    'signal' 컬럼(Buy/Sell/Hold)에 따라 단순 매매 시뮬레이션:
    - Buy 시: 전량 매수 (보유 현금을 예측가격 기준으로 매수)
    - Sell 시: 전량 매도 (보유 포지션을 예측가격에 매도)
    - Hold 시: 아무것도 하지 않음
    최종 보유 자산(현금 혹은 종목가치)을 반환.
    """
    cash = initial_cash
    position = 0  # 보유 중인 종목 수량

    for i in range(len(forecast_df)):
        signal = forecast_df.iloc[i]['signal']
        price = forecast_df.iloc[i]['yhat']

        # 매수 신호
        if signal == "Buy" and cash > 0:
            position = cash / price
            cash = 0
        # 매도 신호
        elif signal == "Sell" and position > 0:
            cash = position * price
            position = 0

    # 마지막에 종목을 보유 중이면, 마지막 예측가로 평가
    final_value = cash if cash > 0 else position * forecast_df.iloc[-1]['yhat']
    return final_value
