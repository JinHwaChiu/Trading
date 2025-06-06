import streamlit as st
import yfinance as yf
from utils import *
import plotly.graph_objects as go
import pandas as pd
# your helper module

st.set_page_config(page_title="VIX", layout="wide")
st.title("SPX Range rolling std/mean")

# You can reuse your logic here
st.write("This is the range analysis page.")


st.sidebar.header("輸入參數")
# symbol = st.sidebar.text_input("Ticker Symbol", value="^GSPC")
period = st.sidebar.selectbox("Period", ["1mo", "2mo", "3mo", "6mo", "12mo", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
rolling = st.sidebar.number_input("rolling window day", min_value=5, step=1)

if st.sidebar.button("執行分析"):

    
    spx = yf.download('^GSPC', period=period, interval=interval)
    # 計算每日 Range（高 - 低）
    spx['Range'] = spx['High'] - spx['Low']
    
    # 計算 10日的 rolling std 與 mean
    spx['Range_std'] = spx['Range'].rolling(window=rolling).std()
    spx['Range_mean'] = spx['Range'].rolling(window=rolling).mean()
    spx['Range_std_ratio'] = spx['Range_std'] / spx['Range_mean']
    
    # 偵測異常收斂或發散
    spx['Contraction_Flag'] = spx['Range_std_ratio'] < 0.2
    spx['Expansion_Flag'] = spx['Range_std_ratio'].pct_change() > 0.5
    
    spx = only_market_close_data(spx)

    # 整理顯示用 DataFrame
    summary_df = spx[['Range', 'Range_std', 'Range_mean', 'Range_std_ratio', 'Contraction_Flag', 'Expansion_Flag']].dropna()
    
    
    
    # 假設 summary_df 已存在
    fig = go.Figure()
    
    # 主曲線
    fig.add_trace(go.Scatter(
        x=summary_df.index,
        y=summary_df['Range_std_ratio'],
        mode='lines+markers',
        name=f'Range Std Ratio ({rolling}d)',
        hovertemplate='日期: %{x}<br>比值: %{y:.4f}'
    ))
    
    # Contraction 點
    fig.add_trace(go.Scatter(
        x=summary_df.index[summary_df['Contraction_Flag']],
        y=summary_df['Range_std_ratio'][summary_df['Contraction_Flag']],
        mode='markers',
        name='Contraction',
        marker=dict(symbol='triangle-down', size=10, color='blue'),
        hovertemplate='日期: %{x}<br>收斂比值: %{y:.4f}'
    ))
    
    # Expansion 點
    fig.add_trace(go.Scatter(
        x=summary_df.index[summary_df['Expansion_Flag']],
        y=summary_df['Range_std_ratio'][summary_df['Expansion_Flag']],
        mode='markers',
        name='Expansion',
        marker=dict(symbol='triangle-up', size=10, color='red'),
        hovertemplate='日期: %{x}<br>發散比值: %{y:.4f}'
    ))
    
    # 門檻線
    fig.add_hline(y=0.2, line_dash="dash", line_color="gray",
                  annotation_text="Contraction Threshold (0.2)", annotation_position="top left")
    
    fig.update_layout(
        title="SPX 波動異常轉折偵測",
        xaxis_title="日期",
        yaxis_title="Std / Mean Ratio",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    # 建立三種不同 rolling ratio
    summary_df['Ratio_5'] = summary_df['Range'].rolling(5).std() / summary_df['Range'].rolling(5).mean()
    summary_df['Ratio_10'] = summary_df['Range'].rolling(10).std() / summary_df['Range'].rolling(10).mean()
    summary_df['Ratio_20'] = summary_df['Range'].rolling(20).std() / summary_df['Range'].rolling(20).mean()
    
    # 畫圖
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=summary_df.index, y=summary_df['Ratio_5'],
                             mode='lines', name='5-Day Std/Mean Ratio', line=dict(width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=summary_df.index, y=summary_df['Ratio_10'],
                             mode='lines', name='10-Day Std/Mean Ratio', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=summary_df.index, y=summary_df['Ratio_20'],
                             mode='lines', name='20-Day Std/Mean Ratio', line=dict(width=2, dash='dash')))
    
    fig.update_layout(
        title="不同 Rolling Window 的波動結構比值對比圖",
        xaxis_title="日期",
        yaxis_title="Std / Mean Ratio",
        template="plotly_white",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
