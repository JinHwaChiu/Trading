import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import streamlit as st
from utils import spx_history,single_bar_stats,day_range_stats


def intraday_period_range(data,time):
    data = np.array(data)
    
    # 擬合 log-normal：取 log，計算 μ, σ
    log_data = np.log(data)
    mu = np.mean(log_data)
    sigma = np.std(log_data)
    
    # 建立 log-normal 分布物件
    dist = lognorm(s=sigma, scale=np.exp(mu))
    
    # 計算統計量
    q1 = dist.ppf(0.25)
    median = dist.median()
    mean = dist.mean()
    q3 = dist.ppf(0.75)
    min_val = data.min()
    max_val = data.max()
    
    # CDF 機率（表示落在該點以下的機率）
    cdf_vals = {
        'Min': dist.cdf(min_val),
        'Q1': dist.cdf(q1),
        'Median': dist.cdf(median),
        'Mean': dist.cdf(mean),
        'Q3': dist.cdf(q3),
        'Max': dist.cdf(max_val),
    }
    
    # 繪圖
    fig = plt.figure(figsize=(8, 5))
    plt.hist(data, bins=10, alpha=0.5, label='High-Low Histogram', color='skyblue', edgecolor='black')
    
    # 加上統計標記線與文字
    for label, val in zip(['Min', 'Q1', 'Median', 'Mean', 'Q3', 'Max'], [min_val, q1, median, mean, q3, max_val]):
        plt.axvline(val, linestyle='--', label=f'{label}: {val:.2f} ({cdf_vals[label]*100:.1f}%)')
    dict_time = {6:'06:30~07:30', 7:'07:30~08:30', 8:'08:30~09:30',9:'09:30~10:30',10:'10:30~11:30',11:'11:30~12:30',12:'12:30~13:00'}
    plt.title(f'Log-Normal Stats for SPX {dict_time[time]} High-Low')
    plt.xlabel('High-Low Range (pts)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    return fig 

st.set_page_config(page_title="SPX Range Analysis", layout="wide")
st.title("Intraday 1h range")

# You can reuse your logic here
st.write("This is the range analysis page.")

test = spx_history('^GSPC',period='12mo', interval='1h')
test.index = test.index.tz_convert('America/Los_Angeles')
test = only_market_close_data(test)
cols = st.columns(2)  # Create 2 columns
col_idx = 0           # Track which column to use

for i, j in test.groupby(test.index.hour)['High-Low']:
    data = pd.DataFrame(j)
    fig = intraday_period_range(data, i)
    
    # Plot in alternating columns
    with cols[col_idx]:
        st.pyplot(fig)
    
    # Switch to next column, reset after 1
    col_idx = (col_idx + 1) % 2
    if col_idx == 0:
        cols = st.columns(2)  # Start new row after every 2 plots
