import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, percentileofscore

def spx_history(symbol, period, interval, start=None, **kwargs): 
    if start:
        spx = yf.download(symbol, start=start, period=period, interval=interval)
        spy = yf.download('SPY', start=start, period=period, interval=interval)
    else:
        spx = yf.download(symbol, period=period, interval=interval)
        spy = yf.download('SPY', period=period, interval=interval)

    spx['spy_vol'] = spy.Volume
    spx.columns = spx.columns.droplevel(level=1)
    
    # Calculate True Range (TR)
    spx['High-Low'] = spx['High'] - spx['Low']
    spx['High-PrevClose'] = abs(spx['High'] - spx['Close'].shift(1))
    spx['Low-PrevClose'] = abs(spx['Low'] - spx['Close'].shift(1))
    spx['abs(Close-Open)'] = abs(spx['Close'] - spx['Open'])
    spx['High-Open'] = abs(spx['High'] - spx['Open'])
    spx['abs(Low-Open)'] = abs(spx['Open'] -  spx['Low'])
    spx['Open Gap pts'] = abs(spx['Open']-spx['Close'].shift(1))
    # spx['UpWick'] = abs(spx['High'] - spx['Close']) if spx['Close'] >= spx['Open'] else abs(spx['High'] - spx['Open'])
    # spx['DnWick'] = abs(spx['Open'] - spx['Low']) if spx['Close'] >= spx['Open'] else abs(spx['Close'] - spx['Low'])
    # Calculate the upper wick
    spx['UpWick'] = np.where(spx['Close'] >= spx['Open'],
                             spx['High'] - spx['Close'],
                             spx['High'] - spx['Open'])
    
    # Calculate the lower wick
    spx['DnWick'] = np.where(spx['Close'] >= spx['Open'],
                             spx['Open'] - spx['Low'],
                             spx['Close'] - spx['Low'])

    # True Range is the maximum of the above
    spx['TR'] = spx[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Calculate ATR (14-day by default)
    # atr_period = 14
    # spx['ATR'] = spx['TR'].rolling(window=atr_period).mean()
    spx = spx.dropna(axis=0)

    return spx

def day_range_stats(days,spx):
    spx_data = spx.tail(days)
    # Calculate the daily price range (High - Low)
    spx_data['Price Range'] = spx_data['High'] - spx_data['Low']
    stat = {
            'Max Range': spx_data['Price Range'].max(),
            'Third Quartile (Q3)': spx_data['Price Range'].quantile(0.75),
            'Average Range': spx_data['Price Range'].mean(),
            'First Quartile (Q1)': spx_data['Price Range'].quantile(0.25),
            'Minimum Range': spx_data['Price Range'].min(),
            'Median Range': spx_data['Price Range'].median()
        }
    # Prepare data for the second table (statistical summary)
    stats_df = pd.DataFrame.from_dict(stat, orient='index', columns=['Value'])
    stats_df['Value'] = stats_df['Value'].round(2)

    return stats_df

def single_bar_stats(df):
    df['Date'] = df.index
    df['Median'] = (df['High'] + df['Low']) / 2
    df['C-O'] = (df['Close'] - df['Open'])
    df['%'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    df = df.tail(1).round(2)
    return df



def spx_range_report_v2(symbol='^GSPC',period='2mo',interval='1d'):
    # Step 1: 下載資料
    # spx = yf.download('^GSPC', period='2mo', interval='1d')
    # spx['Range'] = spx['High'] - spx['Low']
    # range_data = spx['Range'].dropna()

    #spx = spx_history('^GSPC',period='2mo', interval='1d')
    spx = spx_history(symbol,period=period, interval=interval)
    range_data = spx['High-Low'].dropna()

    # Step 2: 擬合 log-normal 分布
    mean_x = np.mean(range_data)
    std_x = np.std(range_data, ddof=1)
    sigma = np.sqrt(np.log(1 + (std_x / mean_x)**2))
    mu = np.log(mean_x) - 0.5 * sigma**2
    lognorm_dist = lognorm(s=sigma, scale=np.exp(mu))

    # Step 3: 最近 10 天統計
    recent_range = spx['High-Low'].iloc[-10:]
    mean_val = recent_range.mean()
    min_val = recent_range.min()
    q1 = np.percentile(recent_range, 25)
    median = np.median(recent_range)
    q3 = np.percentile(recent_range, 75)
    max_val = recent_range.max()

    # Step 4: PDF / CDF 計算
    x = np.linspace(range_data.min(), range_data.max(), 1000)
    pdf_vals = lognorm_dist.pdf(x)
    cdf_vals = lognorm_dist.cdf(x)

    # Step 5: 計算 CDF / ECDF
    cdf_val = lognorm_dist.cdf(mean_val)
    right_tail_prob = 1 - cdf_val
    ecdf_percentile = percentileofscore(range_data, median)

    # Step 6: 畫圖（PDF + CDF 同圖雙軸）
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(x, pdf_vals, label='PDF (Density)', color='blue')
    # ax1.axvline(mean_val, color='red', linestyle='--', label=f'Avg = {mean_val:.2f}')
    # ax1.axvline(min_val, color='teal', linestyle='--', label=f'10-Day Min = {min_val:.2f}')
    ax1.set_ylabel('PDF', color='blue')

    # 統計線與值一起放入 legend
    for val, label, color in zip(
        [mean_val, min_val, q1, median, q3, max_val],
        ['Avg', 'Min', 'Q1', 'Median', 'Q3', 'Max'],
        ['red', 'gray', 'purple', 'orange', 'blue', 'black']
    ):
        label_with_value = f"{label} = {val:.2f}"
        ax1.axvline(val, linestyle=':', color=color, label=label_with_value)

    # Twin axis for CDF
    ax2 = ax1.twinx()
    ax2.plot(x, cdf_vals, label='CDF (Cumulative)', color='green')
    ax2.axhline(cdf_val, color='green', linestyle='--')
    ax2.set_ylabel('CDF', color='green')

    # 統計點 1 - CDF 標註平均值右尾機率
    for val, label, color in zip(
        [mean_val, min_val, q1, median, q3, max_val],
        ['Avg','Min', 'Q1', 'Median', 'Q3', 'Max'],
        ['red','gray', 'purple', 'orange', 'blue', 'black']
    ):
        prob = 1 - lognorm_dist.cdf(val)
        ax2.text(val + 3, 0.1, f"{label}\n1-CDF={prob:.3f}", fontsize=9, color=color)

    # 標註平均值右尾機率
    # ax2.text(mean_val + 3, 0.1, f'1-CDF = {right_tail_prob:.3f}', color='red')

    ax1.set_xlabel('SPX High - Low Range (points)')
    ax1.set_title('SPX Intraday Range - PDF + CDF (past 2 months)')
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax1.grid(True)
    plt.tight_layout()
    # plt.show()

    # Step 7: 顯示報告
    report = {
        "average range": round(mean_val, 2),
        "log-normal 1 - CDF (右尾機率)": round(right_tail_prob, 4),
        "empirical percentile (ECDF)": f"{round(ecdf_percentile, 2)}%",
        "Min": round(min_val, 2),
        "Q1": round(q1, 2),
        "Median": round(median, 2),
        "Q3": round(q3, 2),
        "Max": round(max_val, 2)
    }
    # 額外：建立統計表格 DataFrame
    stats_labels = ['Min', 'Q1', 'Mean', 'Median', 'Q3', 'Max']
    stats_values = [min_val, q1, mean_val, median, q3, max_val]
    cdf_1_minus = [1 - lognorm_dist.cdf(v) for v in stats_values]
    ecdf_vals = [1 - percentileofscore(range_data, v) / 100 for v in stats_values]

    stats_df = pd.DataFrame({
        "Right tail Statistic": stats_labels,
        "Value": np.round(stats_values, 2),
        "1 - CDF (Right Tail Prob)": np.round(cdf_1_minus, 4),
        "1 - ECDF (Percentile)": [f"{int(p * 100)}%" for p in ecdf_vals]
    })

    # print("\n=== SPX Range CDF vs ECDF Summary ===")
    # print(stats_df.to_markdown(index=False))

    return report, fig
