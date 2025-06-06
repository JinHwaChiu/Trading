import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, percentileofscore
from datetime import datetime
import seaborn as sns
from scipy.stats import pearsonr

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



#### Page_1 spx_range_analysis

def fit_lognormal_distribution(data):
    log_data = np.log(data)
    mu = np.mean(log_data)
    sigma = np.std(log_data)
    dist = lognorm(s=sigma, scale=np.exp(mu))
    return dist, mu, sigma


def calculate_summary_stats(data):
    return {
        "mean": data.mean(),
        "min": data.min(),
        "q1": np.percentile(data, 25),
        "median": np.median(data),
        "q3": np.percentile(data, 75),
        "max": data.max()
    }


def generate_pdf_cdf_plot(data, dist, summary):
    x = np.linspace(data.min(), data.max(), 1000)
    pdf_vals = dist.pdf(x)
    cdf_vals = dist.cdf(x)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(x, pdf_vals, label='PDF (Density)', color='blue')
    ax1.set_ylabel('PDF', color='blue')

    for label, color in zip(['mean', 'min', 'q1', 'median', 'q3', 'max'],
                            ['red', 'gray', 'purple', 'orange', 'blue', 'black']):
        val = summary[label]
        ax1.axvline(val, linestyle=':', color=color, label=f"{label.upper()} = {val:.2f}")

    ax2 = ax1.twinx()
    ax2.plot(x, cdf_vals, label='CDF (Cumulative)', color='green')
    ax2.set_ylabel('CDF', color='green')

    for label, color in zip(['mean', 'min', 'q1', 'median', 'q3', 'max'],
                            ['red', 'gray', 'purple', 'orange', 'blue', 'black']):
        val = summary[label]
        prob = 1 - dist.cdf(val)
        ax2.text(val + 3, 0.1, f"{label}\n1-CDF={prob:.3f}", fontsize=9, color=color)

    ax1.set_xlabel('SPX High - Low Range (points)')
    ax1.set_title('SPX Intraday Range - PDF + CDF (past 2 months)')
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax1.grid(True)
    plt.tight_layout()
    return fig


def build_summary_table(data, dist, summary):
    labels = ['min', 'q1', 'mean', 'median', 'q3', 'max']
    values = [summary[l] for l in labels]
    cdf_1_minus = [1 - dist.cdf(v) for v in values]
    ecdf_vals = [1 - percentileofscore(data, v) / 100 for v in values]

    df = pd.DataFrame({
        "Right tail Statistic": [l.upper() for l in labels],
        "Value": np.round(values, 2),
        "1 - CDF (Right Tail Prob)": np.round(cdf_1_minus, 4),
        "1 - ECDF (Percentile)": [f"{int(p * 100)}%" for p in ecdf_vals]
    })
    return df


def spx_range_report(data,past_few_days):
    dist, mu, sigma = fit_lognormal_distribution(data)
    summary = calculate_summary_stats(data[-past_few_days:])
    fig = generate_pdf_cdf_plot(data, dist, summary)
    df = build_summary_table(data, dist, summary)
    # print("\n=== SPX Range CDF vs ECDF Summary ===")
    return df,fig
    # return {
    #     "average range": round(summary['mean'], 2),
    #     "log-normal 1 - CDF (average 右尾機率)": round(1 - dist.cdf(summary['mean']), 4),
    #     "empirical percentile (ECDF)": f"{round(percentileofscore(data, summary['median']), 2)}%",
    #     "Min": round(summary['min'], 2),
    #     "Q1": round(summary['q1'], 2),
    #     "Median": round(summary['median'], 2),
    #     "Q3": round(summary['q3'], 2),
    #     "Max": round(summary['max'], 2)
    # }


### Page_1 spx_range_analysis with spy volume


def create_volume_bins(df):
    df = df.dropna(subset=['High-Low', 'spy_vol']).copy()
    df['VolumeBin'] = pd.qcut(df['spy_vol'], q=3, labels=['Low', 'Medium', 'High'])
    _, bin_edges = pd.qcut(df['spy_vol'], q=3, retbins=True)
    bin_ranges = {
        'Low': f"{bin_edges[0]:,.0f} - {bin_edges[1]:,.0f}",
        'Medium': f"{bin_edges[1]:,.0f} - {bin_edges[2]:,.0f}",
        'High': f"{bin_edges[2]:,.0f} - {bin_edges[3]:,.0f}"
    }
    return df, bin_edges, bin_ranges

def compute_tail_and_table(data, dist, current_range, current_volume, bin_label, bin_edges):
    stats_labels = ['Min', 'Q1', 'Mean', 'Median', 'Q3', 'Max']
    last10 = data.iloc[-10:]
    stats_values = [
        last10.min(),
        np.percentile(last10, 25),
        last10.mean(),
        np.median(last10),
        np.percentile(last10, 75),
        last10.max()
    ]
    cdf_1_minus = [1 - dist.cdf(v) for v in stats_values]
    ecdf_vals = [1 - percentileofscore(data, v, kind='weak') / 100 for v in stats_values]

    table = pd.DataFrame({
        "Right tail Statistic": stats_labels,
        "Value": np.round(stats_values, 2),
        "1 - CDF (Right Tail Prob)": np.round(cdf_1_minus, 4),
        "1 - ECDF (Percentile)": [f"{int(p * 100)}%" for p in ecdf_vals],
        "Volume Bin": bin_label
    })
    divider = pd.DataFrame({k: ['-' * len(k)] for k in table.columns})
    return table, divider

def spx_range_pdf_cdf_by_volume_bin(df, current_range=None, current_volume=None):
    df, bin_edges, bin_ranges = create_volume_bins(df)
    x = np.linspace(df['High-Low'].min(), df['High-Low'].max(), 1000)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    tail_prob = None
    detected_bin = None
    stats_tables = []

    for label, group in df.groupby('VolumeBin'):
        data = group['High-Low']
        # mu = np.log(data.mean()) - 0.5 * np.log(1 + (data.std(ddof=1) / data.mean()) ** 2)
        # sigma = np.sqrt(np.log(1 + (data.std(ddof=1) / data.mean()) ** 2))
        # dist = lognorm(s=sigma, scale=np.exp(mu))
        log_data = np.log(data)
        mu = np.mean(log_data)
        sigma = np.std(log_data)
        dist = lognorm(s=sigma, scale=np.exp(mu))

        ax1.plot(x, dist.pdf(x), label=f'PDF | Volume={label}')
        ax2.plot(x, dist.cdf(x), linestyle='--', label=f'CDF | Volume={label}')

        if current_range is not None and current_volume is not None:
            if (
                (label == 'Low' and current_volume <= bin_edges[1]) or
                (label == 'Medium' and bin_edges[1] < current_volume <= bin_edges[2]) or
                (label == 'High' and current_volume > bin_edges[2])
            ):
                tail_prob = 1 - dist.cdf(current_range)
                detected_bin = label
                ax1.axvline(current_range, color='red', linestyle='--', label=f'Now: {current_range:.1f} pts')
                ax2.axhline(1 - tail_prob, color='red', linestyle='--')
                ax2.scatter([current_range], [1 - tail_prob], color='red')
                ax2.text(current_range + 1, 1 - tail_prob, f'{tail_prob:.2%}\nVol={label}', color='red')

        table, divider = compute_tail_and_table(data, dist, current_range, current_volume, label, bin_edges)
        stats_tables += [table, divider]

    ax1.set_ylabel('PDF', color='blue')
    ax2.set_ylabel('CDF', color='green')
    ax1.set_xlabel('SPX High - Low Range (points)')
    ax1.set_title('SPX Intraday Range - PDF + CDF by Volume Bin (past 2 months)')
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    ax1.grid(True)
    plt.tight_layout()
    plt.show()

    stats_df = pd.concat(stats_tables, ignore_index=True)
    # print(stats_df.to_markdown(index=False))

    result = {
        "Volume Bin Ranges": bin_ranges,
        "Detected Volume Bin": detected_bin,
        "Current Volume": current_volume,
        "Current Range": current_range,
        "Tail Probability (1 - CDF)": round(tail_prob, 4) if tail_prob is not None else None
    }

    return result, stats_df, fig



##  SPX_VIX range box plot

def spx_vix_range(start_date=datetime(datetime.now().year, 1, 1),end_date=datetime.now(), range_points=0):
    # 設定時間範圍
    start_date = start_date
    end_date = end_date
    
    # 下載 VIX 和 SPX 的歷史資料
    vix = yf.download('^VIX', start=start_date, end=end_date)
    spx = yf.download('^GSPC', start=start_date, end=end_date)
    
    # 計算 SPX 的日內震幅（最高價 - 最低價）
    spx['Intraday_Range'] = spx['High'] - spx['Low']
    
    # 合併 VIX 和 SPX 的資料，並確保索引對齊
    data = pd.DataFrame({
        'VIX_Close': vix[('Close', '^VIX')],
        'SPX_Range': spx['Intraday_Range']
    }).dropna()
    
    # 定義 VIX 的分組區間
    bins = [0, 15, 20, 25, 30, 40, 100]
    labels = ['<15', '15–20', '20–25', '25–30', '30–40', '>40']
    data['VIX_Level'] = pd.cut(data['VIX_Close'], bins=bins, labels=labels)

    # 計算各區間總天數與震幅 > 80 點的天數
    rang = range_points
    group_total = data.groupby('VIX_Level',observed=False).size()
    group_above_80 = data[data['SPX_Range'] > rang].groupby('VIX_Level',observed=False).size()
    # 分組計算 SPX 日內震幅的統計數據
    group_stats = data.groupby('VIX_Level', observed=False)['SPX_Range'].describe()
    # print(group_stats)

    # 組合成結果表格
    prob_table = pd.DataFrame({
        'Total Days': group_total,
        f'Days > {rang} pts': group_above_80,
    })
    prob_table[f'Probability (>{rang} pts)'] = (prob_table[f'Days > {rang} pts'] / prob_table['Total Days']).fillna(0).round(3)
    # print(prob_table)
        
    # 繪製箱型圖以視覺化不同 VIX 水平下的 SPX 日內震幅分布
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x='VIX_Level', y='SPX_Range', hue='VIX_Level', data=data, palette='Set2', legend=False)
    plt.title(f'SPX intraday range vs differnet VIX level distribution since {start_date}')
    plt.xlabel('VIX level')
    plt.ylabel('SPX intraday range (points)')
    plt.grid(True)
    return group_stats, prob_table, fig




##

def spx_range_in_period(period,interval):
    
    vix = yf.download('^VIX', period=period, interval=interval)
    spy = yf.download('SPY', period=period, interval=interval)
    spx = yf.download('^GSPC', period=period, interval=interval)
    
    # 整理資料
    spx['Range'] = spx['High'] - spx['Low']
    df = pd.DataFrame({
        'VIX': vix[('Close', '^VIX')],
        'SPY_Volume': spy[('Volume', 'SPY')],
        'SPX_Range': spx['Range']
    }).dropna()

    
    # 計算相關係數
    vix_spy_vol_corr, _ = pearsonr(df['VIX'], df['SPY_Volume'])
    vix_spx_range_corr, _ = pearsonr(df['VIX'], df['SPX_Range'])
    spy_vol_spx_range_corr, _ = pearsonr(df['SPY_Volume'], df['SPX_Range'])
    # 計算三分位
    vix_tertiles = df['VIX'].quantile([0, 1/3, 2/3, 1])
    vol_tertiles = df['SPY_Volume'].quantile([0, 1/3, 2/3, 1])
    
    # 定義分組標籤含實際數值範圍
    vix_labels = [
        f"Low VIX\n(≤ {vix_tertiles[1/3]:.1f})",
        f"Mid VIX\n({vix_tertiles[1/3]:.1f} ~ {vix_tertiles[2/3]:.1f})",
        f"High VIX\n(> {vix_tertiles[2/3]:.1f})"
    ]
    
    vol_labels = [
        f"Low Vol\n(≤ {vol_tertiles[1/3]:.0f})",
        f"Mid Vol\n({vol_tertiles[1/3]:.0f} ~ {vol_tertiles[2/3]:.0f})",
        f"High Vol\n(> {vol_tertiles[2/3]:.0f})"
    ]
    
    
    # 單一圖表繪製：hist + KDE + 統計框 + ±1 std shading
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(df['SPX_Range'], bins=50, kde=True, color='steelblue', edgecolor='black')
    
    # 計算統計數據
    x = df['SPX_Range']
    mean = x.mean()
    std = x.std()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    p95 = x.quantile(0.95)
    
    # ±1 std shading
    plt.axvspan(mean - std, mean + std, color='orange', alpha=0.2, label='±1 std')
    
    # 95th percentile 虛線
    plt.axvline(p95, color='red', linestyle='--', label='95th percentile')
    
    # 統計文字框
    textstr = f"mean={mean:.1f}\nstd={std:.1f}\nQ1={q1:.1f}\nQ3={q3:.1f}\n95%={p95:.1f}"
    plt.text(0.97, 0.95, textstr, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             fontsize=9, bbox=dict(facecolor='white', alpha=0.6))
    
    # 標題與格式
    plt.title("SPX Intraday Range Distribution\n(with KDE and ±1 Std / 95th)", fontsize=14)
    plt.xlabel("SPX Range (High - Low)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    
    
    
    # 分組bb
    df['VIX_Group'] = pd.qcut(df['VIX'], 3, labels=vix_labels)
    # df['Volume_Group'] = pd.qcut(df['SPY_Volume'], 3, labels=vol_labels)
    
    # FacetGrid with KDE and mean ± std bands
    fig_vix = sns.FacetGrid(df, col='VIX_Group', margin_titles=True, sharex=False, sharey=False)
    fig_vix.map_dataframe(sns.histplot, x='SPX_Range', kde=True, bins=20)
    
    # 加上 mean ± std 範圍與統計值
    for ax, (key, subdf) in zip(fig_vix.axes.flat, df.groupby(['VIX_Group'])):
        subset = subdf['SPX_Range']
        m = subset.mean()
        s = subset.std()
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        
        # Shaded mean ± std band
        ax.axvspan(m - s, m + s, color='orange', alpha=0.2, label='±1 std')
        
        stats = f"mean={m:.1f}\nstd={s:.1f}\nQ1={q1:.1f}\nQ3={q3:.1f}"
        ax.text(0.85, 0.95, stats, ha='right', va='top', transform=ax.transAxes,
                fontsize=8, bbox=dict(facecolor='white', alpha=0.6))
    
    plt.suptitle('SPX Range by VIX \n(With ±1 Std Highlight)', y=1.03)
    plt.tight_layout()

    df['Volume_Group'] = pd.qcut(df['SPY_Volume'], 3, labels=vol_labels)
    # FacetGrid with KDE and mean ± std bands
    fig_vol = sns.FacetGrid(df, col='Volume_Group', margin_titles=True, sharex=False, sharey=False)
    fig_vol.map_dataframe(sns.histplot, x='SPX_Range', kde=True, bins=20)
    
    # 加上 mean ± std 範圍與統計值
    for ax, (key, subdf) in zip(fig_vol.axes.flat, df.groupby(['Volume_Group'])):
        subset = subdf['SPX_Range']
        m = subset.mean()
        s = subset.std()
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        
        # Shaded mean ± std band
        ax.axvspan(m - s, m + s, color='orange', alpha=0.2, label='±1 std')
        
        stats = f"mean={m:.1f}\nstd={s:.1f}\nQ1={q1:.1f}\nQ3={q3:.1f}"
        ax.text(0.85, 0.95, stats, ha='right', va='top', transform=ax.transAxes,
                fontsize=8, bbox=dict(facecolor='white', alpha=0.6))
    
    plt.suptitle('SPX Range SPY Volume\n(With ±1 Std Highlight)', y=1.03)
    plt.tight_layout()
    
    # df['VIX_Group'] = pd.qcut(df['VIX'], 3, labels=vix_labels)
    # df['Volume_Group'] = pd.qcut(df['SPY_Volume'], 3, labels=vol_labels)
    
    # FacetGrid with KDE and mean ± std bands
    fig_vix_vol = sns.FacetGrid(df, row='VIX_Group', col='Volume_Group', margin_titles=True, sharex=False, sharey=False)
    fig_vix_vol.map_dataframe(sns.histplot, x='SPX_Range', kde=True, bins=20)
    
    # 加上 mean ± std 範圍與統計值
    for ax, (key, subdf) in zip(fig_vix_vol.axes.flat, df.groupby(['VIX_Group', 'Volume_Group'])):
        subset = subdf['SPX_Range']
        m = subset.mean()
        s = subset.std()
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        
        # Shaded mean ± std band
        ax.axvspan(m - s, m + s, color='orange', alpha=0.2, label='±1 std')
        
        stats = f"mean={m:.1f}\nstd={s:.1f}\nQ1={q1:.1f}\nQ3={q3:.1f}"
        ax.text(0.85, 0.95, stats, ha='right', va='top', transform=ax.transAxes,
                fontsize=8, bbox=dict(facecolor='white', alpha=0.6))
    
    plt.suptitle('SPX Range by VIX and SPY Volume\n(With ±1 Std Highlight)', y=1.03)
    plt.tight_layout()
            
    return fig, fig_vix, fig_vol, fig_vix_vol


