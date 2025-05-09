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


def spx_range_report(data):
    dist, mu, sigma = fit_lognormal_distribution(data)
    summary = calculate_summary_stats(data[-10:])
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
