# Save the final Streamlit app with "執行分析" button to a .py file
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, percentileofscore

st.set_page_config(page_title="SPX Range Analysis", layout="wide")
st.title("SPX Intraday Range - PDF + CDF Analysis")

# Button trigger
if st.button("執行分析"):

    # Download SPX data
    def spx_history(symbol, period="2mo", interval="1d"):
        df = yf.download(symbol, period=period, interval=interval)
        df["High-Low"] = df["High"] - df["Low"]
        return df

    # Fetch data
    spx = spx_history("^GSPC", period="2mo", interval="1d")
    range_data = spx["High-Low"].dropna()

    # Fit log-normal distribution
    mean_x = np.mean(range_data)
    std_x = np.std(range_data, ddof=1)
    sigma = np.sqrt(np.log(1 + (std_x / mean_x) ** 2))
    mu = np.log(mean_x) - 0.5 * sigma ** 2
    lognorm_dist = lognorm(s=sigma, scale=np.exp(mu))

    # Recent 10-day stats
    recent_range = spx["High-Low"].iloc[-10:]
    mean_val = recent_range.mean()
    min_val = recent_range.min()
    q1 = np.percentile(recent_range, 25)
    median = np.median(recent_range)
    q3 = np.percentile(recent_range, 75)
    max_val = recent_range.max()

    # Plot PDF and CDF
    x = np.linspace(range_data.min(), range_data.max(), 1000)
    pdf_vals = lognorm_dist.pdf(x)
    cdf_vals = lognorm_dist.cdf(x)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(x, pdf_vals, label="PDF (Density)", color="blue")
    ax1.set_ylabel("PDF", color="blue")

    for val, label, color in zip(
        [mean_val, min_val, q1, median, q3, max_val],
        ["Avg", "Min", "Q1", "Median", "Q3", "Max"],
        ["red", "gray", "purple", "orange", "blue", "black"]
    ):
        label_with_value = f"{label} = {val:.2f}"
        ax1.axvline(val, linestyle=":", color=color, label=label_with_value)

    ax2 = ax1.twinx()
    ax2.plot(x, cdf_vals, label="CDF (Cumulative)", color="green")
    ax2.set_ylabel("CDF", color="green")

    for val, label, color in zip(
        [mean_val, min_val, q1, median, q3, max_val],
        ["Avg", "Min", "Q1", "Median", "Q3", "Max"],
        ["red", "gray", "purple", "orange", "blue", "black"]
    ):
        prob = 1 - lognorm_dist.cdf(val)
        ax2.text(val + 1, 0.1, f"{label}\\n1-CDF={prob:.3f}", fontsize=9, color=color)

    ax1.set_xlabel("SPX High - Low Range (points)")
    ax1.set_title("SPX Intraday Range - PDF + CDF (past 2 months)")
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)
    ax1.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # Summary Table
    stats_labels = ["Min", "Q1", "Mean", "Median", "Q3", "Max"]
    stats_values = [min_val, q1, mean_val, median, q3, max_val]
    cdf_1_minus = [1 - lognorm_dist.cdf(v) for v in stats_values]
    ecdf_vals = [1 - percentileofscore(range_data, v) / 100 for v in stats_values]

    stats_df = pd.DataFrame({
        "Right tail Statistic": stats_labels,
        "Value": np.round(stats_values, 2),
        "1 - CDF (Right Tail Prob)": np.round(cdf_1_minus, 4),
        "1 - ECDF (Percentile)": [f"{int(p * 100)}%" for p in ecdf_vals]
    })

    st.subheader("SPX Range Summary Table")
    st.dataframe(stats_df, use_container_width=True)

    # Additional metrics
    right_tail_prob = 1 - lognorm_dist.cdf(mean_val)
    ecdf_percentile = percentileofscore(range_data, median)

    st.info(f"**Average Range:** {mean_val:.2f} pts")
    st.info(f"**Log-normal 1 - CDF (Right Tail Probability at Mean):** {right_tail_prob:.4f}")
    st.info(f"**Empirical Percentile of Median (ECDF):** {ecdf_percentile:.2f}%")

