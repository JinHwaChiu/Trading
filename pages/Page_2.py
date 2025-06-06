import streamlit as st
from utils import spx_range_report, spx_history, spx_range_pdf_cdf_by_volume_bin# your helper module
st.set_page_config(page_title="SPX Range Analysis", layout="wide")
st.title("past 10 days in 2 month dist: (it is more sensitive to the intraday price move recently)")

# You can reuse your logic here
st.write("This is the range analysis page.")

# 🔹 Sidebar input
st.sidebar.header("輸入參數")
symbol = st.sidebar.text_input("Ticker Symbol", value="^GSPC")
period = st.sidebar.selectbox("Period", ["5d", "1mo", "2mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
past_few_days = st.sidebar.number_input("past few days", min_value=10, step=1)
current_range = st.sidebar.number_input("當前 High-Low Range（點數）", min_value=0.0, step=1.0)
current_volume = st.sidebar.number_input("當前 SPY 成交量", min_value=0)

# 🔘 Run button
if st.sidebar.button("執行分析"):
    spx = spx_history(symbol,period=period, interval=interval)
    report, fig = spx_range_report(spx['High-Low'].dropna(),past_few_days)

    st.subheader("SPX Intraday Range Probability Stats")
    st.pyplot(fig)
    st.dataframe(report, use_container_width=True)

    # # Additional metrics
    # right_tail_prob = 1 - lognorm_dist.cdf(mean_val)
    # ecdf_percentile = percentileofscore(range_data, median)

    # st.info(f"**Average Range:** {mean_val:.2f} pts")
    # st.info(f"**Log-normal 1 - CDF (Right Tail Probability at Mean):** {right_tail_prob:.4f}")
    # st.info(f"**Empirical Percentile of Median (ECDF):** {ecdf_percentile:.2f}%")
    result, report, fig = spx_range_pdf_cdf_by_volume_bin(
        df=spx,               # 需包含 'High-Low' 與 'spy_vol' 欄位
        current_range=current_range,               # 當下intraday range
        current_volume=current_volume       # 當下 SPY 成交量
    )
    st.subheader("+SPY Volume")
    st.pyplot(fig)
    st.dataframe(report, use_container_width=True, height=800)
    for i in result:
        st.write(i,':',result[i])

    # st.write("📊 分析結果:")
    # for k, v in report.items():
    #     st.write(f"**{k}**: {v}")
