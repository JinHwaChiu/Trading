import streamlit as st
from utils import spx_range_report_v2  # your helper module

st.set_page_config(page_title="SPX Range Analysis", layout="wide")
st.title("SPX Intraday Range - PDF + CDF Analysis")

# 🔹 Sidebar input
st.sidebar.header("輸入參數")
symbol = st.sidebar.text_input("Ticker Symbol", value="^GSPC")
period = st.sidebar.selectbox("Period", ["5d", "1mo", "2mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

# 🔘 Run button
if st.sidebar.button("執行分析"):
    report, fig = spx_range_report_v2(symbol,period,interval)
    st.pyplot(fig)
    st.subheader("SPX Range Summary Table")
    st.dataframe(report, use_container_width=True)

    # # Additional metrics
    # right_tail_prob = 1 - lognorm_dist.cdf(mean_val)
    # ecdf_percentile = percentileofscore(range_data, median)

    # st.info(f"**Average Range:** {mean_val:.2f} pts")
    # st.info(f"**Log-normal 1 - CDF (Right Tail Probability at Mean):** {right_tail_prob:.4f}")
    # st.info(f"**Empirical Percentile of Median (ECDF):** {ecdf_percentile:.2f}%")
