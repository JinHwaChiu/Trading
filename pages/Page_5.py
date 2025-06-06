import streamlit as st
from utils import spx_range_report, spx_history, spx_range_pdf_cdf_by_volume_bin, spx_vix_range, spx_range_in_period
# your helper module

st.set_page_config(page_title="12 month SPX Range groupby VIX, SPY Vol")
st.title("12 month SPX Range groupby VIX, SPY Vol")

# You can reuse your logic here
st.write("This is the range analysis page.")

st.sidebar.header("輸入參數")
# symbol = st.sidebar.text_input("Ticker Symbol", value="^GSPC")
period = st.sidebar.selectbox("Period", ["5d", "1mo", "2mo", "3mo", "6mo", "12mo", "2y", "5y"], index=5)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

if st.sidebar.button("執行分析"):

    fig, fig_vix, fig_vol, fig_vix_vol = spx_range_in_period(period,interval)
    st.pyplot(fig)
    st.pyplot(fig_vix)
    st.pyplot(fig_vol)
    st.pyplot(fig_vix_vol)