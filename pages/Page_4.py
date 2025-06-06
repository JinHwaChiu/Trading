
import streamlit as st
from utils import spx_range_report, spx_history, spx_range_pdf_cdf_by_volume_bin, spx_vix_range# your helper module
import streamlit.components.v1 as components


# page = st.sidebar.radio("daytrade streamlit", ["Page 1", "Page 2", "Page 3", "Page 4"])
# if page == "Page_4":
# if st.sidebar.button("執行分析"):
st.set_page_config(page_title="", layout="wide")
st.subheader("SPX vs VIX from beginning of the year")
range_points = st.sidebar.number_input("intraday range", min_value=0.0, step=1.0)

table1, table2, fig = spx_vix_range(range_points=range_points)
st.pyplot(fig)
st.dataframe(table1, use_container_width=True)
st.sidebar.header("輸入參數")
if st.sidebar.button("執行分析"):
    st.subheader(f" Probability > {range_points}")
    
    st.dataframe(table2, use_container_width=True)
    
