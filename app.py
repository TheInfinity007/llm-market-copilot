import streamlit as st
import pandas as pd
from copilot import run_query

st.set_page_config(page_title="Market Brief Copilot", layout="wide")
st.title("Market Brief Copilot")
st.caption("LangChain + EODHOD. Minimal internal-styled brief, with tool-backed metrics.")

with st.sidebar:
    st.header("Inputs")
    
    query = st.text_area("Query", value="For AAPL.US, compute total return over the last 60 trading days. Fetch PE and PB. Pull 5 latest headlines Brief  interpretation.")
    default_ticker = st.text_input("Default ticker (used only if query doesn't mention one)", value="AAPL.US")
    default_n_days = st.slider("Default trading days window (used only if query doesn't mention one)", min_value=20, max_value=180, value=60, step=5)
    
    st.divider()
    
    with st.sidebar.expander("Optional parameters (force include)"):
        include_fundamentals = st.checkbox("Fundamentals (PE, PB, etc.)", value=False)
        include_risk = st.checkbox("Risk metrics (volatility, drawdown)", value=False)
        include_news = st.checkbox("Headlines", value=False)
        news_limit = st.slider("Headlines count", min_value=3, max_value=10, value=5, step=1, disabled=not include_news)
        
    run_btn = st.button("Generate brief", type="primary")