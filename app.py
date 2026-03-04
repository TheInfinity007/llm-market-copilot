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
    
def _render_metrics(artifacts: dict):
    cols = st.columns(3)
    
    price = artifacts.get("price")
    valuation = artifacts.get("valuation")
    risk = artifacts.get("risk")
    headlines = artifacts.get("headlines")
    
    with cols[0]:
        st.subheader("Price window")
        if isinstance(price, dict) and "error" not in price:
            st.metric("Total return", f"{price.get('total_return', 0.0) * 100:.2f}%")
            st.caption(f"{price.get('start_date')} to {price.get('end_date')} . N={price.get('n')}")
            st.write(
                pd.DataFrame([price]).rename(
                    columns={
                        "first_close": "first_close",
                        "last_close": "last_close",
                        "total_return": "total_return (decimal)",
                    }
                ).T
            )
        elif isinstance(price, dict) and "error" in price:
            st.warning(price["error"])
        else:
            st.info("No price tool output (not requested or tool not used).")
        
    with cols[1]:
        st.subheader("Fundamentals")
        if isinstance(valuation, dict) and "error" not in valuation:
            df = pd.DataFrame([valuation])
            keep = ["ticker", "name", "sector", "market_cap", "pe", "pb", "beta", "dividend_yield", "profit_margin"]
            keep = [c for c in keep if c in df.columns]
            st.write(df[keep].T)
        elif isinstance(valuation, dict) and "error" in valuation:
            st.warning(valuation["error"])
        else:
            st.info("No fundamentals tool output (not requested or tool not used).")
            
    with cols[2]:
        st.subheader("Risk")
        if isinstance(risk, dict) and "error" not in risk:
            st.metric("Volatility (ann.)", f"{risk.get('volatility_ann', 0.0) * 100:.2f}%")
            st.metric("Max drawdown", f"{risk.get('max_drawdown', 0.0) * 100:.2f}%")    
            st.caption(f"{risk.get('start_date')} to {risk.get('end_date')} . N={risk.get('n')}")
            st.write(pd.DataFrame([risk]).T)
        elif isinstance(risk, dict) and "error" in risk:
            st.warning(risk["error"])
        else:
            st.info("No risk tool output (not requested or tool not used).")
            
    st.subheader("Headlines")
    if isinstance(headlines, list) and len(headlines) > 0:
        for h in headlines:
            title = h.get("title", "Untitled")
            link = h.get("link")
            src = h.get("source")
            dt = h.get("date")
            line = f"- {title}"
            if src:
                line += f" ({src})"
            if dt:
                line += f" . {dt}"
            if link:
                st.markdown(f"{line} \n {link}")
            else:
                st.markdown(line)
                
    else:
        st.info("No headlines tool output (not requested or tool not used).")

if run_btn:
    with st.spinner("Running tools and generating brief..."):
        brief_md, artifacts = run_query(
            query=query,
            default_ticker=default_ticker,
            default_n_days=default_n_days,
            force_fundamentals=include_fundamentals,
            force_risk=include_risk,
            force_news=include_news,
            news_limit=news_limit    
        )
    
    # artifacts = {
    #         "price": { "ticker": "AAPL.US", "total_return": 0.0299, "start_date": "2025-11-07", "end_date": "2026-02-04", "n": 60, "first_close": 268.4, "last_close": 276.4 },
    #         "headlines": [
    #             {"title": "Apple releases new iPhone model", "link": "https://example.com/apple-iphone", "source": "TechCrunch", "date": "2026-02-03"},
    #             {"title": "Apple's quarterly earnings beat expectations", "link": "https://example.com/apple-earnings", "source": "Bloomberg", "date": "2026-01-30"},
    #             {"title": "Apple faces supply chain issues in Asia", "link": None, "source": "Reuters", "date": "2026-01-25"}
    #         ],
    #         "valuation": {"ticker": "AAPL.US", "name": "Apple Inc.", "sector": "Technology", "market_cap": 2.5e12, "pe": 28.5, "pb": 7.5, "beta": 1.2, "dividend_yield": 0.006, "profit_margin": 0.25},
    #         "risk": {"volatility_ann": 0.2, "max_drawdown": 0.15, "start_date": "2025-11-07", "end_date": "2026-02-04", "n": 60}
    #     }
    # brief_md = "#### Snapshots\n- Price up 3% over last 60 days\n- PE of 28.5, PB of 7.5, market cap of $2.5T\n\n#### Metrics\n- Annualized volatility of 20%, max drawdown of 15%\n\n#### What it might mean\nThe modest price increase despite strong earnings and new product release could indicate cautious investor sentiment, possibly due to supply chain concerns and broader market volatility.\n\n#### Caveats\n- The return is relatively small and may not be statistically significant.\n- Fundamentals look solid but the high valuation metrics suggest expectations are already priced in.\n- Risk metrics indicate elevated volatility, which could lead to larger swings in either direction."
        
    left, right = st.columns([1.2,1])
    
    with left:
        st.subheader("Market brief")
        st.markdown(brief_md)
    
    with right:
        st.subheader("Tool-backed metrics")
        _render_metrics(artifacts = artifacts)
        
else:
    st.info("Set inputs on the left and click **Generate brief**.")