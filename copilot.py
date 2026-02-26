from dotenv import load_dotenv
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

eodhd_api_key = os.environ.get('EODHD_API_KEY', 'def')
openai_api_key = os.environ.get('OPENAI_API_KEY', '')

print(f"EODHD_API_KEY: {eodhd_api_key}")
print(f"OPENAI_API_KEY: {openai_api_key}")

# Helper functions

# To fix the user input.
def normalize_ticker(ticker: str) -> str:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return ticker
    if "." in ticker:   # ticker already contains the exchange suffix
        return ticker
    return f"{ticker}.US"   # default exchange suffix for US stocks
    
def _safe_json_loads(x: Any) -> Optional[Any]:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        return None
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return None

def get_eod_prices_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://eodhd.com/api/eod/{ticker}"
    params = {
        "from": start,
        "to": end,
        "api_token": eodhd_api_key,
        "fmt": "json"
    }
    r = requests.get(url, params=params)
    data = r.json()

    if not isinstance(data, list) or not data:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "ticker"])
    
    df = pd.DataFrame(data)
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df

# Data Tools


# Price window
@tool
def last_n_days_prices(ticker: str, n: int = 60) -> Dict[str, Any]:
    """
    Quick return window over last N trading days.
    Returns a compact summary. No raw rows.
    """
    
    ticker = normalize_ticker(ticker)
    
    end = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow().date() - timedelta(days=240)).isoformat()
    
    df = get_eod_prices_raw(ticker, start, end)
    if df.empty:
        return {"ticker": ticker, "error": "no_price_data"}
    
    df = df.tail(int(n)).reset_index(drop=True)
    if df.empty:
        return {"ticker": ticker, "error": "no_price_data"}
    
    first_close = float(df.loc[0, "close"])
    last_close = float(df.loc[len(df) - 1, "close"])
    total_return = float((last_close / first_close) - 1.0)
    
    return {
        "ticker": ticker,
        "n": int(n),
        "start_date": str(df.loc[0, "date"].date()),
        "end_date": str(df.loc[len(df)-1, "date"].date()),
        "first_close": first_close,
        "last_close": last_close,
        "total_return": total_return
    }
    
@tool
def fundamentals_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Lightweight fundamentals snapshot.
    Returns a flat dict
    """
    ticker = normalize_ticker(ticker)
    
    url = f"https://eodhd.com/api/fundamentals/{ticker}"
    params = {
        "api_token": eodhd_api_key,
        "fmt": "json"
    }
    r = requests.get(url, params=params)
    data = r.json()
    
    if not isinstance(data, dict) or not data:
        return {"ticker": ticker, "error": "no_data" }
    
    highlights = data.get("Highlights", {}) or {}
    general = data.get("General", {}) or {}
    valuation = data.get("Valuation", {}) or {}
    technicals = data.get("Technicals", {}) or {}
    
    return {
        "ticker": ticker,
        "name": general.get("Name"),
        "sector": general.get("Sector"),
        "industry": general.get("Industry"),
        "market_cap": highlights.get("MarketCapitalization"),
        "pe": valuation.get("TrailingPE"),
        "pb": valuation.get("PriceBookMQ"),
        "profit_margin": highlights.get("ProfitMargin"),
        "dividend_yield": highlights.get("DividendYield"),
        "beta": technicals.get("Beta"),
    }
    
@tool
def latest_news(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Latest headlines for a ticker.
    Returns a compact list of dicts.
    """
    
    ticker = normalize_ticker(ticker)
    
    url = f"https://eodhd.com/api/news"
    params = {
        "s": ticker,
        "limit": int(limit),
        "offset": 0,
        "api_token": eodhd_api_key,
        "fmt": "json"
    }
    r = requests.get(url, params=params)
    data = r.json()
    
    if not isinstance(data, list) or not data:
        return []
    
    df = pd.DataFrame(data)
    keep = [c for c in ["date", "link", "source", "title"] if c in df.columns]
    df = df[keep].copy()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df = df.sort_values("date", ascending=False)    
        
    out = df.head(int(limit)).reset_index(drop=True).to_dict(orient="records")
    for row in out:
        dt = row.get("date")
        if isinstance(dt, (pd.Timestamp, datetime)):
            row["date"] = dt.isoformat()
    return out

@tool
def risk_metrics(ticker: str, start: str, end: str) -> Dict[str, Any]:
    """
    Risk metrics from daily close prices over a window.
    volatility_ann: annualized vol from daily returns
    max_drawdown: max drawdown over the window
    """
    
    ticker = normalize_ticker(ticker)
    
    df = get_eod_prices_raw(ticker, start, end)
    if df.empty:
        return {"ticker": ticker, "error": "no_price_data"}
    
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = df["close"].pct_change().fillna(0.0)
    
    vol_ann = float(df["ret"].std(ddof=0) * np.sqrt(252))
    
    cummax = df["close"].cummax()
    dd = (df["close"] / cummax) - 1.0
    max_dd = float(dd.min())
    
    first_close = float(df.loc[0, "close"])
    last_close = float(df.loc[len(df) - 1, "close"])
    total_return = float((last_close / first_close) - 1.0)
    
    return {
        "ticker": ticker,
        "start_date": str(df.loc[0, "date"].date()),
        "end_date": str(df.loc[len(df)-1, "date"].date()),
        "n": int(len(df)),
        "total_return": total_return,
        "volatility_ann": vol_ann,
        "max_drawdown": max_dd,
    }
    
@tool
def eod_prices(ticker: str, start: str, end: str) -> List[Dict[str, Any]]:
    """
    Raw OHLCV rows. Use only for custom calculations that cannot be done with other tools.
    """
    
    ticker = normalize_ticker(ticker)
    
    df = get_eod_prices_raw(ticker, start, end)
    return json.loads(df.to_json(orient="records"))


# -----------------------------------------------------------------------
# Creating an agent
# Define how the agent should behave and give it the tools it's allowed to use.
# -----------------------------------------------------------------------

system_prompt = (
    "You are a market brief copilot embedded in a product.\n"
    "Rules:\n"
    "1) Use tools for facts. Never invent numbers.\n"
    "2) Do not dump raw prices rows or long news lists. \n"
    "3) If the user didn't ask for something, don't compute it. \n"
    "4) Output in clean Markdown with sections\n"
    "5) Keep it short and useful, like an internal dashboard note.\n"
    "Tool guidance:\n"
    "- Use last_n_days_prices for return windows.\n"
    "- Use fundamentals_snapshot for PE/PB/market cap/sector/beta.\n"
    "- Use latest_news for headlines.\n"
    "- Use risk_metrics only if asked for volatility or drawdown.\n"
    "- Use eod_prices only if you can't do what the user asks with the other tools.\n"
)

def _build_agent() -> Any:
    llm = ChatOpenAI(
        model="gpt-5-nano",
        temperature=0.0,
        api_key=openai_api_key,
    )
    
    tools = [last_n_days_prices, fundamentals_snapshot, latest_news, risk_metrics, eod_prices]
    return create_react_agent(model=llm, tools=tools)

AGENT = _build_agent()

def _extract_artifacts(messages: List[Any]) -> Dict[str, Any]:
    """
    Pull tool outputs from the LangGraph message list.
    This avoids calling the endpoints twice in Streamlit.
    """
    
    out: Dict[str, Any] = {}
    for msg in messages:
        name = getattr(msg, "name", None)
        content = getattr(msg, "content", None)
        
        if not name:
            continue
        
        payload = _safe_json_loads(content)
        if payload is None:
            continue
        
        if name.endsWith("last_n_days_prices"):
            out["price"] = payload
        elif name.endsWith("fundamentals_snapshot"):
            out["valuation"] = payload
        elif name.endsWith("risk_metrics"):
            out["risk"] = payload
        elif name.endsWith("latest_news"):
            out["headlines"] = payload
        
    return out
    
    

# -----------------------------------------------------------------------
# Tuning the agent into a callable backend
# Up to now, we’ve built tools and an agent. This is the piece that turns it into something your app can call like a regular backend function. One input in, one brief out, plus the structured data you need to render the UI.
# -----------------------------------------------------------------------

def run_brief(
    ticker: str,
    n_days: int = 60,
    include_fundamentals: bool = True,
    include_risk: bool = False,
    include_news: bool = True,
    news_limit: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns:
     - markdown brief (string)
     - artifacts dict with keys like price/valuation/risk/headlines when tools were used
    """
    t = normalize_ticker(ticker)
    
    request_parts = [
        f"ticker={t}",
        f"Compute total return over the last {int(n_days)} trading days.",
    ]
    
    if include_fundamentals:
        request_parts.append("Fetch fundamentals and report PE, BE, market cap, sector, beta.")
    if include_risk:
        request_parts.append("Compute annualized volatility and max drawdown over the same window.")
        request_parts.append("Use the same start_date and end_date as the return window.")
    if include_news:
        request_parts.append(f"Pull {int(news_limit)} latest headlines and reference them briefly.")
    
    request_parts.append("Write a short market brief with sections: Snapshots, Metrics, What it might mean, Caveats.")
    request_parts.append("Keep it concise. Do not paste raw rows.")
    
    user_prompt = " ".join(request_parts)
    
    response = AGENT.invoke(
        { "messages": [("system", system_prompt), ("user", user_prompt)] }
    )
    
    messages = response.get("messages", [])
    final_msg = messages[-1]
    brief_md = getattr(final_msg, "content", "") or ""
    
    artifacts = _extract_artifacts(messages)
    return brief_md, artifacts