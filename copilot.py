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