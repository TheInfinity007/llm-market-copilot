# Project source: https://www.freecodecamp.org/news/build-an-llm-market-copilot-with-langchain

## copilot.py – the engine
This file holds everything that actually makes the copilot work:

* The EODHD data tools (prices, fundamentals, news, risk)
* The agent setup and prompt rules
* A single run_brief() function that takes inputs and returns:
* the markdown brief
* the structured artifacts for the UI

If you want to reuse this copilot anywhere else later, this is the file you keep.

## app.py – the MVP shell

This is just the Streamlit layer:

* Sidebar inputs (ticker, window, query, optional parameters)
* A two-pane layout: left side shows the brief, right side shows tool-backed metrics and headlines

No data logic lives here. It only calls run_brief() and renders what comes back.

### Why this split matters
If everything is mixed into one Streamlit script, you’re stuck with Streamlit forever.

With this split, you can replace Streamlit with FastAPI later without rewriting the core logic. You also keep “product logic” in one place, which makes testing and iteration much easier. And you avoid the notebook trap where UI code and data code become impossible to maintain.