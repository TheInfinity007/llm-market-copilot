## Get EOD ticker prices

##### Request
```
curl --location 'https://eodhd.com/api/eod/AAPL.US?from=2026-01-01&to=2026-02-01&api_token=<api-key>&fmt=json'
```

##### Response
```json
[
    {
        "date": "2026-01-02",
        "open": 272.26,
        "high": 277.84,
        "low": 269,
        "close": 271.01,
        "adjusted_close": 270.7566,
        "volume": 37838100
    },
    {
        "date": "2026-01-05",
        "open": 270.64,
        "high": 271.51,
        "low": 266.14,
        "close": 267.26,
        "adjusted_close": 267.0102,
        "volume": 45647200
    },
    {
        "date": "2026-01-06",
        "open": 267,
        "high": 267.55,
        "low": 262.12,
        "close": 262.36,
        "adjusted_close": 262.1147,
        "volume": 52352100
    }
]
```

## Get Fundamentals

##### Request
```
curl --location 'https://eodhd.com/api/fundamentals/AAPL.US?api_token=<api-key>&fmt=json'
```

##### Response
```json
{
    "General": {
        "Name": "Apple Inc.",
        "Country": "US",
        "Currency": "USD",
        "Exchange": "NASDAQ",
        "Industry": "Consumer Electronics",
        "Sector": "Technology"
    },
    "Valuation": {
        "MarketCapitalization": 2250000000000,
        "EnterpriseValue": 2300000000000,
        "TrailingPE": 28.5,
        "ForwardPE": 25.3,
        "PriceToSalesRatioTTM": 7.8,
        "PriceToBookRatioTTM": 30.2
    },
    "Highlights": {
        "52WeekHigh": 300.00,
        "52WeekLow": 250.00,
        "DividendYield": 0.006,
        "EPS": 5.20
    },
    "Technical": {
        "Beta": 1.2,
        "Vol20D": 35000000,
        "Vol100D": 40000000,
        "20DayAvgVolume": 36000000,
        "100DayAvgVolume": 38000000
    }
}
```

## Get News
##### Request
```
curl --location 'https://eodhd.com/api/news/AAPL.US?from=2026-01-01&to=2026-02-01&api_token=<api-key>&fmt=json'
``` 

##### Response
```json
[
    {
        "date": "2026-01-15",
        "headline": "Apple Unveils New iPhone Model with Revolutionary Features",
        "source": "TechCrunch",
        "url": "https://techcrunch.com/apple-new-iphone-2026"
    },
    {
        "date": "2026-01-20",
        "headline": "Apple's Stock Hits All-Time High Amid Strong Earnings Report",
        "source": "Bloomberg",
        "url": "https://bloomberg.com/apple-stock-high-2026"
    },
    {
        "date": "2026-01-25",
        "headline": "Apple Expands Services Business with New Subscription Offerings",
        "source": "The Verge",
        "url": "https://theverge.com/apple-services-expansion-2026"
    }
]
``` 

## Get Risk Metrics
##### Request
```
curl --location 'https://eodhd.com/api/risk/AAPL.US?from=2026-01-01&to=2026-02-01&api_token=<api-key>&fmt=json'
```

##### Response
```json
{
    "date": "2026-01-31",
    "beta": 1.2,
    "vol20D": 35000000,
    "vol100D": 40000000,
    "20DayAvgVolume": 36000000,
    "100DayAvgVolume": 38000000
}
```

