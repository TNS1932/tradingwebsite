import logging
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import re
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

# ---------------- APP SETUP ----------------
app = FastAPI(title="Portfolio Tracker API")
logger = logging.getLogger("portfolio_app")
logging.basicConfig(level=logging.INFO)

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent

# Thread pool for concurrent API calls
executor = ThreadPoolExecutor(max_workers=10)

# Simple in-memory cache with timestamp
price_cache = {}
sector_cache = {}
CACHE_TTL = 300  # 5 minutes
info_cache = {}
CACHE_TIMEOUT = timedelta(minutes=5)  # Cache for 5 minutes

# Alpha Vantage API key (get free key at https://www.alphavantage.co/support/#api-key)
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "demo")  # Use 'demo' for testing
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"  # Enable demo data

# Demo prices for when APIs are down
DEMO_PRICES = {
    "AAPL": 245.50, "AMC": 7.25, "AMZN": 195.30, "BAC": 41.75, "BACHY": 52.30,
    "BND": 68.45, "BRK.B": 465.20, "CRWV": 15.80, "DIA": 425.60, "F": 11.95,
    "GFL": 42.10, "HOOD": 28.45, "JNJ": 156.80, "MSFT": 425.75, "NFLX": 785.90,
    "NVDA": 925.60, "SPY": 520.40, "T": 18.65, "TM": 215.30, "VOO": 485.20,
    "XOM": 115.40, "YUMC": 46.85
}

# Demo sectors for when APIs are down
DEMO_SECTORS = {
    "AAPL": "Technology", "AMC": "Entertainment", "AMZN": "Consumer Cyclical", 
    "BAC": "Financial Services", "BACHY": "Industrials", "BND": "Fixed Income",
    "BRK.B": "Financial Services", "CRWV": "Industrials", "DIA": "Index Fund",
    "F": "Consumer Cyclical", "GFL": "Industrials", "HOOD": "Financial Services",
    "JNJ": "Healthcare", "MSFT": "Technology", "NFLX": "Communication Services",
    "NVDA": "Technology", "SPY": "Index Fund", "T": "Communication Services",
    "TM": "Consumer Cyclical", "VOO": "Index Fund", "XOM": "Energy", "YUMC": "Consumer Cyclical"
}

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {type(exc).__name__}: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"{type(exc).__name__}: {str(exc)}",
            "path": str(request.url)
        }
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use /tmp for file storage on cloud platforms (Render, Vercel, etc.)
if os.getenv("RENDER") or os.getenv("VERCEL"):
    ACTIVE_PORTFOLIO_FILE = "/tmp/portfolio_data.csv"
else:
    ACTIVE_PORTFOLIO_FILE = "portfolio_data.csv"

# ---------------- CACHING & PARALLEL FETCH ----------------
def get_cached_price(symbol: str):
    """Get cached price or fetch from Alpha Vantage with demo fallback"""
    now = datetime.now()
    if symbol in price_cache:
        price, timestamp = price_cache[symbol]
        if now - timestamp < CACHE_TIMEOUT:
            return price
    
    if ALPHA_VANTAGE_AVAILABLE and ALPHA_VANTAGE_KEY != "demo":
        try:
            ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
            data, meta = ts.get_quote_endpoint(symbol=symbol)
            if not data.empty and '05. price' in data.columns:
                price = float(data['05. price'].iloc[0])
                price_cache[symbol] = (price, now)
                logger.info(f"Fetched price for {symbol} from Alpha Vantage: ${price}")
                return price
        except Exception as e:
            logger.error(f"Alpha Vantage failed for {symbol}: {e}")
    
    # Last resort: Use demo prices if available
    if symbol in DEMO_PRICES:
        logger.warning(f"Using DEMO price for {symbol}: ${DEMO_PRICES[symbol]}")
        return DEMO_PRICES[symbol]
    
    logger.error(f"All data sources failed for {symbol}")
    return None

def get_cached_info(symbol: str):
    """Get sector info from demo data"""
    now = datetime.now()
    if symbol in info_cache:
        info, timestamp = info_cache[symbol]
        if now - timestamp < CACHE_TIMEOUT:
            return info
    
    if symbol in DEMO_SECTORS:
        demo_info = {"sector": DEMO_SECTORS[symbol]}
        logger.warning(f"Using DEMO sector for {symbol}: {DEMO_SECTORS[symbol]}")
        return demo_info
    
    return {}

async def fetch_stock_data(symbol: str, shares: float, avg_cost: float):
    """Fetch stock price and sector data with timeout"""
    loop = asyncio.get_event_loop()
    
    try:
        # Run both price and info fetch concurrently with timeout
        price, info = await asyncio.wait_for(
            asyncio.gather(
                loop.run_in_executor(executor, get_cached_price, symbol),
                loop.run_in_executor(executor, get_cached_info, symbol),
                return_exceptions=True
            ),
            timeout=5.0  # 5 second timeout per stock
        )
        
        if isinstance(price, Exception) or price is None:
            logger.error(f"Failed to get price for {symbol}")
            return None
        if isinstance(info, Exception):
            info = {}
        
        sector = info.get("sector", "Unknown") if info else "Unknown"
        equity = shares * price
        pnl = (price - avg_cost) * shares
        cost_basis = shares * avg_cost
        
        return {
            "symbol": symbol,
            "shares": float(shares),
            "avg_cost": float(avg_cost),
            "current_price": float(price),
            "equity": float(equity),
            "pnl": float(pnl),
            "roi": float((price - avg_cost) / avg_cost * 100),
            "sector": sector,
            "allocation": 0,
            "cost_basis": cost_basis
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "csv_file": ACTIVE_PORTFOLIO_FILE,
        "csv_exists": os.path.exists(ACTIVE_PORTFOLIO_FILE),
        "platform": "render" if os.getenv("RENDER") else "local"
    }

# ---------------- UTILITIES ----------------
def parse_brokerage_csv(filepath: str) -> pd.DataFrame:
    """Parse the brokerage CSV format and extract stock purchases"""
    if not os.path.exists(filepath):
        logger.warning(f"Portfolio file not found: {filepath}")
        return pd.DataFrame(columns=["symbol", "shares", "price", "date", "trans_type"])
    
    try:
        # Read CSV - handle multiline fields properly
        df = pd.read_csv(filepath, on_bad_lines='skip', skipinitialspace=True)
        logger.info(f"CSV columns found: {df.columns.tolist()}")
        logger.info(f"Total rows read: {len(df)}")
        
        # If columns look wrong, try reading with different settings
        if 'Trans Code' not in df.columns:
            logger.info("Retrying CSV read with quoting=csv.QUOTE_ALL")
            import csv
            df = pd.read_csv(filepath, quoting=csv.QUOTE_ALL, on_bad_lines='skip')
            logger.info(f"Retry - CSV columns: {df.columns.tolist()}")
            
    except Exception as e:
        logger.error(f"Error reading CSV: {e}", exc_info=True)
        return pd.DataFrame(columns=["symbol", "shares", "price", "date", "trans_type"])
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Map different CSV column name formats to standardized names
    column_mapping = {
        'Quantity': 'Shares',
        'Price': 'Share_Price', 
        'Amount': 'Traded-Price'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Check for required columns
    required_cols = ['Trans Code', 'Instrument', 'Shares', 'Share_Price', 'Settle Date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"CSV missing required columns: {missing_cols}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns.tolist())}")
    
    # Filter for Buy and Sell transactions only
    df = df[df['Trans Code'].isin(['Buy', 'Sell'])].copy()
    logger.info(f"Buy/Sell transactions: {len(df)}")
    
    if len(df) == 0:
        logger.warning("No Buy/Sell transactions found in CSV")
        return pd.DataFrame(columns=["symbol", "shares", "price", "date", "trans_type"])
    
    # Extract symbol from Instrument column (letters only)
    df['symbol'] = df['Instrument'].astype(str).str.strip().str.upper()
    
    # Remove any rows where symbol is empty or contains numbers
    df = df[df['symbol'].str.match(r'^[A-Z.]+$', na=False)]
    
    # Extract quantity and price from the actual column names in CSV
    df['shares'] = pd.to_numeric(df['Shares'], errors='coerce')
    df['price'] = pd.to_numeric(df['Share_Price'].astype(str).str.replace('$', '').str.replace(',', '').str.replace('(', '-').str.replace(')', ''), errors='coerce')
    df['date'] = pd.to_datetime(df['Settle Date'], errors='coerce')
    df['trans_type'] = df['Trans Code']
    
    # For sells, make shares negative
    df.loc[df['trans_type'] == 'Sell', 'shares'] = -df.loc[df['trans_type'] == 'Sell', 'shares']
    
    # Clean up
    df = df[['date', 'symbol', 'shares', 'price', 'trans_type']].dropna()
    logger.info(f"Final parsed transactions: {len(df)}")
    
    return df

def get_all_transactions(filepath: str) -> pd.DataFrame:
    """Get all transactions including non-trade transactions"""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        df['date'] = pd.to_datetime(df['Settle Date'], errors='coerce')
        # Map new column names to old names for API compatibility
        if 'Traded-Price' in df.columns:
            df['Amount'] = df['Traded-Price']
        if 'Shares' in df.columns:
            df['Quantity'] = df['Shares']
        if 'Share_Price' in df.columns:
            df['Price'] = df['Share_Price']
        return df
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

def load_and_sanitize_portfolio() -> pd.DataFrame:
    if not os.path.exists(ACTIVE_PORTFOLIO_FILE):
        return pd.DataFrame(columns=["symbol", "shares", "price", "date", "trans_type"])
    return parse_brokerage_csv(ACTIVE_PORTFOLIO_FILE)

def get_current_holdings() -> pd.DataFrame:
    """Get current holdings by netting buys and sells"""
    portfolio = load_and_sanitize_portfolio()
    if portfolio.empty:
        return pd.DataFrame(columns=["symbol", "total_shares", "avg_cost"])
    
    holdings = []
    for symbol in portfolio['symbol'].unique():
        trades = portfolio[portfolio['symbol'] == symbol]
        
        # Calculate net shares (buys - sells)
        total_shares = trades['shares'].sum()
        
        if total_shares > 0:
            # Calculate weighted average cost basis (only from buys)
            buys = trades[trades['shares'] > 0]
            if len(buys) > 0:
                avg_cost = (buys['shares'] * buys['price']).sum() / buys['shares'].sum()
                holdings.append({
                    'symbol': symbol,
                    'total_shares': total_shares,
                    'avg_cost': avg_cost
                })
    
    return pd.DataFrame(holdings)

# CSV file should be placed at: portfolio_data.csv
# No upload UI needed - just place your CSV file in the root directory

# ---------------- SERVE HTML ----------------
@apphtml_path = BASE_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(content=f"Error: index.html not found at {html_path}", status_code=500)
    return FileResponse(html_path
def serve_html():
    """Serve the main HTML page with CSV uploader"""
    return FileResponse("index.html")

@app.get("/portfolio_data.csv")
def serve_csv():
    """Serve the portfolio CSV file for frontend parsing"""
    if not os.path.exists(ACTIVE_PORTFOLIO_FILE):
        raise HTTPException(status_code=404, detail="Portfolio CSV not found")
    return FileResponse(ACTIVE_PORTFOLIO_FILE, media_type="text/csv", filename="portfolio_data.csv")

# ---------------- TEMPORARY CSV UPLOAD (NOT SAVED) ----------------
@app.post("/upload/temp")
async def upload_temp_csv(file: UploadFile = File(...)):
    """Process uploaded CSV temporarily without saving - privacy-focused"""
    try:
        # Read CSV content into memory only
        contents = await file.read()
        csv_text = contents.decode('utf-8')
        
        # Parse CSV with pandas
        from io import StringIO
        df = pd.read_csv(StringIO(csv_text))
        
        # Store in session/memory temporarily (never written to disk)
        # Return analytics immediately
        return {
            "status": "success",
            "message": "CSV processed in memory only - not saved",
            "rows": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        logger.error(f"Error processing uploaded CSV: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

# ---------------- GET ALL SYMBOLS ----------------
@app.get("/symbols")
def get_symbols():
    portfolio = load_and_sanitize_portfolio()
    if portfolio.empty:
        return {"symbols": []}
    symbols = sorted(portfolio["symbol"].unique().tolist())
    return {"symbols": symbols}



# ---------------- PORTFOLIO SUMMARY ----------------
@app.get("/portfolio/{symbol}")
def portfolio_data(symbol: str):
    portfolio = load_and_sanitize_portfolio()
    symbol = symbol.strip().upper()
    trades = portfolio[portfolio["symbol"] == symbol]
    if trades.empty:
        return {"error": "symbol not found in portfolio"}

    total_shares = trades["shares"].sum()
    if total_shares == 0:
        return {"error": "symbol has zero shares"}

    avg_cost = (trades["shares"] * trades["price"]).sum() / total_shares
    
    # Get current price using our caching system
    price = get_cached_price(symbol)
    if price is None:
        raise HTTPException(status_code=503, detail="Price lookup failed")
    equity = total_shares * price
    pnl = (price - avg_cost) * total_shares
    roi = (price - avg_cost) / avg_cost * 100

    return {
        "symbol": symbol,
        "total_shares": int(total_shares),
        "avg_cost": round(avg_cost, 2),
        "current_price": round(price, 2),
        "equity": round(equity, 2),
        "pnl": round(pnl, 2),
        "roi_percent": round(roi, 2)
    }

# ---------------- PORTFOLIO ANALYTICS ----------------
@app.get("/portfolio_total")
def portfolio_total(range: str = "1y"):
    portfolio = load_and_sanitize_portfolio()
    if portfolio.empty:
        return []

    # Total portfolio historical data disabled - Yahoo Finance removed
    return {"error": "Historical portfolio data temporarily unavailable. Please use current portfolio overview endpoint."}

# ---------------- PORTFOLIO ANALYTICS ----------------
@app.get("/analytics/overview")
async def portfolio_overview():
    """Get complete portfolio analytics including total value, P&L, allocation"""
    holdings = get_current_holdings()
    if holdings.empty:
        return {"error": "No portfolio data"}
    
    # Fetch all stock data in parallel
    tasks = [
        fetch_stock_data(row['symbol'], row['total_shares'], row['avg_cost'])
        for _, row in holdings.iterrows()
    ]
    
    results = await asyncio.gather(*tasks)
    holdings_list = [r for r in results if r is not None]
    
    if not holdings_list:
        return {"error": "Could not fetch stock data"}
    
    # Calculate totals
    total_equity = sum(h["equity"] for h in holdings_list)
    total_cost = sum(h["cost_basis"] for h in holdings_list)
    
    # Calculate allocations
    for holding in holdings_list:
        holding["allocation"] = (holding["equity"] / total_equity * 100) if total_equity > 0 else 0
        del holding["cost_basis"]  # Remove temporary field
    
    # Sort by equity descending
    holdings_list.sort(key=lambda x: x["equity"], reverse=True)
    
    total_pnl = total_equity - total_cost
    total_roi = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    return {
        "total_equity": round(total_equity, 2),
        "total_cost": round(total_cost, 2),
        "total_pnl": round(total_pnl, 2),
        "total_roi": round(total_roi, 2),
        "holdings": holdings_list
    }

@app.get("/analytics/sectors")
async def sector_allocation():
    """Get portfolio allocation by sector"""
    holdings = get_current_holdings()
    if holdings.empty:
        return {"sectors": []}
    
    # Fetch all stock data in parallel
    tasks = [
        fetch_stock_data(row['symbol'], row['total_shares'], row['avg_cost'])
        for _, row in holdings.iterrows()
    ]
    
    results = await asyncio.gather(*tasks)
    holdings_list = [r for r in results if r is not None]
    
    # Aggregate by sector
    sector_data = {}
    for holding in holdings_list:
        sector = holding["sector"]
        equity = holding["equity"]
        if sector in sector_data:
            sector_data[sector] += equity
        else:
            sector_data[sector] = equity
    
    total = sum(sector_data.values())
    sectors = [
        {
            "sector": sector,
            "value": round(value, 2),
            "percentage": round(value / total * 100, 2) if total > 0 else 0
        }
        for sector, value in sector_data.items()
    ]
    
    sectors.sort(key=lambda x: x["value"], reverse=True)
    return {"sectors": sectors}



@app.get("/analytics/best_worst")
async def best_worst_performers():
    """Get best and worst performing stocks"""
    holdings = get_current_holdings()
    if holdings.empty:
        return {"best": [], "worst": []}
    
    # Fetch all stock data in parallel
    tasks = [
        fetch_stock_data(row['symbol'], row['total_shares'], row['avg_cost'])
        for _, row in holdings.iterrows()
    ]
    
    results = await asyncio.gather(*tasks)
    holdings_list = [r for r in results if r is not None]
    
    performers = [
        {
            "symbol": h["symbol"],
            "roi": round(h["roi"], 2),
            "pnl": round(h["pnl"], 2),
            "current_price": round(h["current_price"], 2),
            "avg_cost": round(h["avg_cost"], 2)
        }
        for h in holdings_list
    ]
    
    performers.sort(key=lambda x: x["roi"], reverse=True)
    
    return {
        "best": performers[:5],
        "worst": performers[-5:][::-1]
    }



# ---------------- EXPORT & REPORTING ----------------
@app.get("/export/csv")
async def export_csv():
    """Export current portfolio as CSV"""
    from fastapi.responses import StreamingResponse
    import io
    
    overview = await portfolio_overview()
    if "error" in overview:
        raise HTTPException(status_code=404, detail=overview["error"])
    
    # Create CSV
    output = io.StringIO()
    output.write("Symbol,Shares,Avg Cost,Current Price,Equity,P&L,ROI %,Sector,Allocation %\n")
    
    for holding in overview["holdings"]:
        output.write(f"{holding['symbol']},{holding['shares']},{holding['avg_cost']},"
                    f"{holding['current_price']},{holding['equity']},{holding['pnl']},"
                    f"{holding['roi']},{holding['sector']},{holding['allocation']}\n")
    
    output.write(f"\nTotal,,,,,{overview['total_equity']},{overview['total_pnl']},{overview['total_roi']},,\n")
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio.csv"}
    )

@app.get("/export/summary")
async def export_summary():
    """Get comprehensive portfolio summary for reporting"""
    overview = await portfolio_overview()
    sectors = await sector_allocation()
    best_worst = await best_worst_performers()
    
    return {
        "overview": overview,
        "sectors": sectors,
        "top_performers": best_worst["best"],
        "worst_performers": best_worst["worst"],
        "generated_at": datetime.utcnow().isoformat()
    }

# ---------------- TRANSACTIONS HISTORY ----------------
@app.get("/transactions/all")
def get_all_transactions_api(
    limit: int = Query(100, description="Number of transactions to return"),
    offset: int = Query(0, description="Offset for pagination"),
    trans_type: str = Query(None, description="Filter by transaction type (Buy, Sell, etc.)")
):
    """Get all transactions with pagination"""
    portfolio = load_and_sanitize_portfolio()
    if portfolio.empty:
        return {"transactions": [], "total": 0}
    
    # Filter by transaction type if specified
    if trans_type:
        filtered = portfolio[portfolio['trans_type'] == trans_type]
    else:
        filtered = portfolio
    
    # Sort by date descending
    filtered = filtered.sort_values('date', ascending=False)
    
    total = len(filtered)
    transactions = filtered.iloc[offset:offset+limit]
    
    return {
        "transactions": transactions.to_dict(orient="records"),
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/transactions/symbol/{symbol}")
def get_symbol_transactions(symbol: str):
    """Get all transactions for a specific symbol"""
    portfolio = load_and_sanitize_portfolio()
    symbol = symbol.strip().upper()
    
    trades = portfolio[portfolio['symbol'] == symbol].sort_values('date', ascending=False)
    
    if trades.empty:
        return {"error": "No transactions found for this symbol"}
    
    return {
        "symbol": symbol,
        "total_transactions": len(trades),
        "transactions": trades.to_dict(orient="records")
    }

@app.get("/transactions/summary")
def transaction_summary():
    """Get summary of all transactions"""
    portfolio = load_and_sanitize_portfolio()
    if portfolio.empty:
        return {"error": "No transactions"}
    
    buys = portfolio[portfolio['trans_type'] == 'Buy']
    sells = portfolio[portfolio['trans_type'] == 'Sell']
    
    buy_value = (buys['shares'] * buys['price']).sum()
    sell_value = (sells['shares'].abs() * sells['price']).sum()
    
    return {
        "total_transactions": len(portfolio),
        "total_buys": len(buys),
        "total_sells": len(sells),
        "buy_volume": round(buy_value, 2),
        "sell_volume": round(sell_value, 2),
        "net_invested": round(buy_value - sell_value, 2),
        "date_range": {
            "first": portfolio['date'].min().strftime('%Y-%m-%d'),
            "last": portfolio['date'].max().strftime('%Y-%m-%d')
        }
    }

# ---------------- NON-TRADE TRANSACTIONS ----------------
@app.get("/nontrade/all")
def get_nontrade_transactions():
    """Get all non-trade transactions (dividends, interest, fees, etc.)"""
    all_trans = get_all_transactions(ACTIVE_PORTFOLIO_FILE)
    if all_trans.empty:
        return {"transactions": []}
    
    # Filter for non Buy/Sell transactions
    nontrade = all_trans[~all_trans['Trans Code'].isin(['Buy', 'Sell'])].copy()
    nontrade = nontrade.sort_values('date', ascending=False)
    
    # Replace NaN with None for JSON serialization
    nontrade = nontrade.replace({pd.NA: None, float('nan'): None})
    nontrade = nontrade.where(pd.notnull(nontrade), None)
    
    return {
        "transactions": nontrade.to_dict(orient="records"),
        "total": len(nontrade)
    }

@app.get("/nontrade/summary")
def nontrade_summary():
    """Get summary of non-trade transactions by type"""
    all_trans = get_all_transactions(ACTIVE_PORTFOLIO_FILE)
    if all_trans.empty:
        return {"summary": []}
    
    nontrade = all_trans[~all_trans['Trans Code'].isin(['Buy', 'Sell'])]
    
    summary = []
    for trans_type in nontrade['Trans Code'].unique():
        if pd.isna(trans_type):
            continue
        type_trans = nontrade[nontrade['Trans Code'] == trans_type]
        
        # Try to parse amounts
        amounts = []
        for amt in type_trans['Amount']:
            if pd.isna(amt):
                continue
            try:
                clean_amt = str(amt).replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                amounts.append(float(clean_amt))
            except:
                continue
        
        total_amount = sum(amounts) if amounts else 0
        
        summary.append({
            "type": trans_type,
            "count": len(type_trans),
            "total_amount": round(total_amount, 2)
        })
    
    summary.sort(key=lambda x: abs(x['total_amount']), reverse=True)
    return {"summary": summary}

# ---------------- STOCK LENDING ----------------
@app.get("/lending/transactions")
def get_lending_transactions():
    """Get stock lending related transactions"""
    all_trans = get_all_transactions(ACTIVE_PORTFOLIO_FILE)
    if all_trans.empty:
        return {"transactions": []}
    
    # Filter for lending-related codes (SLFEE, SLINT, etc.)
    lending = all_trans[all_trans['Trans Code'].str.contains('SL', na=False, case=False)].copy()
    lending = lending.sort_values('date', ascending=False)
    
    # Replace NaN with None for JSON serialization
    lending = lending.replace({pd.NA: None, float('nan'): None})
    lending = lending.where(pd.notnull(lending), None)
    
    return {
        "transactions": lending.to_dict(orient="records"),
        "total": len(lending)
    }

@app.get("/lending/summary")
def lending_summary():
    """Get summary of stock lending income"""
    all_trans = get_all_transactions(ACTIVE_PORTFOLIO_FILE)
    if all_trans.empty:
        return {"total_income": 0, "transactions": 0}
    
    lending = all_trans[all_trans['Trans Code'].str.contains('SL', na=False, case=False)]
    
    # Calculate total income
    total_income = 0
    for amt in lending['Amount']:
        if pd.isna(amt):
            continue
        try:
            clean_amt = str(amt).replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
            total_income += float(clean_amt)
        except:
            continue
    
    return {
        "total_income": round(total_income, 2),
        "transaction_count": len(lending),
        "symbols_lent": lending['Instrument'].nunique() if 'Instrument' in lending.columns else 0
    }

@app.get("/dividends/summary")
def get_dividends_summary():
    """Get dividend summary by symbol"""
    all_trans = get_all_transactions(ACTIVE_PORTFOLIO_FILE)
    if all_trans.empty:
        return {"dividends": [], "total_dividends": 0}
    
    # Filter for dividend transactions (check Description field)
    dividends = all_trans[all_trans['Description'].str.contains('dividend', case=False, na=False)].copy()
    
    if dividends.empty:
        return {"dividends": [], "total_dividends": 0}
    
    # Group by symbol and calculate total dividends
    dividend_summary = []
    total_all = 0
    
    for symbol in dividends['Instrument'].unique():
        if pd.isna(symbol):
            continue
        
        symbol_divs = dividends[dividends['Instrument'] == symbol]
        
        # Calculate total dividend amount
        total = 0
        transactions = []
        for _, row in symbol_divs.iterrows():
            try:
                amt_str = str(row['Amount']).replace('$', '').replace(',', '').replace('(', '').replace(')', '')
                amt = abs(float(amt_str))  # Use absolute value since dividends are shown as negative
                total += amt
                transactions.append({
                    "date": row['date'].isoformat() if pd.notna(row.get('date')) else row['Settle Date'],
                    "amount": round(amt, 2)
                })
            except:
                continue
        
        if total > 0:
            dividend_summary.append({
                "symbol": symbol,
                "total_dividends": round(total, 2),
                "count": len(transactions),
                "transactions": transactions
            })
            total_all += total
    
    # Sort by total dividends descending
    dividend_summary.sort(key=lambda x: x['total_dividends'], reverse=True)
    
    return {
        "dividends": dividend_summary,
        "total_dividends": round(total_all, 2)
    }

# Vercel serverless handler - must be named 'handler' or 'app'
app = app  # Explicit assignment for Vercel
handler = app