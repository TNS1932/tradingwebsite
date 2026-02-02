import logging
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import re
import secrets

# ---------------- APP SETUP ----------------
app = FastAPI(title="Portfolio Tracker API")
logger = logging.getLogger("portfolio_app")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIVE_PORTFOLIO_FILE = "portfolio_data.csv"

# ---------------- UTILITIES ----------------
def parse_brokerage_csv(filepath: str) -> pd.DataFrame:
    """Parse the brokerage CSV format and extract stock purchases"""
    try:
        # Read CSV and skip bad lines (multiline descriptions still exist)
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return pd.DataFrame(columns=["symbol", "shares", "price", "date", "trans_type"])
    
    # Filter for Buy and Sell transactions only
    if 'Trans Code' not in df.columns:
        logger.error("CSV missing 'Trans Code' column")
        return pd.DataFrame(columns=["symbol", "shares", "price", "date", "trans_type"])
    
    df = df[df['Trans Code'].isin(['Buy', 'Sell'])].copy()
    
    # Extract symbol from Instrument column (letters only)
    df['symbol'] = df['Instrument'].astype(str).str.strip().str.upper()
    
    # Remove any rows where symbol is empty or contains numbers
    df = df[df['symbol'].str.match(r'^[A-Z.]+$', na=False)]
    
    # Extract quantity and price from the actual column names in CSV
    df['shares'] = pd.to_numeric(df['Shares'], errors='coerce')
    df['price'] = pd.to_numeric(df['Share_Price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    df['date'] = pd.to_datetime(df['Settle Date'], errors='coerce')
    df['trans_type'] = df['Trans Code']
    
    # For sells, make shares negative
    df.loc[df['trans_type'] == 'Sell', 'shares'] = -df.loc[df['trans_type'] == 'Sell', 'shares']
    
    # Clean up
    df = df[['date', 'symbol', 'shares', 'price', 'trans_type']].dropna()
    
    return df

def get_all_transactions(filepath: str) -> pd.DataFrame:
    """Get all transactions including non-trade transactions"""
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

# ---------------- CSV UPLOAD ----------------
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global ACTIVE_PORTFOLIO_FILE

    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"

    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    ACTIVE_PORTFOLIO_FILE = path
    df = parse_brokerage_csv(path)

    return {
        "message": "CSV uploaded successfully",
        "active_file": path,
        "rows_loaded": len(df),
        "symbols": df["symbol"].unique().tolist()
    }

# ---------------- UPLOAD PASSWORD ----------------
UPLOAD_PASSWORD = os.getenv("UPLOAD_PASSWORD", "portfolio2026")  # Change this in production

# ---------------- CSV UPLOAD ----------------
@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    """Serve upload page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Portfolio Data</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 0;
            }
            .upload-container {
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                max-width: 500px;
                width: 100%;
            }
            h1 {
                color: #667eea;
                margin-bottom: 30px;
            }
            input[type="password"], input[type="file"] {
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                font-size: 16px;
                box-sizing: border-box;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                margin-top: 20px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
            }
            .message {
                padding: 15px;
                margin-top: 20px;
                border-radius: 8px;
                display: none;
            }
            .success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
        </style>
    </head>
    <body>
        <div class="upload-container">
            <h1>📊 Upload Portfolio CSV</h1>
            <form id="uploadForm">
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
                
                <label for="file">CSV File:</label>
                <input type="file" id="file" name="file" accept=".csv" required>
                
                <button type="submit">Upload</button>
            </form>
            <div id="message" class="message"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('file', document.getElementById('file').files[0]);
                formData.append('password', document.getElementById('password').value);
                
                const messageDiv = document.getElementById('message');
                messageDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        messageDiv.className = 'message success';
                        messageDiv.textContent = data.message;
                        messageDiv.style.display = 'block';
                        document.getElementById('uploadForm').reset();
                    } else {
                        messageDiv.className = 'message error';
                        messageDiv.textContent = data.detail || 'Upload failed';
                        messageDiv.style.display = 'block';
                    }
                } catch (error) {
                    messageDiv.className = 'message error';
                    messageDiv.textContent = 'Upload failed: ' + error.message;
                    messageDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), password: str = Form(...)):
    """Upload portfolio CSV with password protection"""
    # Verify password
    if not secrets.compare_digest(password, UPLOAD_PASSWORD):
        raise HTTPException(status_code=403, detail="Invalid password")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Save the uploaded file
        content = await file.read()
        with open(ACTIVE_PORTFOLIO_FILE, 'wb') as f:
            f.write(content)
        
        # Verify it can be parsed
        test_parse = parse_brokerage_csv(ACTIVE_PORTFOLIO_FILE)
        if test_parse.empty:
            raise HTTPException(status_code=400, detail="CSV file appears to be empty or invalid")
        
        return {
            "message": f"Successfully uploaded {file.filename}. Found {len(test_parse)} transactions.",
            "transactions": len(test_parse),
            "symbols": len(test_parse['symbol'].unique()) if 'symbol' in test_parse.columns else 0
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

# ---------------- GET ALL SYMBOLS ----------------
@app.get("/symbols")
def get_symbols():
    portfolio = load_and_sanitize_portfolio()
    if portfolio.empty:
        return {"symbols": []}
    symbols = sorted(portfolio["symbol"].unique().tolist())
    return {"symbols": symbols}

# ---------------- SERVE HTML ----------------
@app.get("/")
def serve_html():
    return FileResponse("index.html")

# ---------------- MARKET DATA ----------------
@app.get("/market/{symbol}")
def market_data(symbol: str):
    symbol = symbol.strip().upper()
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5y")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    if hist.empty:
        raise HTTPException(status_code=404, detail="No market data returned")
    return hist.reset_index().to_dict(orient="records")

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
    stock = yf.Ticker(symbol)
    price_series = stock.history(period="1d")["Close"]
    if price_series.empty:
        raise HTTPException(status_code=503, detail="Price lookup failed")

    price = float(price_series.iloc[-1])
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

# ---------------- TIMELINE CHART ENGINE ----------------
@app.get("/portfolio_timeseries/{symbol}")
def portfolio_timeseries(
    symbol: str,
    range: str = Query("1y", enum=["1d", "1w", "1m", "3m", "1y", "5y"])
):
    portfolio = load_and_sanitize_portfolio()
    symbol = symbol.strip().upper()
    trades = portfolio[portfolio["symbol"] == symbol]
    if trades.empty:
        return {"error": "symbol not found in portfolio"}

    total_shares = trades["shares"].sum()
    if total_shares == 0:
        return {"error": "symbol has zero shares"}

    avg_cost = (trades["shares"] * trades["price"]).sum() / total_shares
    period_map = {
        "1d": "1d",
        "1w": "7d",
        "1m": "1mo",
        "3m": "3mo",
        "1y": "1y",
        "5y": "5y"
    }

    stock = yf.Ticker(symbol)
    hist = stock.history(period=period_map[range])
    if hist.empty:
        raise HTTPException(status_code=503, detail="No historical data")

    hist = hist.reset_index()
    hist["equity"] = hist["Close"] * total_shares
    hist["pnl"] = (hist["Close"] - avg_cost) * total_shares

    return hist[["Date", "Close", "equity", "pnl"]].to_dict(orient="records")

# ---------------- TOTAL PORTFOLIO ----------------
@app.get("/portfolio_total")
def portfolio_total(range: str = "1y"):
    portfolio = load_and_sanitize_portfolio()
    if portfolio.empty:
        return []

    symbols = portfolio["symbol"].unique()
    combined = None

    for sym in symbols:
        shares = portfolio[portfolio["symbol"] == sym]["shares"].sum()
        if shares == 0:
            continue

        stock = yf.Ticker(sym)
        hist = stock.history(period=range)
        if hist.empty:
            continue

        hist["value"] = hist["Close"] * shares
        if combined is None:
            combined = hist["value"]
        else:
            combined += hist["value"]

    if combined is None:
        return []

    combined = combined.reset_index()
    return combined.to_dict(orient="records")

# ---------------- PORTFOLIO ANALYTICS ----------------
@app.get("/analytics/overview")
def portfolio_overview():
    """Get complete portfolio analytics including total value, P&L, allocation"""
    holdings = get_current_holdings()
    if holdings.empty:
        return {"error": "No portfolio data"}
    
    holdings_list = []
    total_equity = 0
    total_cost = 0
    
    for _, row in holdings.iterrows():
        symbol = row['symbol']
        shares = row['total_shares']
        avg_cost = row['avg_cost']
        
        try:
            stock = yf.Ticker(symbol)
            price = stock.history(period="1d")["Close"].iloc[-1]
            info = stock.info
            sector = info.get("sector", "Unknown")
            
            equity = shares * price
            pnl = (price - avg_cost) * shares
            cost_basis = shares * avg_cost
            
            total_equity += equity
            total_cost += cost_basis
            
            holdings_list.append({
                "symbol": symbol,
                "shares": float(shares),
                "avg_cost": float(avg_cost),
                "current_price": float(price),
                "equity": float(equity),
                "pnl": float(pnl),
                "roi": float((price - avg_cost) / avg_cost * 100),
                "sector": sector,
                "allocation": 0  # Will calculate after we have total
            })
        except:
            continue
    
    # Calculate allocations
    for holding in holdings_list:
        holding["allocation"] = (holding["equity"] / total_equity * 100) if total_equity > 0 else 0
    
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
def sector_allocation():
    """Get portfolio allocation by sector"""
    holdings = get_current_holdings()
    if holdings.empty:
        return {"sectors": []}
    
    sector_data = {}
    
    for _, row in holdings.iterrows():
        try:
            stock = yf.Ticker(row['symbol'])
            price = stock.history(period="1d")["Close"].iloc[-1]
            info = stock.info
            sector = info.get("sector", "Unknown")
            
            equity = row['total_shares'] * price
            
            if sector in sector_data:
                sector_data[sector] += equity
            else:
                sector_data[sector] = equity
        except:
            continue
    
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

@app.get("/analytics/performance")
def performance_metrics():
    """Calculate risk metrics and performance vs S&P 500"""
    holdings = get_current_holdings()
    if holdings.empty:
        return {"error": "No portfolio data"}
    
    # Get portfolio historical values
    portfolio_hist = None
    
    for _, row in holdings.iterrows():
        try:
            stock = yf.Ticker(row['symbol'])
            hist = stock.history(period="1y")["Close"]
            if portfolio_hist is None:
                portfolio_hist = hist * row['total_shares']
            else:
                portfolio_hist = portfolio_hist.add(hist * row['total_shares'], fill_value=0)
        except:
            continue
    
    if portfolio_hist is None or portfolio_hist.empty:
        return {"error": "Could not calculate performance"}
    
    # Get S&P 500 data
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="1y")["Close"]
        spy_returns = spy_hist.pct_change().dropna()
    except:
        spy_returns = None
    
    # Calculate returns
    portfolio_returns = portfolio_hist.pct_change().dropna()
    
    # Calculate metrics
    annual_return = (portfolio_hist.iloc[-1] / portfolio_hist.iloc[0] - 1) * 100
    volatility = portfolio_returns.std() * (252 ** 0.5) * 100  # Annualized
    sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std() * (252 ** 0.5)) if portfolio_returns.std() > 0 else 0
    
    result = {
        "annual_return": round(annual_return, 2),
        "volatility": round(volatility, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "total_days": len(portfolio_returns)
    }
    
    if spy_returns is not None:
        spy_annual = (spy_hist.iloc[-1] / spy_hist.iloc[0] - 1) * 100
        result["spy_return"] = round(spy_annual, 2)
        result["outperformance"] = round(annual_return - spy_annual, 2)
        
        # Calculate beta
        aligned_returns = pd.DataFrame({"portfolio": portfolio_returns, "spy": spy_returns}).dropna()
        if len(aligned_returns) > 0:
            covariance = aligned_returns["portfolio"].cov(aligned_returns["spy"])
            spy_variance = aligned_returns["spy"].var()
            beta = covariance / spy_variance if spy_variance > 0 else 1
            result["beta"] = round(beta, 2)
    
    return result

@app.get("/analytics/best_worst")
def best_worst_performers():
    """Get best and worst performing stocks"""
    holdings = get_current_holdings()
    if holdings.empty:
        return {"best": [], "worst": []}
    
    performers = []
    
    for _, row in holdings.iterrows():
        try:
            stock = yf.Ticker(row['symbol'])
            price = stock.history(period="1d")["Close"].iloc[-1]
            
            roi = (price - row['avg_cost']) / row['avg_cost'] * 100
            pnl = (price - row['avg_cost']) * row['total_shares']
            
            performers.append({
                "symbol": row['symbol'],
                "roi": round(roi, 2),
                "pnl": round(pnl, 2),
                "current_price": round(price, 2),
                "avg_cost": round(row['avg_cost'], 2)
            })
        except:
            continue
    
    performers.sort(key=lambda x: x["roi"], reverse=True)
    
    return {
        "best": performers[:5],
        "worst": performers[-5:][::-1]
    }

# ---------------- DATE RANGE ANALYSIS ----------------
@app.get("/analytics/date_range")
def date_range_analysis(
    symbol: str,
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    """Get portfolio performance for a specific date range"""
    portfolio = load_and_sanitize_portfolio()
    symbol = symbol.strip().upper()
    trades = portfolio[portfolio["symbol"] == symbol]
    
    if trades.empty:
        return {"error": "Symbol not found"}
    
    shares = trades["shares"].sum()
    avg_cost = (trades["shares"] * trades["price"]).sum() / shares
    
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)
        
        hist["equity"] = hist["Close"] * shares
        hist["pnl"] = (hist["Close"] - avg_cost) * shares
        
        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "data": hist.reset_index()[["Date", "Close", "equity", "pnl"]].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/monthly_breakdown")
def monthly_breakdown(year: int = Query(2025)):
    """Get monthly P&L breakdown for a given year"""
    holdings = get_current_holdings()
    if holdings.empty:
        return {"months": []}
    
    monthly_data = []
    current_date = datetime.now()
    
    for month in range(1, 13):
        # Skip future months
        if year == current_date.year and month > current_date.month:
            continue
            
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year+1}-01-01"
        else:
            end_date = f"{year}-{month+1:02d}-01"
        
        month_pnl = 0
        month_value = 0
        has_data = False
        
        for _, row in holdings.iterrows():
            try:
                stock = yf.Ticker(row['symbol'])
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
                    month_value += price * row['total_shares']
                    month_pnl += (price - row['avg_cost']) * row['total_shares']
                    has_data = True
            except:
                continue
        
        # Only add months that have actual data
        if has_data:
            monthly_data.append({
                "month": month,
                "month_name": pd.Timestamp(year=year, month=month, day=1).strftime("%B"),
                "value": round(month_value, 2),
                "pnl": round(month_pnl, 2)
            })
    
    return {"year": year, "months": monthly_data}

# ---------------- EXPORT & REPORTING ----------------
@app.get("/export/csv")
def export_csv():
    """Export current portfolio as CSV"""
    from fastapi.responses import StreamingResponse
    import io
    
    overview = portfolio_overview()
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
def export_summary():
    """Get comprehensive portfolio summary for reporting"""
    overview = portfolio_overview()
    sectors = sector_allocation()
    performance = performance_metrics()
    best_worst = best_worst_performers()
    
    return {
        "overview": overview,
        "sectors": sectors,
        "performance": performance,
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