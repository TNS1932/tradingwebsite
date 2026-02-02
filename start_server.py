"""Start the portfolio tracker server"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "maininvestingwebpage:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable auto-reload to prevent constant restarting
    )
