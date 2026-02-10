# This file imports from the main module for Vercel serverless deployment
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from maininvestingwebpage import app, handler

# Export for Vercel
__all__ = ['app', 'handler']