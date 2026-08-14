"""
Institutional Quantitative Trading Terminal & Risk Management Suite
Platform: Streamlit | Execution: DhanHQ v2 / Paper-Trading Fallback | Feeds: Yahoo Finance & Google News
"""

from __future__ import annotations
import sys
import io
import os
import time
import hashlib
import contextlib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Attempt DhanHQ import gracefully
try:
    from dhanhq import dhanhq
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False


# ==============================================================================
# 0. CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="Apex Quant Trading Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 1. CORE UNIVERSE & TICKER ENGINE (200+ NSE & 200+ US EQUITIES)
# ------------------------------------------------------------------------------
NSE_EQUITIES: Dict[str, Dict[str, Any]] = {
    # Nifty 50 & Heavyweights
    "RELIANCE": {"id": "2885", "name": "Reliance Industries Ltd", "sector": "Energy"},
    "TCS": {"id": "11536", "name": "Tata Consultancy Services Ltd", "sector": "IT"},
    "HDFCBANK": {"id": "1333", "name": "HDFC Bank Ltd", "sector": "Financials"},
    "ICICIBANK": {"id": "4963", "name": "ICICI Bank Ltd", "sector": "Financials"},
    "INFY": {"id": "1594", "name": "Infosys Ltd", "sector": "IT"},
    "HINDUNILVR": {"id": "1394", "name": "Hindustan Unilever Ltd", "sector": "FMCG"},
    "ITC": {"id": "1660", "name": "ITC Ltd", "sector": "FMCG"},
    "SBIN": {"id": "3045", "name": "State Bank of India", "sector": "Financials"},
    "BHARTIARTL": {"id": "10604", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
    "KOTAKBANK": {"id": "1922", "name": "Kotak Mahindra Bank Ltd", "sector": "Financials"},
    "LT": {"id": "11483", "name": "Larsen & Toubro Ltd", "sector": "Capital Goods"},
    "AXISBANK": {"id": "5900", "name": "Axis Bank Ltd", "sector": "Financials"},
    "ASIANPAINT": {"id": "236", "name": "Asian Paints Ltd", "sector": "Consumer Discretionary"},
    "HCLTECH": {"id": "7229", "name": "HCL Technologies Ltd", "sector": "IT"},
    "BAJFINANCE": {"id": "317", "name": "Bajaj Finance Ltd", "sector": "Financials"},
    "MARUTI": {"id": "10999", "name": "Maruti Suzuki India Ltd", "sector": "Auto"},
    "SUNPHARMA": {"id": "3351", "name": "Sun Pharmaceutical Inds", "sector": "Healthcare"},
    "TITAN": {"id": "3506", "name": "Titan Company Ltd", "sector": "Consumer Discretionary"},
    "TATAMOTORS": {"id": "3456", "name": "Tata Motors Ltd", "sector": "Auto"},
    "ULTRACEMCO": {"id": "11532", "name": "UltraTech Cement Ltd", "sector": "Materials"},
    "NTPC": {"id": "11630", "name": "NTPC Ltd", "sector": "Utilities"},
    "ONGC": {"id": "2475", "name": "Oil & Natural Gas Corp Ltd", "sector": "Energy"},
    "WIPRO": {"id": "3787", "name": "Wipro Ltd", "sector": "IT"},
    "POWERGRID": {"id": "14977", "name": "Power Grid Corp of India", "sector": "Utilities"},
    "ADANIENT": {"id": "25", "name": "Adani Enterprises Ltd", "sector": "Metals & Mining"},
    "ADANIPORTS": {"id": "15083", "name": "Adani Ports and SEZ Ltd", "sector": "Infrastructure"},
    "M&M": {"id": "2031", "name": "Mahindra & Mahindra Ltd", "sector": "Auto"},
    "TATASTEEL": {"id": "3499", "name": "Tata Steel Ltd", "sector": "Metals & Mining"},
    "COALINDIA": {"id": "20374", "name": "Coal India Ltd", "sector": "Energy"},
    "BAJAJFINSV": {"id": "16675", "name": "Bajaj Finserv Ltd", "sector": "Financials"},
    "JSWSTEEL": {"id": "11723", "name": "JSW Steel Ltd", "sector": "Metals & Mining"},
    "GRASIM": {"id": "1232", "name": "Grasim Industries Ltd", "sector": "Materials"},
    "TECHM": {"id": "13538", "name": "Tech Mahindra Ltd", "sector": "IT"},
    "HINDALCO": {"id": "1363", "name": "Hindalco Industries Ltd", "sector": "Metals & Mining"},
    "CIPLA": {"id": "694", "name": "Cipla Ltd", "sector": "Healthcare"},
    "DRREDDY": {"id": "881", "name": "Dr. Reddy's Laboratories", "sector": "Healthcare"},
    "EICHERMOT": {"id": "910", "name": "Eicher Motors Ltd", "sector": "Auto"},
    "NESTLEIND": {"id": "17963", "name": "Nestle India Ltd", "sector": "FMCG"},
    "APOLLOHOSP": {"id": "157", "name": "Apollo Hospitals Enterprise", "sector": "Healthcare"},
    "DIVISLAB": {"id": "10940", "name": "Divi's Laboratories Ltd", "sector": "Healthcare"},
    "BPCL": {"id": "526", "name": "Bharat Petroleum Corp Ltd", "sector": "Energy"},
    "TATACONSUM": {"id": "3432", "name": "Tata Consumer Products Ltd", "sector": "FMCG"},
    "BRITANNIA": {"id": "547", "name": "Britannia Industries Ltd", "sector": "FMCG"},
    "HEROMOTOCO": {"id": "1348", "name": "Hero MotoCorp Ltd", "sector": "Auto"},
    "SBILIFE": {"id": "21808", "name": "SBI Life Insurance Co Ltd", "sector": "Financials"},
    "HDFCLIFE": {"id": "467", "name": "HDFC Life Insurance Co Ltd", "sector": "Financials"},
    "BAJAJ-AUTO": {"id": "16669", "name": "Bajaj Auto Ltd", "sector": "Auto"},
    "INDUSINDBK": {"id": "5258", "name": "IndusInd Bank Ltd", "sector": "Financials"},
    "SHRIRAMFIN": {"id": "4306", "name": "Shriram Finance Ltd", "sector": "Financials"},
    "TRENT": {"id": "1964", "name": "Trent Ltd", "sector": "Consumer Discretionary"},
    "BEL": {"id": "383", "name": "Bharat Electronics Ltd", "sector": "Capital Goods"},
    "HAL": {"id": "2303", "name": "Hindustan Aeronautics Ltd", "sector": "Capital Goods"},
    "VEDL": {"id": "3063", "name": "Vedanta Ltd", "sector": "Metals & Mining"},
    "DLF": {"id": "14732", "name": "DLF Ltd", "sector": "Real Estate"},
    "ZOMATO": {"id": "5097", "name": "Zomato Ltd", "sector": "Consumer Services"},
    "JIOFIN": {"id": "18143", "name": "Jio Financial Services Ltd", "sector": "Financials"},
    "VBL": {"id": "19367", "name": "Varun Beverages Ltd", "sector": "FMCG"},
    "CHOLAFIN": {"id": "685", "name": "Cholamandalam Invest & Fin", "sector": "Financials"},
    "SIEMENS": {"id": "3150", "name": "Siemens Ltd", "sector": "Capital Goods"},
    "ABB": {"id": "13", "name": "ABB India Ltd", "sector": "Capital Goods"},
    "PFC": {"id": "14299", "name": "Power Finance Corporation", "sector": "Financials"},
    "RECLTD": {"id": "15355", "name": "REC Ltd", "sector": "Financials"},
    "TVSMOTOR": {"id": "8479", "name": "TVS Motor Company Ltd", "sector": "Auto"},
    "HAVELLS": {"id": "9819", "name": "Havells India Ltd", "sector": "Consumer Durables"},
    "PIDILITIND": {"id": "2664", "name": "Pidilite Industries Ltd", "sector": "Materials"},
    "BANKBARODA": {"id": "4668", "name": "Bank of Baroda", "sector": "Financials"},
    "PNB": {"id": "10666", "name": "Punjab National Bank", "sector": "Financials"},
    "CANBK": {"id": "10794", "name": "Canara Bank", "sector": "Financials"},
    "UNIONBANK": {"id": "10839", "name": "Union Bank of India", "sector": "Financials"},
    "INDIGO": {"id": "11195", "name": "InterGlobe Aviation Ltd", "sector": "Aviation"},
    "GAIL": {"id": "4717", "name": "GAIL India Ltd", "sector": "Utilities"},
    "IOC": {"id": "1624", "name": "Indian Oil Corporation Ltd", "sector": "Energy"},
    "AMBUJACEM": {"id": "1270", "name": "Ambuja Cements Ltd", "sector": "Materials"},
    "ACC": {"id": "22", "name": "ACC Ltd", "sector": "Materials"},
    "DABUR": {"id": "772", "name": "Dabur India Ltd", "sector": "FMCG"},
    "GODREJCP": {"id": "10099", "name": "Godrej Consumer Products", "sector": "FMCG"},
    "MARICO": {"id": "2083", "name": "Marico Ltd", "sector": "FMCG"},
    "COLPAL": {"id": "751", "name": "Colgate-Palmolive (India)", "sector": "FMCG"},
    "BERGEPAINT": {"id": "404", "name": "Berger Paints India Ltd", "sector": "Materials"},
    "MOTHERSON": {"id": "4204", "name": "Samvardhana Motherson Int", "sector": "Auto"},
    "BOSCHLTD": {"id": "2181", "name": "Bosch Ltd", "sector": "Auto"},
    "BALKRISIND": {"id": "335", "name": "Balkrishna Industries Ltd", "sector": "Auto"},
    "MRF": {"id": "2277", "name": "MRF Ltd", "sector": "Auto"},
    "APOLLOTYRE": {"id": "163", "name": "Apollo Tyres Ltd", "sector": "Auto"},
    "LUPIN": {"id": "10440", "name": "Lupin Ltd", "sector": "Healthcare"},
    "AUROPHARMA": {"id": "275", "name": "Aurobindo Pharma Ltd", "sector": "Healthcare"},
    "BIOCON": {"id": "11373", "name": "Biocon Ltd", "sector": "Healthcare"},
    "TORNTPHARM": {"id": "3563", "name": "Torrent Pharmaceuticals", "sector": "Healthcare"},
    "ALKEM": {"id": "11703", "name": "Alkem Laboratories Ltd", "sector": "Healthcare"},
    "IPCALAB": {"id": "1633", "name": "IPCA Laboratories Ltd", "sector": "Healthcare"},
    "ZYDUSLIFE": {"id": "7929", "name": "Zydus Lifesciences Ltd", "sector": "Healthcare"},
    "PERSISTENT": {"id": "18365", "name": "Persistent Systems Ltd", "sector": "IT"},
    "COFORGE": {"id": "11543", "name": "Coforge Ltd", "sector": "IT"},
    "MPHASIS": {"id": "4503", "name": "Mphasis Ltd", "sector": "IT"},
    "LTIM": {"id": "17818", "name": "LTIMindtree Ltd", "sector": "IT"},
    "LTTS": {"id": "18564", "name": "L&T Technology Services", "sector": "IT"},
    "KPITTECH": {"id": "2087", "name": "KPIT Technologies Ltd", "sector": "IT"},
    "TATAELXSI": {"id": "3441", "name": "Tata Elxsi Ltd", "sector": "IT"},
    "POLYCAB": {"id": "9590", "name": "Polycab India Ltd", "sector": "Capital Goods"},
    "KEI": {"id": "5190", "name": "KEI Industries Ltd", "sector": "Capital Goods"},
    "VOLTAS": {"id": "3718", "name": "Voltas Ltd", "sector": "Consumer Durables"},
    "BLUESTARCO": {"id": "488", "name": "Blue Star Ltd", "sector": "Consumer Durables"},
    "CROMPTON": {"id": "17094", "name": "Crompton Greaves Cons", "sector": "Consumer Durables"},
    "WHIRLPOOL": {"id": "3775", "name": "Whirlpool of India Ltd", "sector": "Consumer Durables"},
    "DIXON": {"id": "21690", "name": "Dixon Technologies Ltd", "sector": "Capital Goods"},
    "ASTRAL": {"id": "14418", "name": "Astral Ltd", "sector": "Building Materials"},
    "SUPREMEIND": {"id": "3380", "name": "Supreme Industries Ltd", "sector": "Building Materials"},
    "PIIND": {"id": "24184", "name": "PI Industries Ltd", "sector": "Chemicals"},
    "SRF": {"id": "3273", "name": "SRF Ltd", "sector": "Chemicals"},
    "NAVINFLUOR": {"id": "14672", "name": "Navin Fluorine Int Ltd", "sector": "Chemicals"},
    "DEEPAKNTR": {"id": "19943", "name": "Deepak Nitrite Ltd", "sector": "Chemicals"},
    "TATACHEM": {"id": "3405", "name": "Tata Chemicals Ltd", "sector": "Chemicals"},
    "AARTIIND": {"id": "7", "name": "Aarti Industries Ltd", "sector": "Chemicals"},
    "ATUL": {"id": "263", "name": "Atul Ltd", "sector": "Chemicals"},
    "COROMANDEL": {"id": "739", "name": "Coromandel International", "sector": "Chemicals"},
    "UPL": {"id": "11287", "name": "UPL Ltd", "sector": "Chemicals"},
    "PAGEIND": {"id": "14413", "name": "Page Industries Ltd", "sector": "Textiles"},
    "KALYANKJIL": {"id": "2782", "name": "Kalyan Jewellers India", "sector": "Retail"},
    "NYKAA": {"id": "6545", "name": "FSN E-Commerce (Nykaa)", "sector": "Retail"},
    "PAYTM": {"id": "6401", "name": "One 97 Comm (Paytm)", "sector": "Financials"},
    "POLICYBZR": {"id": "6497", "name": "PB Fintech (Policybazaar)", "sector": "Financials"},
    "DELHIVERY": {"id": "7710", "name": "Delhivery Ltd", "sector": "Logistics"},
    "CONCOR": {"id": "4749", "name": "Container Corp of India", "sector": "Logistics"},
    "GMRINFRA": {"id": "13528", "name": "GMR Airports Infra Ltd", "sector": "Infrastructure"},
    "IRCTC": {"id": "13611", "name": "IRCTC Ltd", "sector": "Consumer Services"},
    "IRFC": {"id": "2029", "name": "Indian Railway Finance", "sector": "Financials"},
    "RVNL": {"id": "10028", "name": "Rail Vikas Nigam Ltd", "sector": "Infrastructure"},
    "BHEL": {"id": "438", "name": "Bharat Heavy Electricals", "sector": "Capital Goods"},
    "NATIONALUM": {"id": "6364", "name": "National Aluminium Co Ltd", "sector": "Metals & Mining"},
    "NMDC": {"id": "15332", "name": "NMDC Ltd", "sector": "Metals & Mining"},
    "SAIL": {"id": "2963", "name": "Steel Authority of India", "sector": "Metals & Mining"},
    "JINDALSTEL": {"id": "6733", "name": "Jindal Steel & Power Ltd", "sector": "Metals & Mining"},
    "HINDZINC": {"id": "1424", "name": "Hindustan Zinc Ltd", "sector": "Metals & Mining"},
    "GODREJPROP": {"id": "17875", "name": "Godrej Properties Ltd", "sector": "Real Estate"},
    "OBEROIRLTY": {"id": "20242", "name": "Oberoi Realty Ltd", "sector": "Real Estate"},
    "PHOENIXLTD": {"id": "14088", "name": "The Phoenix Mills Ltd", "sector": "Real Estate"},
    "PRESTIGE": {"id": "20396", "name": "Prestige Estates Projects", "sector": "Real Estate"},
    "BRIGADE": {"id": "15141", "name": "Brigade Enterprises Ltd", "sector": "Real Estate"},
    "LODHA": {"id": "2911", "name": "Macrotech Developers", "sector": "Real Estate"},
    "SUZLON": {"id": "13061", "name": "Suzlon Energy Ltd", "sector": "Capital Goods"},
    "NHPC": {"id": "17465", "name": "NHPC Ltd", "sector": "Utilities"},
    "SJVN": {"id": "19913", "name": "SJVN Ltd", "sector": "Utilities"},
    "TATAPOWER": {"id": "3426", "name": "Tata Power Company Ltd", "sector": "Utilities"},
    "ADANIGREEN": {"id": "356", "name": "Adani Green Energy Ltd", "sector": "Utilities"},
    "ADANIPOWER": {"id": "17534", "name": "Adani Power Ltd", "sector": "Utilities"},
    "TORNTPOWER": {"id": "13786", "name": "Torrent Power Ltd", "sector": "Utilities"},
    "IDFCFIRSTB": {"id": "11184", "name": "IDFC First Bank Ltd", "sector": "Financials"},
    "FEDERALBNK": {"id": "1023", "name": "Federal Bank Ltd", "sector": "Financials"},
    "AUBANK": {"id": "21238", "name": "AU Small Finance Bank", "sector": "Financials"},
    "BANDHANBNK": {"id": "2263", "name": "Bandhan Bank Ltd", "sector": "Financials"},
    "YESBANK": {"id": "11915", "name": "Yes Bank Ltd", "sector": "Financials"},
    "MFSL": {"id": "2142", "name": "Max Financial Services", "sector": "Financials"},
    "ICICIPRULI": {"id": "18652", "name": "ICICI Prudential Life Ins", "sector": "Financials"},
    "ICICIGI": {"id": "21770", "name": "ICICI Lombard General Ins", "sector": "Financials"},
    "HDFCAMC": {"id": "4244", "name": "HDFC Asset Management Co", "sector": "Financials"},
    "NAM-INDIA": {"id": "21980", "name": "Nippon Life India Asset", "sector": "Financials"},
    "MUTHOOTFIN": {"id": "23650", "name": "Muthoot Finance Ltd", "sector": "Financials"},
    "MANAPPURAM": {"id": "19061", "name": "Manappuram Finance Ltd", "sector": "Financials"},
    "POONAWALLA": {"id": "2043", "name": "Poonawalla Fincorp Ltd", "sector": "Financials"},
    "LICHSGFIN": {"id": "1997", "name": "LIC Housing Finance Ltd", "sector": "Financials"},
    "M&MFIN": {"id": "13285", "name": "M&M Financial Services", "sector": "Financials"},
    "SUNDARMFIN": {"id": "3374", "name": "Sundaram Finance Ltd", "sector": "Financials"},
    "L&TFH": {"id": "24948", "name": "L&T Finance Ltd", "sector": "Financials"},
    "SUNTV": {"id": "13404", "name": "Sun TV Network Ltd", "sector": "Media"},
    "PVRINOX": {"id": "13147", "name": "PVR INOX Ltd", "sector": "Media"},
    "ZEEL": {"id": "3812", "name": "Zee Entertainment Ent", "sector": "Media"},
    "JUBLFOOD": {"id": "18096", "name": "Jubilant FoodWorks Ltd", "sector": "Consumer Services"},
    "DEVYANI": {"id": "5764", "name": "Devyani International Ltd", "sector": "Consumer Services"},
    "WESTLIFE": {"id": "1642", "name": "Westlife Foodworld Ltd", "sector": "Consumer Services"},
    "BATAINDIA": {"id": "371", "name": "Bata India Ltd", "sector": "Consumer Discretionary"},
    "RELAXO": {"id": "11822", "name": "Relaxo Footwears Ltd", "sector": "Consumer Discretionary"},
    "METROBRAND": {"id": "6962", "name": "Metro Brands Ltd", "sector": "Consumer Discretionary"},
    "ENDURANCE": {"id": "18635", "name": "Endurance Technologies", "sector": "Auto"},
    "EXIDEIND": {"id": "676", "name": "Exide Industries Ltd", "sector": "Auto"},
    "AMARAJABAT": {"id": "100", "name": "Amara Raja Energy & Mob", "sector": "Auto"},
    "SONACOMS": {"id": "4884", "name": "Sona BLW Precision", "sector": "Auto"},
    "UNOMINDA": {"id": "11985", "name": "Uno Minda Ltd", "sector": "Auto"},
    "CUMMINSIND": {"id": "1901", "name": "Cummins India Ltd", "sector": "Capital Goods"},
    "THERMAX": {"id": "3536", "name": "Thermax Ltd", "sector": "Capital Goods"},
    "AIAENG": {"id": "11451", "name": "AIA Engineering Ltd", "sector": "Capital Goods"},
    "KEC": {"id": "1738", "name": "KEC International Ltd", "sector": "Capital Goods"},
    "PRAJIND": {"id": "2701", "name": "Praj Industries Ltd", "sector": "Capital Goods"},
    "MAZDOCK": {"id": "2068", "name": "Mazagon Dock Shipbuilders", "sector": "Capital Goods"},
    "COCHINSHIP": {"id": "21727", "name": "Cochin Shipyard Ltd", "sector": "Capital Goods"},
    "GRSE": {"id": "352", "name": "Garden Reach Shipbuilders", "sector": "Capital Goods"},
    "BDL": {"id": "2168", "name": "Bharat Dynamics Ltd", "sector": "Capital Goods"},
    "FACT": {"id": "1016", "name": "Fert & Chem Travancore", "sector": "Chemicals"},
    "RCFLTD": {"id": "2837", "name": "Rashtriya Chem & Fert", "sector": "Chemicals"},
    "GNFC": {"id": "1187", "name": "Gujarat Narmada Valley", "sector": "Chemicals"},
    "GSFC": {"id": "1206", "name": "Gujarat State Fertilizers", "sector": "Chemicals"},
    "CHAMBLFERT": {"id": "637", "name": "Chambal Fertilisers", "sector": "Chemicals"},
    "GLENMARK": {"id": "7406", "name": "Glenmark Pharmaceuticals", "sector": "Healthcare"},
    "ABBOTINDIA": {"id": "14", "name": "Abbott India Ltd", "sector": "Healthcare"},
    "SANOFI": {"id": "3011", "name": "Sanofi India Ltd", "sector": "Healthcare"},
    "GLAXO": {"id": "1172", "name": "GlaxoSmithKline Pharma", "sector": "Healthcare"},
    "PFIZER": {"id": "2643", "name": "Pfizer Ltd", "sector": "Healthcare"},
    "FORTIS": {"id": "14592", "name": "Fortis Healthcare Ltd", "sector": "Healthcare"},
    "MAXHEALTH": {"id": "22223", "name": "Max Healthcare Institute", "sector": "Healthcare"},
    "MEDANTA": {"id": "11570", "name": "Global Health (Medanta)", "sector": "Healthcare"},
    "LALPATHLAB": {"id": "11654", "name": "Dr. Lal PathLabs Ltd", "sector": "Healthcare"},
    "METROPOLIS": {"id": "9581", "name": "Metropolis Healthcare", "sector": "Healthcare"},
    "SYNGENE": {"id": "10243", "name": "Syngene International", "sector": "Healthcare"},
    "LAURUSLABS": {"id": "19234", "name": "Laurus Labs Ltd", "sector": "Healthcare"},
    "GRANULES": {"id": "1186", "name": "Granules India Ltd", "sector": "Healthcare"},
    "JBCHEPHARM": {"id": "1675", "name": "J.B. Chemicals & Pharma", "sector": "Healthcare"},
    "NATCOPHARM": {"id": "2393", "name": "Natco Pharma Ltd", "sector": "Healthcare"},
    "BLS": {"id": "18408", "name": "BLS International Serv", "sector": "Consumer Services"},
    "EASEMYTRIP": {"id": "2792", "name": "Easy Trip Planners Ltd", "sector": "Consumer Services"},
    "INDIAMART": {"id": "10726", "name": "IndiaMART InterMESH", "sector": "IT"},
    "JUSTDIAL": {"id": "9524", "name": "Just Dial Ltd", "sector": "IT"},
    "AFFLE": {"id": "10991", "name": "Affle (India) Ltd", "sector": "IT"},
    "ROUTE": {"id": "19415", "name": "Route Mobile Ltd", "sector": "IT"},
    "TANLA": {"id": "13430", "name": "Tanla Platforms Ltd", "sector": "IT"},
    "MASTEK": {"id": "2110", "name": "Mastek Ltd", "sector": "IT"},
    "SONATSOFTW": {"id": "3267", "name": "Sonata Software Ltd", "sector": "IT"},
    "ZENSARTECH": {"id": "3824", "name": "Zensar Technologies Ltd", "sector": "IT"},
    "CYIENT": {"id": "1517", "name": "Cyient Ltd", "sector": "IT"},
    "BSOFT": {"id": "6892", "name": "Birlasoft Ltd", "sector": "IT"},
    "INTELLECT": {"id": "17388", "name": "Intellect Design Arena", "sector": "IT"},
    "DATAPATTNS": {"id": "7037", "name": "Data Patterns (India)", "sector": "Capital Goods"},
    "KAYNES": {"id": "11983", "name": "Kaynes Technology India", "sector": "Capital Goods"},
    "CENTURYPLY": {"id": "628", "name": "Century Plyboards (India)", "sector": "Building Materials"},
    "KAJARIACER": {"id": "1710", "name": "Kajaria Ceramics Ltd", "sector": "Building Materials"},
    "CERA": {"id": "630", "name": "Cera Sanitaryware Ltd", "sector": "Building Materials"},
    "FINPIPE": {"id": "1041", "name": "Finolex Pipes Ltd", "sector": "Building Materials"},
    "PRINCEPIPE": {"id": "18029", "name": "Prince Pipes and Fittings", "sector": "Building Materials"},
}

US_EQUITIES: Dict[str, Dict[str, Any]] = {
    # Tech Titans & Nasdaq 100 / S&P 500 Leaders
    "AAPL": {"id": "AAPL", "name": "Apple Inc.", "sector": "Information Technology"},
    "MSFT": {"id": "MSFT", "name": "Microsoft Corporation", "sector": "Information Technology"},
    "NVDA": {"id": "NVDA", "name": "NVIDIA Corporation", "sector": "Semiconductors"},
    "AMZN": {"id": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
    "GOOGL": {"id": "GOOGL", "name": "Alphabet Inc. (Class A)", "sector": "Communication Services"},
    "GOOG": {"id": "GOOG", "name": "Alphabet Inc. (Class C)", "sector": "Communication Services"},
    "META": {"id": "META", "name": "Meta Platforms Inc.", "sector": "Communication Services"},
    "TSLA": {"id": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Discretionary"},
    "BRK-B": {"id": "BRK-B", "name": "Berkshire Hathaway Inc.", "sector": "Financials"},
    "AVGO": {"id": "AVGO", "name": "Broadcom Inc.", "sector": "Semiconductors"},
    "LLY": {"id": "LLY", "name": "Eli Lilly and Company", "sector": "Healthcare"},
    "JPM": {"id": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials"},
    "V": {"id": "V", "name": "Visa Inc.", "sector": "Financials"},
    "UNH": {"id": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Healthcare"},
    "XOM": {"id": "XOM", "name": "Exxon Mobil Corporation", "sector": "Energy"},
    "MA": {"id": "MA", "name": "Mastercard Incorporated", "sector": "Financials"},
    "JNJ": {"id": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    "PG": {"id": "PG", "name": "Procter & Gamble Company", "sector": "Consumer Staples"},
    "HD": {"id": "HD", "name": "The Home Depot Inc.", "sector": "Consumer Discretionary"},
    "COST": {"id": "COST", "name": "Costco Wholesale Corp.", "sector": "Consumer Staples"},
    "ABBV": {"id": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare"},
    "MRK": {"id": "MRK", "name": "Merck & Co. Inc.", "sector": "Healthcare"},
    "ORCL": {"id": "ORCL", "name": "Oracle Corporation", "sector": "Information Technology"},
    "AMD": {"id": "AMD", "name": "Advanced Micro Devices", "sector": "Semiconductors"},
    "CRM": {"id": "CRM", "name": "Salesforce Inc.", "sector": "Information Technology"},
    "CVX": {"id": "CVX", "name": "Chevron Corporation", "sector": "Energy"},
    "BAC": {"id": "BAC", "name": "Bank of America Corp.", "sector": "Financials"},
    "NFLX": {"id": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services"},
    "WMT": {"id": "WMT", "name": "Walmart Inc.", "sector": "Consumer Staples"},
    "ADBE": {"id": "ADBE", "name": "Adobe Inc.", "sector": "Information Technology"},
    "QCOM": {"id": "QCOM", "name": "QUALCOMM Incorporated", "sector": "Semiconductors"},
    "LIN": {"id": "LIN", "name": "Linde plc", "sector": "Materials"},
    "TMO": {"id": "TMO", "name": "Thermo Fisher Scientific", "sector": "Healthcare"},
    "CSCO": {"id": "CSCO", "name": "Cisco Systems Inc.", "sector": "Information Technology"},
    "WFC": {"id": "WFC", "name": "Wells Fargo & Company", "sector": "Financials"},
    "INTC": {"id": "INTC", "name": "Intel Corporation", "sector": "Semiconductors"},
    "IBM": {"id": "IBM", "name": "International Business Machines", "sector": "Information Technology"},
    "DIS": {"id": "DIS", "name": "The Walt Disney Company", "sector": "Communication Services"},
    "TXN": {"id": "TXN", "name": "Texas Instruments Inc.", "sector": "Semiconductors"},
    "AMAT": {"id": "AMAT", "name": "Applied Materials Inc.", "sector": "Semiconductors"},
    "PFE": {"id": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare"},
    "PM": {"id": "PM", "name": "Philip Morris International", "sector": "Consumer Staples"},
    "MS": {"id": "MS", "name": "Morgan Stanley", "sector": "Financials"},
    "ABT": {"id": "ABT", "name": "Abbott Laboratories", "sector": "Healthcare"},
    "CAT": {"id": "CAT", "name": "Caterpillar Inc.", "sector": "Industrials"},
    "NOW": {"id": "NOW", "name": "ServiceNow Inc.", "sector": "Information Technology"},
    "INTU": {"id": "INTU", "name": "Intuit Inc.", "sector": "Information Technology"},
    "GE": {"id": "GE", "name": "General Electric Company", "sector": "Industrials"},
    "VZ": {"id": "VZ", "name": "Verizon Communications", "sector": "Communication Services"},
    "AMGN": {"id": "AMGN", "name": "Amgen Inc.", "sector": "Healthcare"},
    "GS": {"id": "GS", "name": "The Goldman Sachs Group", "sector": "Financials"},
    "CMCSA": {"id": "CMCSA", "name": "Comcast Corporation", "sector": "Communication Services"},
    "UBER": {"id": "UBER", "name": "Uber Technologies Inc.", "sector": "Technology"},
    "DHR": {"id": "DHR", "name": "Danaher Corporation", "sector": "Healthcare"},
    "MU": {"id": "MU", "name": "Micron Technology Inc.", "sector": "Semiconductors"},
    "ISRG": {"id": "ISRG", "name": "Intuitive Surgical Inc.", "sector": "Healthcare"},
    "BKNG": {"id": "BKNG", "name": "Booking Holdings Inc.", "sector": "Consumer Discretionary"},
    "RTX": {"id": "RTX", "name": "RTX Corporation", "sector": "Industrials"},
    "PLTR": {"id": "PLTR", "name": "Palantir Technologies", "sector": "Information Technology"},
    "SPGI": {"id": "SPGI", "name": "S&P Global Inc.", "sector": "Financials"},
    "HON": {"id": "HON", "name": "Honeywell International", "sector": "Industrials"},
    "UNP": {"id": "UNP", "name": "Union Pacific Corporation", "sector": "Industrials"},
    "LOW": {"id": "LOW", "name": "Lowe's Companies Inc.", "sector": "Consumer Discretionary"},
    "SYK": {"id": "SYK", "name": "Stryker Corporation", "sector": "Healthcare"},
    "LRCX": {"id": "LRCX", "name": "Lam Research Corporation", "sector": "Semiconductors"},
    "COP": {"id": "COP", "name": "ConocoPhillips", "sector": "Energy"},
    "ELV": {"id": "ELV", "name": "Elevance Health Inc.", "sector": "Healthcare"},
    "SCHW": {"id": "SCHW", "name": "The Charles Schwab Corp", "sector": "Financials"},
    "BA": {"id": "BA", "name": "The Boeing Company", "sector": "Industrials"},
    "DE": {"id": "DE", "name": "Deere & Company", "sector": "Industrials"},
    "BLK": {"id": "BLK", "name": "BlackRock Inc.", "sector": "Financials"},
    "TJX": {"id": "TJX", "name": "The TJX Companies Inc.", "sector": "Consumer Discretionary"},
    "PANW": {"id": "PANW", "name": "Palo Alto Networks", "sector": "Information Technology"},
    "KLAC": {"id": "KLAC", "name": "KLA Corporation", "sector": "Semiconductors"},
    "PGR": {"id": "PGR", "name": "The Progressive Corp", "sector": "Financials"},
    "MDT": {"id": "MDT", "name": "Medtronic plc", "sector": "Healthcare"},
    "CI": {"id": "CI", "name": "The Cigna Group", "sector": "Healthcare"},
    "BMY": {"id": "BMY", "name": "Bristol-Myers Squibb", "sector": "Healthcare"},
    "ADI": {"id": "ADI", "name": "Analog Devices Inc.", "sector": "Semiconductors"},
    "GILD": {"id": "GILD", "name": "Gilead Sciences Inc.", "sector": "Healthcare"},
    "SBUX": {"id": "SBUX", "name": "Starbucks Corporation", "sector": "Consumer Discretionary"},
    "VRTX": {"id": "VRTX", "name": "Vertex Pharmaceuticals", "sector": "Healthcare"},
    "C": {"id": "C", "name": "Citigroup Inc.", "sector": "Financials"},
    "SNPS": {"id": "SNPS", "name": "Synopsys Inc.", "sector": "Information Technology"},
    "CDNS": {"id": "CDNS", "name": "Cadence Design Systems", "sector": "Information Technology"},
    "MDLZ": {"id": "MDLZ", "name": "Mondelez International", "sector": "Consumer Staples"},
    "MMC": {"id": "MMC", "name": "Marsh & McLennan Companies", "sector": "Financials"},
    "CB": {"id": "CB", "name": "Chubb Limited", "sector": "Financials"},
    "REGN": {"id": "REGN", "name": "Regeneron Pharmaceuticals", "sector": "Healthcare"},
    "FI": {"id": "FI", "name": "Fiserv Inc.", "sector": "Financials"},
    "CVS": {"id": "CVS", "name": "CVS Health Corporation", "sector": "Healthcare"},
    "AMT": {"id": "AMT", "name": "American Tower Corp.", "sector": "Real Estate"},
    "MO": {"id": "MO", "name": "Altria Group Inc.", "sector": "Consumer Staples"},
    "ADP": {"id": "ADP", "name": "Automatic Data Processing", "sector": "Industrials"},
    "CRWD": {"id": "CRWD", "name": "CrowdStrike Holdings", "sector": "Information Technology"},
    "LMT": {"id": "LMT", "name": "Lockheed Martin Corp.", "sector": "Industrials"},
    "SO": {"id": "SO", "name": "The Southern Company", "sector": "Utilities"},
    "DUK": {"id": "DUK", "name": "Duke Energy Corporation", "sector": "Utilities"},
    "BSX": {"id": "BSX", "name": "Boston Scientific Corp.", "sector": "Healthcare"},
    "T": {"id": "T", "name": "AT&T Inc.", "sector": "Communication Services"},
    "EQIX": {"id": "EQIX", "name": "Equinix Inc.", "sector": "Real Estate"},
    "SHW": {"id": "SHW", "name": "The Sherwin-Williams Co", "sector": "Materials"},
    "WM": {"id": "WM", "name": "Waste Management Inc.", "sector": "Industrials"},
    "ICE": {"id": "ICE", "name": "Intercontinental Exchange", "sector": "Financials"},
    "ITW": {"id": "ITW", "name": "Illinois Tool Works Inc.", "sector": "Industrials"},
    "EOG": {"id": "EOG", "name": "EOG Resources Inc.", "sector": "Energy"},
    "CSX": {"id": "CSX", "name": "CSX Corporation", "sector": "Industrials"},
    "SLB": {"id": "SLB", "name": "Schlumberger Limited", "sector": "Energy"},
    "MAR": {"id": "MAR", "name": "Marriott International", "sector": "Consumer Discretionary"},
    "CMG": {"id": "CMG", "name": "Chipotle Mexican Grill", "sector": "Consumer Discretionary"},
    "CL": {"id": "CL", "name": "Colgate-Palmolive Co", "sector": "Consumer Staples"},
    "NOC": {"id": "NOC", "name": "Northrop Grumman Corp.", "sector": "Industrials"},
    "PYPL": {"id": "PYPL", "name": "PayPal Holdings Inc.", "sector": "Financials"},
    "FDX": {"id": "FDX", "name": "FedEx Corporation", "sector": "Industrials"},
    "ORLY": {"id": "ORLY", "name": "O'Reilly Automotive", "sector": "Consumer Discretionary"},
    "NXPI": {"id": "NXPI", "name": "NXP Semiconductors N.V.", "sector": "Semiconductors"},
    "GD": {"id": "GD", "name": "General Dynamics Corp.", "sector": "Industrials"},
    "PXD": {"id": "PXD", "name": "Pioneer Natural Resources", "sector": "Energy"},
    "MCK": {"id": "MCK", "name": "McKesson Corporation", "sector": "Healthcare"},
    "CTAS": {"id": "CTAS", "name": "Cintas Corporation", "sector": "Industrials"},
    "COF": {"id": "COF", "name": "Capital One Financial", "sector": "Financials"},
    "AON": {"id": "AON", "name": "Aon plc", "sector": "Financials"},
    "EMR": {"id": "EMR", "name": "Emerson Electric Co.", "sector": "Industrials"},
    "PNC": {"id": "PNC", "name": "PNC Financial Services", "sector": "Financials"},
    "NSC": {"id": "NSC", "name": "Norfolk Southern Corp.", "sector": "Industrials"},
    "MCO": {"id": "MCO", "name": "Moody's Corporation", "sector": "Financials"},
    "ROP": {"id": "ROP", "name": "Roper Technologies Inc.", "sector": "Information Technology"},
    "APD": {"id": "APD", "name": "Air Products and Chemicals", "sector": "Materials"},
    "FCX": {"id": "FCX", "name": "Freeport-McMoRan Inc.", "sector": "Materials"},
    "AJG": {"id": "AJG", "name": "Arthur J. Gallagher & Co", "sector": "Financials"},
    "ECL": {"id": "ECL", "name": "Ecolab Inc.", "sector": "Materials"},
    "AZO": {"id": "AZO", "name": "AutoZone Inc.", "sector": "Consumer Discretionary"},
    "OXY": {"id": "OXY", "name": "Occidental Petroleum", "sector": "Energy"},
    "HUM": {"id": "HUM", "name": "Humana Inc.", "sector": "Healthcare"},
    "MSI": {"id": "MSI", "name": "Motorola Solutions Inc.", "sector": "Information Technology"},
    "BDX": {"id": "BDX", "name": "Becton Dickinson and Co", "sector": "Healthcare"},
    "ADSK": {"id": "ADSK", "name": "Autodesk Inc.", "sector": "Information Technology"},
    "COHR": {"id": "COHR", "name": "Coherent Corp.", "sector": "Information Technology"},
    "SMCI": {"id": "SMCI", "name": "Super Micro Computer", "sector": "Information Technology"},
    "ARM": {"id": "ARM", "name": "Arm Holdings plc", "sector": "Semiconductors"},
    "DELL": {"id": "DELL", "name": "Dell Technologies Inc.", "sector": "Information Technology"},
    "COIN": {"id": "COIN", "name": "Coinbase Global Inc.", "sector": "Financials"},
    "HOOD": {"id": "HOOD", "name": "Robinhood Markets Inc.", "sector": "Financials"},
    "MSTR": {"id": "MSTR", "name": "MicroStrategy Inc.", "sector": "Information Technology"},
    "SNOW": {"id": "SNOW", "name": "Snowflake Inc.", "sector": "Information Technology"},
    "DDOG": {"id": "DDOG", "name": "Datadog Inc.", "sector": "Information Technology"},
    "NET": {"id": "NET", "name": "Cloudflare Inc.", "sector": "Information Technology"},
    "ZS": {"id": "ZS", "name": "Zscaler Inc.", "sector": "Information Technology"},
    "WDAY": {"id": "WDAY", "name": "Workday Inc.", "sector": "Information Technology"},
    "TEAM": {"id": "TEAM", "name": "Atlassian Corporation", "sector": "Information Technology"},
    "MDB": {"id": "MDB", "name": "MongoDB Inc.", "sector": "Information Technology"},
    "SHOP": {"id": "SHOP", "name": "Shopify Inc.", "sector": "Information Technology"},
    "SQ": {"id": "SQ", "name": "Block Inc.", "sector": "Financials"},
    "ABNB": {"id": "ABNB", "name": "Airbnb Inc.", "sector": "Consumer Discretionary"},
    "RBLX": {"id": "RBLX", "name": "Roblox Corporation", "sector": "Communication Services"},
    "PATH": {"id": "PATH", "name": "UiPath Inc.", "sector": "Information Technology"},
    "AFRM": {"id": "AFRM", "name": "Affirm Holdings Inc.", "sector": "Financials"},
    "UPST": {"id": "UPST", "name": "Upstart Holdings Inc.", "sector": "Financials"},
    "SOFI": {"id": "SOFI", "name": "SoFi Technologies Inc.", "sector": "Financials"},
    "RIVN": {"id": "RIVN", "name": "Rivian Automotive Inc.", "sector": "Auto"},
    "LCID": {"id": "LCID", "name": "Lucid Group Inc.", "sector": "Auto"},
    "NIO": {"id": "NIO", "name": "NIO Inc.", "sector": "Auto"},
    "LI": {"id": "LI", "name": "Li Auto Inc.", "sector": "Auto"},
    "XPEV": {"id": "XPEV", "name": "XPeng Inc.", "sector": "Auto"},
    "BABA": {"id": "BABA", "name": "Alibaba Group Holding", "sector": "Consumer Discretionary"},
    "PDD": {"id": "PDD", "name": "PDD Holdings Inc.", "sector": "Consumer Discretionary"},
    "JD": {"id": "JD", "name": "JD.com Inc.", "sector": "Consumer Discretionary"},
    "BIDU": {"id": "BIDU", "name": "Baidu Inc.", "sector": "Communication Services"},
    "NTES": {"id": "NTES", "name": "NetEase Inc.", "sector": "Communication Services"},
    "SE": {"id": "SE", "name": "Sea Limited", "sector": "Consumer Discretionary"},
    "GRAB": {"id": "GRAB", "name": "Grab Holdings Ltd", "sector": "Technology"},
    "MELI": {"id": "MELI", "name": "MercadoLibre Inc.", "sector": "Consumer Discretionary"},
    "NU": {"id": "NU", "name": "Nu Holdings Ltd.", "sector": "Financials"},
}


# ==============================================================================
# 2. STATE MANAGEMENT & SESSION LEDGER INITIALIZATION
# ==============================================================================
def init_session_state() -> None:
    """Initialize persistent in-memory variables across reruns."""
    defaults = {
        "authenticated": False,
        "circuit_broken": False,
        "live_trading": False,
        "starting_ledger": 1_000_000.00,  # 10 Lakhs / $1M baseline
        "available_cash": 1_000_000.00,
        "utilized_margin": 0.00,
        "realized_pnl": 0.00,
        "paper_positions": {},      # symbol -> {shares, entry_price, market, type, atr, target, sl}
        "order_book": [],           # List of executed orders
        "logs": [],                 # Execution terminal logs
        "scanner_cache": {},        # Cached indicator dataframes
        "last_scan_time": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

def log_event(msg: str, level: str = "INFO") -> None:
    """Append structured log to session history and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    entry = f"[{timestamp}] [{level.upper()}] {msg}"
    st.session_state.logs.append(entry)
    if len(st.session_state.logs) > 1000:
        st.session_state.logs.pop(0)


# ==============================================================================
# 3. AUTHENTICATION & SECURITY GATE
# ==============================================================================
def verify_credentials(password_input: str) -> bool:
    """Secure SHA-256 hash comparison."""
    app_pwd = st.secrets.get("APP_PASSWORD", "admin123")
    target_hash = hashlib.sha256(app_pwd.encode()).hexdigest()
    input_hash = hashlib.sha256(password_input.encode()).hexdigest()
    return target_hash == input_hash

if not st.session_state.authenticated:
    st.title("🔒 APEX QUANTITATIVE PLATFORM ACCESS")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("auth_form"):
            st.markdown("### Authentication Gate")
            pwd = st.text_input("Enter Execution Master Password", type="password")
            submitted = st.form_submit_button("Authenticate System", use_container_width=True)
            if submitted:
                if verify_credentials(pwd):
                    st.session_state.authenticated = True
                    log_event("User authenticated successfully.", "AUTH")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Access Denied.")
                    log_event("Unauthorized login attempt detected.", "WARN")
    st.stop()


# ==============================================================================
# 4. DHANHQ API WRAPPER & BROKER ENGINE
# ==============================================================================
class DhanBrokerEngine:
    """Production-grade broker connector with transparent paper fallback."""

    def __init__(self):
        self.client_id = st.secrets.get("DHAN_CLIENT_ID", os.getenv("DHAN_CLIENT_ID", ""))
        self.access_token = st.secrets.get("DHAN_ACCESS_TOKEN", os.getenv("DHAN_ACCESS_TOKEN", ""))
        self.client = None
        self.connected = False

        if DHAN_AVAILABLE and self.client_id and self.access_token:
            try:
                self.client = dhanhq(self.client_id, self.access_token)
                fund_res = self.client.get_fund_limits()
                if fund_res.get("status") == "success":
                    self.connected = True
                    log_event("Connected to DhanHQ API v2 Live Broker Gateway", "BROKER")
            except Exception as e:
                log_event(f"DhanHQ connection initialization failed: {e}", "WARN")

    def get_wallet_limits(self) -> Dict[str, float]:
        """Fetch live Dhan wallet or fall back to internal simulated ledger."""
        if self.connected and st.session_state.live_trading:
            try:
                res = self.client.get_fund_limits()
                if res.get("status") == "success":
                    data = res.get("data", {})
                    avail = float(data.get("availabelBalance", 0.0))
                    utilized = float(data.get("utilizedAmount", 0.0))
                    return {"available": avail, "utilized": utilized}
            except Exception as e:
                log_event(f"Failed to fetch Dhan limits: {e}", "ERROR")

        return {
            "available": float(st.session_state.available_cash),
            "utilized": float(st.session_state.utilized_margin)
        }

    def place_bracket_order(
        self,
        symbol: str,
        security_id: str,
        qty: int,
        side: str,
        target_price: float,
        stop_loss: float,
        ltp: float
    ) -> Dict[str, Any]:
        """Execute DhanHQ Super/Bracket Order or fallback Paper Order."""
        if qty <= 0:
            return {"status": "error", "message": "Order quantity must be positive."}

        order_cost = qty * ltp
        if order_cost > st.session_state.available_cash:
            return {"status": "error", "message": "Insufficient margin available."}

        if self.connected and st.session_state.live_trading:
            try:
                dhan_side = self.client.BUY if side == "BUY" else self.client.SELL
                res = self.client.place_order(
                    security_id=security_id,
                    exchange_segment=self.client.NSE,
                    transaction_type=dhan_side,
                    quantity=qty,
                    order_type=self.client.LIMIT,
                    product_type=self.client.BO,
                    price=ltp,
                    trigger_price=0.0,
                    target_price=target_price,
                    stop_loss_price=stop_loss
                )
                log_event(f"Live DhanHQ BO Placed: {symbol} Qty:{qty} @ {ltp}", "ORDER")
                return {"status": "success", "data": res}
            except Exception as e:
                log_event(f"Live execution failed: {e}. Defaulting to paper ledger.", "ERROR")

        # Paper Execution Logic
        st.session_state.available_cash -= order_cost
        st.session_state.utilized_margin += order_cost

        st.session_state.paper_positions[symbol] = {
            "shares": qty if side == "BUY" else -qty,
            "entry_price": ltp,
            "current_price": ltp,
            "target": target_price,
            "sl": stop_loss,
            "side": side,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

        order_record = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": ltp,
            "target": target_price,
            "sl": stop_loss,
            "type": "PAPER_BRACKET" if not st.session_state.live_trading else "LIVE_BO"
        }
        st.session_state.order_book.insert(0, order_record)
        log_event(f"Order Filled: {side} {qty}x {symbol} @ {ltp:.2f} (SL: {stop_loss:.2f}, TP: {target_price:.2f})", "EXEC")
        return {"status": "success", "data": order_record}

    def close_position(self, symbol: str, current_ltp: float) -> bool:
        """Square off a specific position."""
        if symbol not in st.session_state.paper_positions:
            return False

        pos = st.session_state.paper_positions.pop(symbol)
        shares = pos["shares"]
        entry = pos["entry_price"]
        pnl = (current_ltp - entry) * shares

        released_capital = abs(shares) * entry
        st.session_state.available_cash += (released_capital + pnl)
        st.session_state.utilized_margin = max(0.0, st.session_state.utilized_margin - released_capital)
        st.session_state.realized_pnl += pnl

        order_record = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "symbol": symbol,
            "side": "SQUARE_OFF",
            "qty": abs(shares),
            "price": current_ltp,
            "pnl": pnl,
            "type": "CLOSE"
        }
        st.session_state.order_book.insert(0, order_record)
        log_event(f"Position Closed: {symbol} @ {current_ltp:.2f} | Realized P&L: {pnl:+.2f}", "SQUAREOFF")
        return True

    def panic_kill_switch(self) -> int:
        """Cancel all orders and immediately close all open positions."""
        closed_count = 0
        symbols = list(st.session_state.paper_positions.keys())
        for sym in symbols:
            pos = st.session_state.paper_positions[sym]
            self.close_position(sym, pos["current_price"])
            closed_count += 1

        if self.connected and st.session_state.live_trading:
            try:
                self.client.cancel_all_orders()
                log_event("Live broker: all open orders cancelled.", "EMERGENCY")
            except Exception as e:
                log_event(f"Live broker emergency cancel failure: {e}", "CRITICAL")

        st.session_state.circuit_broken = True
        log_event(f"PANIC KILL SWITCH TRIGGERED: {closed_count} positions flattened.", "CRITICAL")
        return closed_count

broker = DhanBrokerEngine()


# ==============================================================================
# 5. QUANTITATIVE TECHNICAL ENGINE & SIGNAL RADAR
# ==============================================================================
class QuantitativeEngine:
    """Vectorized mathematical technical indicator calculations."""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate complete technical suite on an OHLCV dataframe."""
        if df.empty or len(df) < 35:
            return df

        df = df.copy()

        # Simple Moving Averages
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # Relative Strength Index (RSI 14)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        # Bollinger Bands (20, 2)
        df["BB_MID"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_UPPER"] = df["BB_MID"] + (2 * bb_std)
        df["BB_LOWER"] = df["BB_MID"] - (2 * bb_std)

        # Average True Range (ATR 14)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()

        # Stochastic Oscillator (%K 14, %D 3)
        low_min = df["Low"].rolling(window=14).min()
        high_max = df["High"].rolling(window=14).max()
        df["STOCH_K"] = 100 * ((df["Close"] - low_min) / ((high_max - low_min) + 1e-9))
        df["STOCH_D"] = df["STOCH_K"].rolling(window=3).mean()

        # Volume Surge Factor
        df["VOL_SMA20"] = df["Volume"].rolling(window=20).mean()
        df["VOL_SURGE"] = df["Volume"] / (df["VOL_SMA20"] + 1e-9)

        return df

    @staticmethod
    def score_signal(row: pd.Series) -> Tuple[float, str, List[str]]:
        """Calculate multi-condition quantitative score from 0.0 to 100.0%."""
        score = 50.0
        reasons = []

        if pd.isna(row.get("RSI")) or pd.isna(row.get("ATR")):
            return 50.0, "NEUTRAL", ["Insufficient Data"]

        # RSI Evaluation
        if row["RSI"] < 30:
            score += 20
            reasons.append("RSI Oversold (<30)")
        elif row["RSI"] > 70:
            score -= 20
            reasons.append("RSI Overbought (>70)")

        # MACD Crossover
        if row["MACD"] > row["MACD_SIGNAL"] and row["MACD_HIST"] > 0:
            score += 15
            reasons.append("MACD Bullish Momentum")
        elif row["MACD"] < row["MACD_SIGNAL"]:
            score -= 15
            reasons.append("MACD Bearish Momentum")

        # Bollinger Band Mean Reversion
        if row["Close"] <= row["BB_LOWER"]:
            score += 15
            reasons.append("Price at Lower Bollinger Band")
        elif row["Close"] >= row["BB_UPPER"]:
            score -= 15
            reasons.append("Price at Upper Bollinger Band")

        # Volume Surge
        if row.get("VOL_SURGE", 1.0) > 1.5:
            if score > 50:
                score += 10
                reasons.append("Bullish Volume Breakout (>1.5x avg)")
            else:
                score -= 10
                reasons.append("Bearish Volume Distribution")

        # Moving Average Trend Alignment
        if row["Close"] > row.get("SMA_50", row["Close"]):
            score += 5
            reasons.append("Above 50 SMA")
        else:
            score -= 5
            reasons.append("Below 50 SMA")

        score = float(np.clip(score, 0.0, 100.0))

        if score >= 65.0:
            bias = "STRONG_BUY" if score >= 80 else "BUY"
        elif score <= 35.0:
            bias = "STRONG_SELL" if score <= 20 else "SELL"
        else:
            bias = "NEUTRAL"

        return score, bias, reasons


# ==============================================================================
# 6. MARKET DATA INGESTION (YFINANCE ENGINE)
# ==============================================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ticker_data(symbol: str, market: str, period: str = "6mo") -> pd.DataFrame:
    """Ingest market OHLCV with formatted ticker suffix."""
    yf_symbol = f"{symbol}.NS" if market == "NSE" else symbol
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval="1d")
        if not df.empty:
            df = QuantitativeEngine.calculate_indicators(df)
            return df
    except Exception as e:
        log_event(f"Data ingestion error for {yf_symbol}: {e}", "ERROR")
    return pd.DataFrame()


# ==============================================================================
# 7. LIVE NEWS DESK & SENTIMENT SCORING ENGINE
# ==============================================================================
class NewsDeskEngine:
    """Fetch live ticker news and compute lexicon sentiment polarity."""

    BULLISH_WORDS = {
        "surge", "surges", "surged", "jump", "jumps", "rally", "rallies", "profit",
        "beat", "beats", "growth", "high", "upgrade", "outperform", "bull", "bullish",
        "gain", "gains", "record", "expansion", "positive", "breakout", "boost"
    }
    BEARISH_WORDS = {
        "fall", "falls", "fell", "plunge", "plunges", "drop", "drops", "loss",
        "miss", "misses", "down", "downgrade", "underperform", "bear", "bearish",
        "fraud", "investigation", "slump", "negative", "crash", "decline", "warning"
    }

    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def get_ticker_news(ticker: str, market: str) -> List[Dict[str, Any]]:
        """Fetch RSS feed from Google News and score sentiment."""
        query = f"{ticker} stock {('NSE India' if market == 'NSE' else 'US market')}"
        encoded_query = query.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        articles = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = entry.title
                summary = entry.get("summary", "")
                text_to_score = f"{title} {summary}".lower()

                # Calculate polarity
                words = set(text_to_score.split())
                pos_count = len(words.intersection(NewsDeskEngine.BULLISH_WORDS))
                neg_count = len(words.intersection(NewsDeskEngine.BEARISH_WORDS))
                total = pos_count + neg_count
                polarity = (pos_count - neg_count) / total if total > 0 else 0.0

                published = entry.get("published", datetime.now().strftime("%a, %d %b %Y"))

                articles.append({
                    "title": title,
                    "link": entry.link,
                    "published": published,
                    "polarity": polarity,
                    "source": entry.get("source", {}).get("title", "Market Wire")
                })
        except Exception as e:
            log_event(f"News fetch error for {ticker}: {e}", "WARN")

        return articles


# ==============================================================================
# 8. RISK SIZING ENGINE & PORTFOLIO CIRCUIT BREAKER
# ==============================================================================
class RiskEngine:
    """Calculate sizing limits and monitor systemic portfolio exposure."""

    @staticmethod
    def calculate_position_size(
        account_size: float,
        risk_pct: float,
        ltp: float,
        stop_loss_distance: float,
        max_capital_pct: float
    ) -> int:
        """Fixed fractional risk sizing bounded by max capital constraints."""
        if stop_loss_distance <= 0 or ltp <= 0:
            return 0

        # Maximum risk currency amount
        risk_capital = account_size * (risk_pct / 100.0)
        shares_by_risk = int(risk_capital / stop_loss_distance)

        # Max allocation boundary
        max_position_cost = account_size * (max_capital_pct / 100.0)
        shares_by_cap = int(max_position_cost / ltp)

        final_shares = max(1, min(shares_by_risk, shares_by_cap))
        return final_shares

    @staticmethod
    def evaluate_circuit_breaker(starting_balance: float, current_net_balance: float) -> bool:
        """Halt execution if net drawdown hits 3.0% limit."""
        drawdown_pct = ((starting_balance - current_net_balance) / starting_balance) * 100.0
        if drawdown_pct >= 3.0:
            return True
        return False


# ==============================================================================
# 9. REAL-TIME MTM & WALLET CALCULATIONS
# ==============================================================================
def compute_live_portfolio_metrics() -> Dict[str, float]:
    """Compute instant Mark-To-Market across all open holdings."""
    unrealized_pnl = 0.0
    deployed_margin = 0.0

    for sym, pos in list(st.session_state.paper_positions.items()):
        # Quick price probe
        df_tick = fetch_ticker_data(sym, "NSE" if sym in NSE_EQUITIES else "US", period="5d")
        if not df_tick.empty:
            curr_price = float(df_tick["Close"].iloc[-1])
            pos["current_price"] = curr_price
        else:
            curr_price = pos["current_price"]

        shares = pos["shares"]
        entry = pos["entry_price"]
        position_pnl = (curr_price - entry) * shares
        unrealized_pnl += position_pnl
        deployed_margin += abs(shares) * entry

    st.session_state.utilized_margin = deployed_margin
    total_day_pnl = st.session_state.realized_pnl + unrealized_pnl
    net_equity = st.session_state.available_cash + deployed_margin + unrealized_pnl

    # Check Circuit Breaker
    if RiskEngine.evaluate_circuit_breaker(st.session_state.starting_ledger, net_equity):
        if not st.session_state.circuit_broken:
            log_event("CIRCUIT BREAKER TRIGGERED: Daily portfolio drawdown exceeds 3.0%.", "CRITICAL")
            broker.panic_kill_switch()

    risk_exposure_ratio = (deployed_margin / (st.session_state.starting_ledger + 1e-9)) * 100.0

    return {
        "available": st.session_state.available_cash,
        "utilized": deployed_margin,
        "realized": st.session_state.realized_pnl,
        "unrealized": unrealized_pnl,
        "net_pnl": total_day_pnl,
        "net_equity": net_equity,
        "risk_ratio": risk_exposure_ratio
    }

portfolio_metrics = compute_live_portfolio_metrics()


# ==============================================================================
# 10. SIDEBAR - RISK & EXECUTION PARAMETERS
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Risk Configuration")

    st.session_state.live_trading = st.toggle("🔴 Live DhanHQ Execution", value=False)
    if st.session_state.live_trading:
        if not broker.connected:
            st.warning("⚠️ Live API not connected. Running in Paper Mode.")
        else:
            st.success("✅ Connected to DhanHQ Live Gateway")
    else:
        st.info("📝 Paper-Trading Simulator Active")

    st.markdown("---")
    st.markdown("### Risk Parameters")
    risk_per_trade_pct = st.slider("Risk Per Trade (%)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    atr_sl_mult = st.slider("Stop-Loss ATR Multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    atr_tp_mult = st.slider("Target ATR Multiplier", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    max_capital_per_stock = st.slider("Max Allocation / Stock (%)", min_value=5.0, max_value=50.0, value=15.0, step=5.0)

    st.markdown("---")
    st.markdown("### Universe & Filtering")
    market_selection = st.selectbox("Active Market Universe", ["Combined Watchlist", "NSE India Equities", "US Equities"])

    sector_filter = st.selectbox(
        "Sector Filter",
        ["All Sectors"] + sorted(list({
            d["sector"] for d in (
                list(NSE_EQUITIES.values()) if market_selection == "NSE India Equities"
                else list(US_EQUITIES.values()) if market_selection == "US Equities"
                else list(NSE_EQUITIES.values()) + list(US_EQUITIES.values())
            )
        }))
    )

    st.markdown("---")
    if st.button("🚨 EMERGENCY KILL SWITCH", type="primary", use_container_width=True):
        count = broker.panic_kill_switch()
        st.error(f"Kill Switch Activated: {count} Positions Flattened.")
        st.rerun()

    if st.session_state.circuit_broken:
        st.error("🛑 CIRCUIT BREAKER TRIGGERED - TRADING LOCKED")
        if st.button("Reset Circuit Breaker"):
            st.session_state.circuit_broken = False
            log_event("Circuit breaker manually reset by administrator.", "AUTH")
            st.rerun()


# ==============================================================================
# 11. TOP HEADER: FINANCIAL LEDGER & WALLET HUD
# ==============================================================================
st.markdown("## ⚡ APEX QUANTITATIVE TRADING SUITE")

hud_c1, hud_c2, hud_c3, hud_c4, hud_c5, hud_c6 = st.columns(6)

hud_c1.metric(
    "Available Cash",
    f"₹{portfolio_metrics['available']:,.2f}" if market_selection == "NSE India Equities" else f"${portfolio_metrics['available']:,.2f}",
    help="Free margin ready for deployment"
)
hud_c2.metric(
    "Deployed Margin",
    f"₹{portfolio_metrics['utilized']:,.2f}" if market_selection == "NSE India Equities" else f"${portfolio_metrics['utilized']:,.2f}",
    help="Total capital currently engaged in active trades"
)
hud_c3.metric(
    "Realized P&L",
    f"{portfolio_metrics['realized']:+,.2f}",
    delta=f"{portfolio_metrics['realized']:+,.2f}",
    help="Locked-in gains/losses from closed trades today"
)
hud_c4.metric(
    "Unrealized MTM",
    f"{portfolio_metrics['unrealized']:+,.2f}",
    delta=f"{portfolio_metrics['unrealized']:+,.2f}",
    help="Floating mark-to-market across open inventory"
)
hud_c5.metric(
    "Net Day P&L",
    f"{portfolio_metrics['net_pnl']:+,.2f}",
    delta=f"{portfolio_metrics['net_pnl']:+,.2f}",
    help="Combined Realized + Mark-to-Market Day Result"
)
hud_c6.metric(
    "Risk Exposure",
    f"{portfolio_metrics['risk_ratio']:.1f}%",
    delta=f"{(portfolio_metrics['risk_ratio'] - 100):.1f}%" if portfolio_metrics['risk_ratio'] > 100 else "Nominal",
    delta_color="inverse"
)

st.markdown("---")


# ==============================================================================
# 12. MAIN NAVIGATION TABS
# ==============================================================================
tab_scanner, tab_portfolio, tab_news, tab_cli = st.tabs([
    "📊 Quantitative Radar & Scanner",
    "💼 Positions & Order Ledger",
    "📰 Live News & Sentiment Desk",
    "💻 Interactive Shell & CLI"
])


# ------------------------------------------------------------------------------
# TAB 1: QUANTITATIVE SIGNAL SCANNER
# ------------------------------------------------------------------------------
with tab_scanner:
    st.subheader("Automated Quantitative Radar")

    # Determine Active Universe
    active_pool = {}
    if market_selection in ("NSE India Equities", "Combined Watchlist"):
        for k, v in NSE_EQUITIES.items():
            active_pool[k] = {**v, "market": "NSE"}
    if market_selection in ("US Equities", "Combined Watchlist"):
        for k, v in US_EQUITIES.items():
            active_pool[k] = {**v, "market": "US"}

    if sector_filter != "All Sectors":
        active_pool = {k: v for k, v in active_pool.items() if v.get("sector") == sector_filter}

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_query = st.text_input("🔍 Filter Ticker by Symbol or Company Name", "").upper()
    with col_s2:
        radar_scan_limit = st.slider("Scan Depth", min_value=5, max_value=min(60, len(active_pool)), value=15)

    filtered_symbols = [
        s for s, d in active_pool.items()
        if search_query in s or search_query in d["name"].upper()
    ][:radar_scan_limit]

    scanner_rows = []
    with st.spinner("Calculating quantitative matrices across selected tickers..."):
        for sym in filtered_symbols:
            meta = active_pool[sym]
            df_sym = fetch_ticker_data(sym, meta["market"], period="6mo")
            if df_sym.empty:
                continue

            last_row = df_sym.iloc[-1]
            score, bias, reasons = QuantitativeEngine.score_signal(last_row)
            ltp = float(last_row["Close"])
            atr = float(last_row.get("ATR", ltp * 0.02))

            upper_target = ltp + (atr * atr_tp_mult)
            lower_sl = ltp - (atr * atr_sl_mult)

            scanner_rows.append({
                "Symbol": sym,
                "Company": meta["name"],
                "Market": meta["market"],
                "Sector": meta["sector"],
                "LTP": round(ltp, 2),
                "RSI (14)": round(last_row.get("RSI", 50.0), 1),
                "MACD": round(last_row.get("MACD", 0.0), 2),
                "ATR": round(atr, 2),
                "Target (+ATR)": round(upper_target, 2),
                "Stop Loss (-ATR)": round(lower_sl, 2),
                "Score": score,
                "Signal": bias,
                "Drivers": ", ".join(reasons[:2])
            })

    if scanner_rows:
        scan_df = pd.DataFrame(scanner_rows)

        # Style Dataframe with clean visual indicators
        def highlight_signal(val):
            if "BUY" in str(val):
                return "color: #00E676; font-weight: bold;"
            elif "SELL" in str(val):
                return "color: #FF5252; font-weight: bold;"
            return "color: #B0BEC5;"

        st.dataframe(
            scan_df.style.applymap(highlight_signal, subset=["Signal"]),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.subheader("⚡ Instant Quantitative Execution Terminal")

        exec_c1, exec_c2, exec_c3, exec_c4, exec_c5 = st.columns([2, 1, 1, 1, 1])
        with exec_c1:
            selected_trade_sym = st.selectbox("Select Target Ticker", scan_df["Symbol"].tolist())
        with exec_c2:
            trade_side = st.selectbox("Action", ["BUY", "SELL"])

        target_row = scan_df[scan_df["Symbol"] == selected_trade_sym].iloc[0]
        sel_ltp = float(target_row["LTP"])
        sel_atr = float(target_row["ATR"])
        sel_target = float(target_row["Target (+ATR)"])
        sel_sl = float(target_row["Stop Loss (-ATR)"])

        calc_qty = RiskEngine.calculate_position_size(
            st.session_state.available_cash,
            risk_per_trade_pct,
            sel_ltp,
            abs(sel_ltp - sel_sl),
            max_capital_per_stock
        )

        with exec_c3:
            trade_qty = st.number_input("Calculated Quantity", min_value=1, value=max(1, calc_qty))
        with exec_c4:
            st.metric("Total Cost", f"{sel_ltp * trade_qty:,.2f}")
        with exec_c5:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button(f"⚡ Execute {trade_side}", type="primary", use_container_width=True):
                if st.session_state.circuit_broken:
                    st.error("Circuit breaker active. Trading disabled.")
                else:
                    sec_id = NSE_EQUITIES.get(selected_trade_sym, {}).get("id", "0")
                    res = broker.place_bracket_order(
                        symbol=selected_trade_sym,
                        security_id=sec_id,
                        qty=trade_qty,
                        side=trade_side,
                        target_price=sel_target,
                        stop_loss=sel_sl,
                        ltp=sel_ltp
                    )
                    if res["status"] == "success":
                        st.success(f"Executed {trade_side} for {selected_trade_sym}")
                        st.rerun()
                    else:
                        st.error(res["message"])

        # Technical Chart Inspection
        with st.expander(f"📈 Inspect Technical Chart: {selected_trade_sym}", expanded=False):
            df_chart = fetch_ticker_data(
                selected_trade_sym,
                target_row["Market"],
                period="6mo"
            )
            if not df_chart.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df_chart.index,
                    open=df_chart['Open'],
                    high=df_chart['High'],
                    low=df_chart['Low'],
                    close=df_chart['Close'],
                    name="Price"
                ), row=1, col=1)

                # Overlays
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], line=dict(color='orange', width=1), name="50 SMA"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_UPPER'], line=dict(color='gray', dash='dot'), name="Upper BB"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_LOWER'], line=dict(color='gray', dash='dot'), name="Lower BB"), row=1, col=1)

                # MACD
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['MACD_HIST'], marker_color='teal', name="MACD Hist"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD'], line=dict(color='blue', width=1), name="MACD"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MACD_SIGNAL'], line=dict(color='red', width=1), name="Signal"), row=2, col=1)

                fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0), xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tickers match active filters.")


# ------------------------------------------------------------------------------
# TAB 2: PORTFOLIO & LIVE ORDERS
# ------------------------------------------------------------------------------
with tab_portfolio:
    st.subheader("Active Holdings & Inventory")

    if st.session_state.paper_positions:
        pos_list = []
        for s, p in st.session_state.paper_positions.items():
            cur = p["current_price"]
            ent = p["entry_price"]
            sh = p["shares"]
            pos_pnl = (cur - ent) * sh
            ret_pct = ((cur - ent) / ent) * 100.0 if ent > 0 else 0.0

            pos_list.append({
                "Symbol": s,
                "Position": "LONG" if sh > 0 else "SHORT",
                "Shares": abs(sh),
                "Entry Price": f"{ent:,.2f}",
                "Mark Price": f"{cur:,.2f}",
                "Target (TP)": f"{p['target']:,.2f}",
                "Stop Loss (SL)": f"{p['sl']:,.2f}",
                "Floating P&L": pos_pnl,
                "Return %": f"{ret_pct:+.2f}%",
                "Timestamp": p["timestamp"]
            })

        pos_df = pd.DataFrame(pos_list)
        st.dataframe(
            pos_df.style.applymap(
                lambda v: "color: #00E676; font-weight: bold;" if v > 0 else "color: #FF5252; font-weight: bold;",
                subset=["Floating P&L"]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Manual Position Management")
        sq_col1, sq_col2 = st.columns([3, 1])
        with sq_col1:
            sym_to_close = st.selectbox("Select Open Holding to Liquidate", list(st.session_state.paper_positions.keys()))
        with sq_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Square Off Position", type="primary", use_container_width=True):
                broker.close_position(sym_to_close, st.session_state.paper_positions[sym_to_close]["current_price"])
                st.success(f"Position for {sym_to_close} squared off.")
                st.rerun()
    else:
        st.info("No active open positions. Ledger is clear.")

    st.markdown("---")
    st.subheader("Audit Order Book")
    if st.session_state.order_book:
        st.dataframe(pd.DataFrame(st.session_state.order_book), use_container_width=True, hide_index=True)
    else:
        st.caption("No orders recorded during this trading session.")


# ------------------------------------------------------------------------------
# TAB 3: LIVE NEWS DESK & SENTIMENT
# ------------------------------------------------------------------------------
with tab_news:
    st.subheader("Market Intelligence & Sentiment Radar")

    news_c1, news_c2 = st.columns([2, 1])
    with news_c1:
        target_news_ticker = st.selectbox("Select Asset for Live Intelligence", list(NSE_EQUITIES.keys()) + list(US_EQUITIES.keys()))
    with news_c2:
        mkt_tag = "NSE" if target_news_ticker in NSE_EQUITIES else "US"
        st.markdown(f"**Asset Profile:** `{target_news_ticker}` | Market: `{mkt_tag}`")

    with st.spinner("Ingesting Google News Wire..."):
        articles = NewsDeskEngine.get_ticker_news(target_news_ticker, mkt_tag)

    if articles:
        avg_sentiment = np.mean([a["polarity"] for a in articles])
        sent_color = "#00E676" if avg_sentiment > 0.1 else "#FF5252" if avg_sentiment < -0.1 else "#B0BEC5"
        sent_label = "BULLISH" if avg_sentiment > 0.1 else "BEARISH" if avg_sentiment < -0.1 else "NEUTRAL"

        st.markdown(
            f"<div style='padding: 12px; background-color: #1E1E1E; border-left: 5px solid {sent_color}; border-radius: 4px; margin-bottom: 20px;'>"
            f"<h4 style='margin:0; color: white;'>Aggregate Headline Sentiment: <span style='color:{sent_color};'>{sent_label} ({avg_sentiment:+.2f})</span></h4>"
            f"</div>",
            unsafe_allow_html=True
        )

        for art in articles:
            pol = art["polarity"]
            badge_color = "#00E676" if pol > 0.05 else "#FF5252" if pol < -0.05 else "#78909C"
            tag = "BULLISH" if pol > 0.05 else "BEARISH" if pol < -0.05 else "NEUTRAL"

            st.markdown(
                f"**[{art['title']}]({art['link']})**  \n"
                f"<span style='color: #9E9E9E; font-size: 0.85em;'>Source: {art['source']} | Published: {art['published']}</span> | "
                f"<span style='color: {badge_color}; font-size: 0.85em; font-weight: bold;'>[{tag} {pol:+.2f}]</span>",
                unsafe_allow_html=True
            )
            st.markdown("---")
    else:
        st.info("No recent news found for this security.")


# ------------------------------------------------------------------------------
# TAB 4: INTERACTIVE COMMAND-LINE CONSOLE (CLI)
# ------------------------------------------------------------------------------
with tab_cli:
    st.subheader("Quantitative Operations Console & REPL")
    st.caption("Execute system macros (`wallet`, `positions`, `risk`, `scan`), shell commands (`!pip list`), or raw Python code.")

    cli_input = st.text_input("Enter Command / Macro / Python Code", key="cli_cmd_input")

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        exec_clicked = st.button("Execute", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("Clear Log Stream"):
            st.session_state.logs.clear()
            st.rerun()

    if exec_clicked and cli_input:
        cmd = cli_input.strip()
        log_event(f"> {cmd}", "CLI")

        # 1. Macro Commands
        if cmd == "wallet":
            w = broker.get_wallet_limits()
            log_event(f"WALLET STATUS -> Available: {w['available']:.2f} | Utilized: {w['utilized']:.2f}", "MACRO")
        elif cmd == "positions":
            log_event(f"OPEN POSITIONS ({len(st.session_state.paper_positions)}): {list(st.session_state.paper_positions.keys())}", "MACRO")
        elif cmd == "risk":
            log_event(
                f"RISK AUDIT -> Starting Ledger: {st.session_state.starting_ledger} | "
                f"Net Day PnL: {st.session_state.realized_pnl + portfolio_metrics['unrealized']:.2f} | "
                f"Circuit Status: {'TRIPPED' if st.session_state.circuit_broken else 'NOMINAL'}",
                "MACRO"
            )
        elif cmd.startswith("close --all"):
            c = broker.panic_kill_switch()
            log_event(f"MACRO: Squared off {c} positions.", "MACRO")

        # 2. System Shell Commands
        elif cmd.startswith("!"):
            shell_cmd = cmd[1:]
            try:
                result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, timeout=10)
                out = result.stdout or result.stderr
                log_event(f"SHELL OUTPUT:\n{out.strip()}", "SHELL")
            except Exception as e:
                log_event(f"Shell execution failed: {e}", "ERROR")

        # 3. Python In-Memory REPL
        else:
            stdout_buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec_globals = {
                        "st": st,
                        "pd": pd,
                        "np": np,
                        "broker": broker,
                        "engine": QuantitativeEngine,
                        "session": st.session_state,
                        "positions": st.session_state.paper_positions,
                        "orders": st.session_state.order_book
                    }
                    exec(cmd, exec_globals)
                out = stdout_buf.getvalue()
                if out:
                    log_event(f"REPL OUTPUT:\n{out.strip()}", "PYTHON")
            except Exception as e:
                log_event(f"REPL Error: {e}", "ERROR")

        st.rerun()

    # Log Terminal Screen Output
    terminal_output = "\n".join(st.session_state.logs)
    st.text_area("Console Terminal Stream", value=terminal_output, height=400, disabled=True)

    st.download_button(
        label="📥 Download Audit Logs",
        data=terminal_output,
        file_name=f"apex_quant_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        mime="text/plain"
    )
