#!/usr/bin/env python3
"""
FastAPI Gold Trading Signal System
Automated data pipeline with scheduled updates and REST API endpoints

Deploy on Render with:
- Python 3.9+
- pip install -r requirements.txt
- Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

Author: AI Assistant
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import datetime
import requests
from datetime import datetime, timedelta
import joblib
import time
import json
import warnings
import os
import asyncio
import aiofiles
import uvicorn
from typing import Dict, List, Optional
import logging
from contextlib import asynccontextmanager
import schedule
import threading
from pathlib import Path
import httpx
import subprocess

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

warnings.filterwarnings('ignore')

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required libraries
try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from textblob import TextBlob
    import yfinance as yf
    # Try to import TensorFlow, but make it optional
    try:
        from tensorflow.keras.models import load_model
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        logger.warning("TensorFlow not available. ML predictions will be disabled.")
        TENSORFLOW_AVAILABLE = False
        load_model = None
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    raise

# Global variables
data_pipeline = None
trading_system = None
current_data = None
latest_prediction = None
scheduler_thread = None

# Configuration - Load from environment variables
CONFIG = {
    'data_file': os.getenv('DATA_FILE', 'gold_processed_data.csv'),
    'model_path': 'model.h5',
    'scaler_path': 'scaler.pkl',
    'model_url': os.getenv('MODEL_URL', 'https://github.com/yourusername/yourrepo/releases/download/v1/model.h5'),
    'scaler_url': os.getenv('SCALER_URL', 'https://github.com/yourusername/yourrepo/releases/download/v1/scaler.pkl'),
    'update_interval_hours': int(os.getenv('UPDATE_INTERVAL_HOURS', '1')),
    'seq_len': int(os.getenv('SEQUENCE_LENGTH', '60')),
    'prediction_hours': int(os.getenv('PREDICTION_HOURS', '5')),
    'fallback_mode': os.getenv('FALLBACK_MODE', 'true').lower() == 'true',
    'tensorflow_enabled': os.getenv('TENSORFLOW_ENABLED', 'true').lower() == 'true',
    'economic_data_enabled': os.getenv('ECONOMIC_DATA_ENABLED', 'false').lower() == 'true'
}

class AutomatedDataPipeline:
    """Handles complete data pipeline from fetching to preprocessing"""

    def __init__(self):
        self.gold_data = None
        self.economic_data = None
        self.combined_data = None

    async def fetch_economic_data(self, start_date, end_date):
        """Fetch economic calendar data"""
        if not CONFIG.get('economic_data_enabled', False):
            logger.info("Economic data fetching is disabled")
            return pd.DataFrame()

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

            url = 'https://economic-calendar.tradingview.com/events'
            headers = {'Origin': 'https://in.tradingview.com'}
            payload = {
                'from': start_dt.isoformat() + '.000Z',
                'to': end_dt.isoformat() + '.000Z',
                'countries': ','.join(['US', 'CN', 'IN', 'EU', 'GB', 'RU', 'AU'])
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if 'result' in data and data['result']:
                    econ_df = pd.DataFrame(data['result'])

                    # await this if it's an async function
                    if asyncio.iscoroutinefunction(self.process_economic_data):
                        processed_econ = await self.process_economic_data(econ_df)
                    else:
                        processed_econ = self.process_economic_data(econ_df)

                    print(f"✅ Economic data fetched: {len(processed_econ)} records")
                    self.economic_data = processed_econ
                    return processed_econ
                else:
                    print("⚠️  No economic data available for the specified period")
                    return pd.DataFrame()
            else:
                print(f"⚠️  Economic data fetch failed: HTTP {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Error fetching economic data: {e}")
            return pd.DataFrame()

    async def process_economic_data(self, df):
        """Process economic calendar data"""
        if df.empty:
            return df

        try:
            logger.info("🔄 Processing economic data...")

            # Clean up the dataframe
            processed_df = df.copy()

            # Drop unnecessary columns
            columns_to_drop = ['source', 'id']
            processed_df = processed_df.drop(columns=columns_to_drop, errors='ignore')

            # Convert date column
            if 'date' in processed_df.columns:
                processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')

            # Convert shorthand notation in numeric columns
            for col in processed_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'title', 'comment', 'country', 'indicator', 'category', 'currency', 'period']:
                    processed_df[col] = processed_df[col].apply(self.convert_shorthand)

            # Group by date and aggregate
            if 'date' in processed_df.columns:
                string_cols = processed_df.select_dtypes(include='object').columns
                numeric_cols = processed_df.select_dtypes(exclude='object').columns

                # Remove 'date' from string_cols if it exists
                string_cols = [col for col in string_cols if col != 'date']

                agg_dict = {}

                # Aggregate string columns
                for col in string_cols:
                    if col in ['title', 'comment']:
                        agg_dict[col] = lambda x: " | ".join(x.dropna().astype(str))
                    else:
                        agg_dict[col] = lambda x: ", ".join(x.dropna().astype(str))

                # Aggregate numeric columns
                for col in numeric_cols:
                    if col != 'date':
                        agg_dict[col] = 'mean'

                if agg_dict:
                    processed_df = processed_df.groupby("date").agg(agg_dict).reset_index()

            # Encode categorical variables
            categorical_columns = ['country', 'indicator', 'category', 'currency', 'period']
            for col in categorical_columns:
                if col in processed_df.columns:
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].fillna('unknown').astype(str))

            # Add sentiment analysis
            if 'title' in processed_df.columns:
                processed_df['sentiment_title'] = processed_df['title'].fillna('').apply(
                    lambda x: self.get_sentiment_score(x))

            if 'comment' in processed_df.columns:
                processed_df['sentiment_comment'] = processed_df['comment'].fillna('').apply(
                    lambda x: self.get_sentiment_score(x))

            # Drop text columns after sentiment analysis
            text_columns = ['title', 'comment']
            processed_df = processed_df.drop(columns=text_columns, errors='ignore')

            # Rename date column to timestamp for consistency
            if 'date' in processed_df.columns:
                processed_df = processed_df.rename(columns={'date': 'timestamp'})

            logger.info(f"✅ Economic data processed: {len(processed_df)} records")

            return processed_df

        except Exception as e:
            logger.info(f"❌ Error processing economic data: {e}")
            return pd.DataFrame()

    def convert_shorthand(self, value):
        """Convert shorthand notation (K, M, B) to numbers"""
        if isinstance(value, str):
            value = value.replace('.', '').replace(',', '').strip()
            multiplier = 1

            if value.endswith('K'):
                multiplier = 1_000
                value = value[:-1]
            elif value.endswith('M'):
                multiplier = 1_000_000
                value = value[:-1]
            elif value.endswith('B'):
                multiplier = 1_000_000_000
                value = value[:-1]

            try:
                return float(value) * multiplier
            except:
                return np.nan

        try:
            return float(value)
        except:
            return np.nan

    def get_sentiment_score(self, text):
        """Get sentiment score from text"""
        try:
            if text and len(str(text).strip()) > 0:
                return TextBlob(str(text)).sentiment.polarity
            return 0.0
        except:
            return 0.0

    async def fetch_dukascopy_data(self, start_date="2025-01-01", end_date=None):
        """Fetch gold data using dukascopy-node with fallback to Yahoo Finance"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching XAU/USD data from {start_date} to {end_date}...")

        # If fallback mode is enabled, skip Dukascopy and go directly to Yahoo Finance
        if CONFIG.get('fallback_mode', True):
            logger.info("Using fallback mode - fetching from Yahoo Finance directly")
            return await self.fetch_fallback_gold_data(start_date, end_date)

        subprocess.run(["npm", "install", "-g", "dukascopy-node"], check=True)
        logger.info("✅ dukascopy-node installed successfully")

        # Fetch data using dukascopy-node
        output_file = f"gold_data_{start_date}_to_{end_date}.csv"

        cmd = [
                "npx", "dukascopy-node",
                "-i", "xauusd",
                "-from", start_date,
                "-to", end_date,
                "-t", "h1",
                "-f", "csv",
                "-v", "true",
                "-df", "YYYY-MM-DD HH:mm:ss"
          ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.info(f"❌ Dukascopy fetch failed: {result.stderr}")
            return None

        # Fix file path handling - look in current directory and common locations
        time.sleep(5)
        possible_paths = [
                'download/',
                './',  # Current directory
                './download/',
                os.getcwd()
        ]

        csv_files = []
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    files = [f for f in os.listdir(path) if f.startswith('xauusd') and f.endswith('.csv')]
                    csv_files.extend([os.path.join(path, f) for f in files])
                except (PermissionError, OSError):
                    continue

        if not csv_files:
            logger.info("❌ No CSV file generated by dukascopy-node")
            return None

        latest_file = csv_files[0]
        logger.info(f"✅ Data fetched successfully: {os.path.basename(latest_file)}")

        # Load and process the data
        gold_df = pd.read_csv(latest_file)
        return gold_df
                        
            

    async def fetch_fallback_gold_data(self, start_date, end_date):
        """Fallback method to fetch gold data using yfinance"""
        logger.info("Using fallback method (Yahoo Finance) for gold data...")

        try:
            loop = asyncio.get_event_loop()
            
            def fetch_data():
                gold_ticker = "GC=F" # Gold futures
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

                try:
                    gold_df = gold_ticker.history(start=start_dt, end=end_dt, interval="1h")
                except:
                    logger.warning("Hourly data not available, fetching daily data...")
                    gold_df = yf.download(gold_ticker,start=start_dt, end=end_dt)

                return gold_df

            gold_df = await loop.run_in_executor(None, fetch_data)

            if gold_df.empty:
                raise Exception("No gold data available from fallback source")

            # Convert to Dukascopy format
            gold_df = gold_df.reset_index()
            
            # Create the expected Dukascopy format
            processed_df = pd.DataFrame({
                'timestamp': gold_df['Datetime'] if 'Datetime' in gold_df.columns else gold_df['Date'],
                'open': gold_df['Open'],
                'high': gold_df['High'],
                'low': gold_df['Low'],
                'close': gold_df['Close']
            })

            logger.info(f"Fallback data fetched successfully: {len(processed_df)} records")
            return await self.process_dukascopy_data(processed_df)

        except Exception as e:
            logger.error(f"Error fetching fallback gold data: {e}")
            return None

    async def process_dukascopy_data(self, df):
        """Process raw Dukascopy data according to original requirements"""
        logger.info("Processing gold data...")

        try:
            processed_df = df.copy()

            # Remove volume column if it exists (as per original code)
            if 'volume' in processed_df.columns:
                processed_df = processed_df.drop('volume', axis=1)
                logger.info("Volume column removed")

            # Ensure we have the required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            if 'timestamp' not in processed_df.columns:
                if 'datetime' in processed_df.columns:
                    processed_df['timestamp'] = processed_df['datetime']
                elif processed_df.index.name in ['datetime', 'date', 'time']:
                    processed_df = processed_df.reset_index()
                    processed_df['timestamp'] = processed_df[processed_df.columns[0]]
                else:
                    # Create timestamp from index
                    processed_df['timestamp'] = processed_df.index

            # Convert timestamp to proper datetime
            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])

            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close']
            for col in numeric_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

            # Sort by timestamp
            processed_df = processed_df.sort_values('timestamp').reset_index(drop=True)

            # Calculate change percentage for each row (exactly as in original)
            processed_df['change'] = 0.0  # Initialize
            for i in range(1, len(processed_df)):
                current_close = processed_df.loc[i, 'close']
                previous_close = processed_df.loc[i-1, 'close']
                if previous_close != 0:
                    change_pct = ((current_close - previous_close) / previous_close) * 100
                    processed_df.loc[i, 'change'] = change_pct

            # Reorder columns exactly as in original: timestamp, change, close, open, high, low
            column_order = ['timestamp', 'change', 'close', 'open', 'high', 'low']
            existing_columns = [col for col in column_order if col in processed_df.columns]
            other_columns = [col for col in processed_df.columns if col not in column_order]

            processed_df = processed_df[existing_columns + other_columns]

            # Remove rows with NaN values
            processed_df = processed_df.dropna()

            logger.info(f"Gold data processed: {len(processed_df)} records")
            logger.info(f"Date range: {processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}")
            logger.info(f"Columns: {list(processed_df.columns)}")

            self.gold_data = processed_df
            return processed_df

        except Exception as e:
            logger.error(f"Error processing gold data: {e}")
            return None

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataset"""
        try:
            df_with_indicators = df.copy()

            if 'close' not in df_with_indicators.columns:
                return df_with_indicators

            close_series = df_with_indicators['close']
            high_series = df_with_indicators.get('high', close_series)
            low_series = df_with_indicators.get('low', close_series)

            # RSI
            try:
                df_with_indicators['RSI'] = RSIIndicator(close=close_series).rsi()
            except:
                df_with_indicators['RSI'] = 50

            # MACD
            try:
                macd = MACD(close=close_series)
                df_with_indicators['MACD'] = macd.macd()
                df_with_indicators['MACD_signal'] = macd.macd_signal()
                df_with_indicators['MACD_diff'] = macd.macd_diff()
            except:
                df_with_indicators['MACD'] = 0
                df_with_indicators['MACD_signal'] = 0
                df_with_indicators['MACD_diff'] = 0

            # Moving Averages
            try:
                df_with_indicators['SMA_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()
                df_with_indicators['SMA_50'] = SMAIndicator(close=close_series, window=50).sma_indicator()
                df_with_indicators['EMA_12'] = EMAIndicator(close=close_series, window=12).ema_indicator()
                df_with_indicators['EMA_26'] = EMAIndicator(close=close_series, window=26).ema_indicator()
            except:
                df_with_indicators['SMA_20'] = close_series
                df_with_indicators['SMA_50'] = close_series
                df_with_indicators['EMA_12'] = close_series
                df_with_indicators['EMA_26'] = close_series

            # Bollinger Bands
            try:
                bb = BollingerBands(close=close_series)
                df_with_indicators['BB_upper'] = bb.bollinger_hband()
                df_with_indicators['BB_lower'] = bb.bollinger_lband()
                df_with_indicators['BB_middle'] = bb.bollinger_mavg()
                df_with_indicators['BB_width'] = bb.bollinger_wband()
            except:
                df_with_indicators['BB_upper'] = close_series * 1.02
                df_with_indicators['BB_lower'] = close_series * 0.98
                df_with_indicators['BB_middle'] = close_series
                df_with_indicators['BB_width'] = 0.04

            # ATR
            try:
                df_with_indicators['ATR'] = AverageTrueRange(
                    high=high_series, low=low_series, close=close_series
                ).average_true_range()
            except:
                df_with_indicators['ATR'] = close_series * 0.01

            # Time-based features
            if 'timestamp' in df_with_indicators.columns:
                df_with_indicators['hour'] = df_with_indicators['timestamp'].dt.hour
                df_with_indicators['day_of_week'] = df_with_indicators['timestamp'].dt.dayofweek
                df_with_indicators['month'] = df_with_indicators['timestamp'].dt.month
                df_with_indicators['is_weekend'] = (df_with_indicators['day_of_week'] >= 5).astype(int)

            # Price-based features
            if all(col in df_with_indicators.columns for col in ['open', 'high', 'low', 'close']):
                df_with_indicators['price_range'] = ((high_series - low_series) / close_series) * 100
                df_with_indicators['open_close_ratio'] = df_with_indicators['open'] / close_series
                df_with_indicators['high_close_ratio'] = high_series / close_series
                df_with_indicators['low_close_ratio'] = low_series / close_series

            # Fill NaN values
            df_with_indicators = df_with_indicators.fillna(method='ffill').fillna(method='bfill').fillna(0)

            logger.info(f"Technical indicators added: {len(df_with_indicators.columns)} total features")
            return df_with_indicators

        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    async def run_pipeline(self, start_date="2025-01-01", end_date=None, append_only=False):
        """Run the complete data pipeline"""
        try:
            logger.info("Starting data pipeline...")

            # Fetch gold data using Dukascopy or fallback
            gold_data = await self.fetch_dukascopy_data(start_date, end_date)
            if gold_data is None:
                return None

            # Fetch economic data (optional, simplified for stability)
            try:
                economic_data = await self.fetch_economic_data(start_date, end_date or datetime.now().strftime("%Y-%m-%d"))
            except Exception as e:
                logger.warning(f"Economic data fetch failed: {e}. Continuing without economic data.")
                economic_data = pd.DataFrame()

            # Combine datasets
            combined_data = await self.combine_datasets(gold_data, economic_data)
            if combined_data is None:
                return None
            
            if append_only and os.path.exists(CONFIG['data_file']):
                # Load existing data and append only new records
                try:
                    existing_data = pd.read_csv(CONFIG['data_file'])
                    existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
                    
                    if len(existing_data) > 0:
                        last_timestamp = existing_data['timestamp'].max()
                        new_data = combined_data[combined_data['timestamp'] > last_timestamp]
                        
                        if len(new_data) > 0:
                            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                            logger.info(f"Appended {len(new_data)} new records")
                        else:
                            combined_data = existing_data
                            logger.info("No new data to append")
                except Exception as e:
                    logger.warning(f"Error loading existing data: {e}. Using new data only.")

            # Save processed data
            combined_data.to_csv(CONFIG['data_file'], index=False)
            
            self.combined_data = combined_data
            logger.info(f"Pipeline completed: {len(combined_data)} records")
            
            return combined_data

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return None

    async def combine_datasets(self, gold_data, economic_data):
        """Combine gold and economic data"""
        if self.gold_data is None:
            logger.info("❌ No gold data available for combination")
            return None

        logger.info("🔄 Combining gold and economic data...")

        try:
            # Start with gold data
            combined_df = self.gold_data.copy()

            # If we have economic data, merge it
            if self.economic_data is not None and not self.economic_data.empty:
                # Ensure both have timestamp columns
                if 'timestamp' in combined_df.columns and 'timestamp' in self.economic_data.columns:
                    # Round timestamps to nearest hour for matching
                    combined_df['timestamp_hour'] = combined_df['timestamp'].dt.floor('H')
                    econ_df_hour = self.economic_data.copy()
                    econ_df_hour['timestamp_hour'] = econ_df_hour['timestamp'].dt.floor('H')

                    # ✅ Fix timezone handling - ensure both are timezone-naive
                    if combined_df['timestamp_hour'].dt.tz is not None:
                        combined_df['timestamp_hour'] = combined_df['timestamp_hour'].dt.tz_localize(None)
                    if econ_df_hour['timestamp_hour'].dt.tz is not None:
                        econ_df_hour['timestamp_hour'] = econ_df_hour['timestamp_hour'].dt.tz_localize(None)

                    # Merge on rounded timestamp
                    combined_df = combined_df.merge(
                        econ_df_hour.drop('timestamp', axis=1),
                        on='timestamp_hour',
                        how='left'
                    )

                    # Drop the helper column
                    combined_df = combined_df.drop('timestamp_hour', axis=1)

                    logger.info(f"✅ Data combined with economic indicators")
                else:
                    logger.info("⚠️  Could not merge economic data - timestamp column missing")
            else:
                logger.info("⚠️  No economic data to combine")

            # Add technical indicators
            combined_df = self.add_technical_indicators(combined_df)

            # Fill NaN values
            combined_df = self.fill_missing_values(combined_df)

            # Final cleanup
            combined_df.fillna(0, inplace=True)

            # ✅ Ensure timestamp is properly formatted and sorted
            if 'timestamp' in combined_df.columns:
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

                # Print actual date range for debugging
                logger.info(f"📊 Actual date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

            logger.info(f"✅ Combined dataset created: {len(combined_df)} records with {len(combined_df.columns)} features")

            self.combined_data = combined_df
            return combined_df

        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            return None

    def fill_missing_values(self, df):
        """Fill missing values in the dataset"""
        try:
            logger.info("Filling missing values...")
            df_filled = df.copy()

            # Fill numeric columns with forward fill, then backward fill, then median
            numeric_columns = df_filled.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                if df_filled[col].isnull().sum() > 0:
                    # Forward fill
                    df_filled[col] = df_filled[col].fillna(method='ffill')
                    # Backward fill
                    df_filled[col] = df_filled[col].fillna(method='bfill')
                    # Fill remaining with median
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())

            # Fill categorical columns with mode
            categorical_columns = df_filled.select_dtypes(exclude=[np.number]).columns
            categorical_columns = [col for col in categorical_columns if col != 'timestamp']

            for col in categorical_columns:
                if df_filled[col].isnull().sum() > 0:
                    mode_value = df_filled[col].mode()
                    if len(mode_value) > 0:
                        df_filled[col] = df_filled[col].fillna(mode_value[0])
                    else:
                        df_filled[col] = df_filled[col].fillna('unknown')

            logger.info("Missing values filled")
            return df_filled

        except Exception as e:
            logger.error(f"Error filling missing values: {e}")
            return df

class GoldTradingSignalSystem:
    """Trading signal system for FastAPI"""

    def __init__(self, model=None, scaler=None, seq_len=60):
        self.model = model
        self.scaler = scaler
        self.seq_len = seq_len
        self.predictions_history = []

        self.signal_thresholds = {
            'strong_buy': 0.15,
            'buy': 0.05,
            'hold': 0.02,
            'sell': -0.05,
            'strong_sell': -0.15
        }

    def prepare_data_for_prediction(self, data_df):
        """Prepare data for model prediction"""
        try:
            df = data_df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Select numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if 'change' in numeric_columns:
                target_idx = numeric_columns.index('change')
                if target_idx != 0:
                    numeric_columns.insert(0, numeric_columns.pop(target_idx))

            feature_df = df[numeric_columns].copy()
            feature_df = feature_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            return feature_df, df

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None

    async def make_multi_hour_prediction(self, recent_data_df, hours_ahead=5):
        """Make predictions for multiple hours ahead"""
        if not TENSORFLOW_AVAILABLE or self.model is None or self.scaler is None:
            logger.warning("Model or scaler not loaded. Using simple trend analysis.")
            return await self.make_simple_prediction(recent_data_df, hours_ahead)

        try:
            feature_df, original_df = self.prepare_data_for_prediction(recent_data_df)
            if feature_df is None:
                return None

            if len(feature_df) < self.seq_len:
                logger.error(f"Not enough data points. Need {self.seq_len}, got {len(feature_df)}")
                return None

            # Get prediction in thread pool
            loop = asyncio.get_event_loop()
            
            def make_prediction():
                last_sequence = feature_df.iloc[-self.seq_len:].values
                sequence_scaled = self.scaler.transform(last_sequence)
                sequence_reshaped = sequence_scaled.reshape(1, self.seq_len, -1)
                prediction = self.model.predict(sequence_reshaped, verbose=0)
                
                if len(prediction.shape) == 3:
                    prediction = prediction[:, 0]
                if len(prediction.shape) == 1:
                    prediction = prediction.reshape(-1, 1)

                n_features = sequence_scaled.shape[1]
                if prediction.shape[1] < n_features:
                    padding = sequence_scaled[-1, prediction.shape[1]:].reshape(1, -1)
                    dummy_array = np.hstack([prediction, padding])
                else:
                    dummy_array = prediction[:, :n_features]

                prediction_inverse = self.scaler.inverse_transform(dummy_array)
                return prediction_inverse[0, 0]

            predicted_change = await loop.run_in_executor(None, make_prediction)

            # Get current values
            current_close = float(original_df['close'].iloc[-1])
            current_time = original_df['timestamp'].iloc[-1]

            # Generate multi-hour predictions
            multi_hour_predictions = []
            cumulative_change = 0

            for hour in range(1, hours_ahead + 1):
                decay_factor = 0.85 ** (hour - 1)
                hourly_change = predicted_change * decay_factor
                cumulative_change += hourly_change

                predicted_price = current_close * (1 + cumulative_change / 100)
                future_time = current_time + timedelta(hours=hour)

                multi_hour_predictions.append({
                    'hour_ahead': hour,
                    'timestamp': future_time.isoformat(),
                    'predicted_change': float(hourly_change),
                    'cumulative_change': float(cumulative_change),
                    'predicted_price': float(predicted_price),
                    'confidence_decay': float(decay_factor)
                })

            # Generate signal
            signal = self.generate_signal(predicted_change, multi_hour_predictions)

            prediction_result = {
                'timestamp': current_time.isoformat(),
                'current_price': float(current_close),
                'predicted_change': float(predicted_change),
                'predicted_price': float(current_close * (1 + predicted_change / 100)),
                'multi_hour_predictions': multi_hour_predictions,
                'signal': signal['action'],
                'signal_strength': signal['strength'],
                'confidence': signal['confidence']
            }

            self.predictions_history.append(prediction_result)
            return prediction_result

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return await self.make_simple_prediction(recent_data_df, hours_ahead)

    async def make_simple_prediction(self, recent_data_df, hours_ahead=5):
        """Simple trend-based prediction when ML model is not available"""
        try:
            df = recent_data_df.copy()
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate simple trend
            recent_changes = df['change'].tail(10).mean()
            current_close = float(df['close'].iloc[-1])
            current_time = df['timestamp'].iloc[-1]
            
            # Generate simple predictions based on recent trend
            multi_hour_predictions = []
            cumulative_change = 0
            
            for hour in range(1, hours_ahead + 1):
                decay_factor = 0.9 ** (hour - 1)
                hourly_change = recent_changes * decay_factor
                cumulative_change += hourly_change
                
                predicted_price = current_close * (1 + cumulative_change / 100)
                future_time = current_time + timedelta(hours=hour)
                
                multi_hour_predictions.append({
                    'hour_ahead': hour,
                    'timestamp': future_time.isoformat(),
                    'predicted_change': float(hourly_change),
                    'cumulative_change': float(cumulative_change),
                    'predicted_price': float(predicted_price),
                    'confidence_decay': float(decay_factor)
                })
            
            # Generate signal based on trend
            signal = self.generate_signal(recent_changes, multi_hour_predictions)
            
            prediction_result = {
                'timestamp': current_time.isoformat(),
                'current_price': float(current_close),
                'predicted_change': float(recent_changes),
                'predicted_price': float(current_close * (1 + recent_changes / 100)),
                'multi_hour_predictions': multi_hour_predictions,
                'signal': signal['action'],
                'signal_strength': signal['strength'],
                'confidence': signal['confidence'],
                'method': 'simple_trend'
            }
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making simple prediction: {e}")
            return None

    def generate_signal(self, predicted_change, multi_hour_predictions):
        """Generate trading signal"""
        try:
            # Calculate weighted prediction
            short_term_changes = [p['predicted_change'] for p in multi_hour_predictions[:2]]
            medium_term_changes = [p['predicted_change'] for p in multi_hour_predictions[2:4]]
            long_term_changes = [p['predicted_change'] for p in multi_hour_predictions[4:]]

            short_term_avg = np.mean(short_term_changes) if short_term_changes else 0
            medium_term_avg = np.mean(medium_term_changes) if medium_term_changes else 0
            long_term_avg = np.mean(long_term_changes) if long_term_changes else 0

            weighted_prediction = (short_term_avg * 0.5 + medium_term_avg * 0.3 + long_term_avg * 0.2)

            # Generate signal
            if weighted_prediction >= self.signal_thresholds['strong_buy']:
                signal = 'STRONG_BUY'
                strength = min(10, int(weighted_prediction * 20))
            elif weighted_prediction >= self.signal_thresholds['buy']:
                signal = 'BUY'
                strength = min(7, int(weighted_prediction * 25))
            elif weighted_prediction <= self.signal_thresholds['strong_sell']:
                signal = 'STRONG_SELL'
                strength = min(10, int(abs(weighted_prediction) * 20))
            elif weighted_prediction <= self.signal_thresholds['sell']:
                signal = 'SELL'
                strength = min(7, int(abs(weighted_prediction) * 25))
            else:
                signal = 'HOLD'
                strength = 1

            confidence = min(100, max(10, 50 + (strength * 5)))

            return {
                'action': signal,
                'strength': strength,
                'confidence': float(confidence)
            }

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {
                'action': 'HOLD',
                'strength': 1,
                'confidence': 50.0
            }

async def download_model_files():
    """Download model and scaler files if they don't exist"""
    if not TENSORFLOW_AVAILABLE:
        logger.info("TensorFlow not available. Skipping model download.")
        return

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for file_path, url in [(CONFIG['model_path'], CONFIG['model_url']),
                               (CONFIG['scaler_path'], CONFIG['scaler_url'])]:
            if not os.path.exists(file_path):
                try:
                    logger.info(f"Downloading {file_path}...")
                    response = await client.get(url)
                    response.raise_for_status()

                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(response.content)

                    logger.info(f"Downloaded {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to download {file_path}: {e}")


async def load_model_and_scaler():
    """Load model and scaler"""
    if not TENSORFLOW_AVAILABLE:
        logger.info("TensorFlow not available. ML predictions will be disabled.")
        return None, None

    try:
        await download_model_files()

        if os.path.exists(CONFIG['model_path']) and os.path.exists(CONFIG['scaler_path']):
            loop = asyncio.get_event_loop()

            def load_files():
                try:
                    model = load_model(CONFIG['model_path'], compile=False)
                    model.compile(optimizer='adam', loss='mse')
                except Exception as model_error:
                    logger.warning(f"Error loading model: {model_error}")
                    model = None
                try:
                    scaler = joblib.load(CONFIG['scaler_path'])
                except Exception as scaler_error:
                    logger.warning(f"Error loading scaler: {scaler_error}")
                    scaler = None
                return model, scaler

            model, scaler = await loop.run_in_executor(None, load_files)

            if model is not None and scaler is not None:
                logger.info("Model and scaler loaded successfully")
                return model, scaler
            else:
                logger.warning("Model or scaler loading failed. Using simple prediction method.")
                return None, None
        else:
            logger.warning("Model or scaler files not found. Using simple prediction method.")
            return None, None

    except Exception as e:
        logger.warning(f"Error loading model/scaler: {e}. Using simple prediction method.")
        return None, None

async def update_data_and_predict():
    """Update data and make predictions"""
    global current_data, latest_prediction, data_pipeline, trading_system
    
    try:
        logger.info("Starting scheduled data update...")
        
        # Update data (append only new data)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        current_data = await data_pipeline.run_pipeline(
            start_date=start_date,
            end_date=end_date,
            append_only=True
        )
        
        if current_data is not None and len(current_data) >= CONFIG['seq_len']:
            # Make prediction
            latest_prediction = await trading_system.make_multi_hour_prediction(
                current_data.tail(CONFIG['seq_len']),
                hours_ahead=CONFIG['prediction_hours']
            )
            logger.info("Data updated and prediction made successfully")
        else:
            logger.warning("Insufficient data for prediction")
            
    except Exception as e:
        logger.error(f"Error in scheduled update: {e}")

def run_scheduler():
    """Run the scheduler in a separate thread"""
    def job():
        asyncio.run(update_data_and_predict())
    
    schedule.every(CONFIG['update_interval_hours']).hours.do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_pipeline, trading_system, current_data, scheduler_thread

    logger.info("Starting FastAPI Gold Trading System...")

    # Initialize data pipeline
    data_pipeline = AutomatedDataPipeline()
    
    # Background task for slow initializations
    async def background_setup():
        global trading_system, current_data

        try:
            model, scaler = await load_model_and_scaler()
            trading_system = GoldTradingSignalSystem(model, scaler, CONFIG['seq_len'])

            # Load data
            if not os.path.exists(CONFIG['data_file']):
                logger.info("Initial data load...")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                current_data = await data_pipeline.run_pipeline(start_date=start_date)
            else:
                current_data = pd.read_csv(CONFIG['data_file'])
                current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
                logger.info(f"Loaded existing data: {len(current_data)} records")

            # Initial prediction
            if current_data is not None and len(current_data) >= CONFIG['seq_len']:
                await update_data_and_predict()

        except Exception as e:
            logger.error(f"Background setup error: {e}")

    asyncio.create_task(background_setup())  # ✅ Don't block lifespan

    # Start scheduler
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Scheduler started")

    yield
    logger.info("Shutting down...")


# FastAPI app
app = FastAPI(
    title="Gold Trading Signal API",
    description="Real-time gold price prediction and trading signals",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gold Trading Signal API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/prediction",
            "/historical",
            "/status",
            "/health"
        ]
    }

@app.get("/prediction")
async def get_current_prediction():
    """Get current prediction with future forecasts"""
    global latest_prediction
    
    if latest_prediction is None:
        # Try to make a prediction if we have data
        if current_data is not None and len(current_data) >= CONFIG['seq_len']:
            latest_prediction = await trading_system.make_multi_hour_prediction(
                current_data.tail(CONFIG['seq_len']),
                hours_ahead=CONFIG['prediction_hours']
            )
        
        if latest_prediction is None:
            raise HTTPException(status_code=503, detail="No prediction available")
    
    return JSONResponse(content=latest_prediction)

@app.get("/historical")
async def get_historical_data(hours: int = 60):
    """Get last N hours of historical data with indicators"""
    global current_data
    
    if current_data is None or len(current_data) == 0:
        raise HTTPException(status_code=503, detail="No historical data available")
    
    try:
        # Get last N hours of data
        recent_data = current_data.tail(hours).copy()
        
        # Convert to JSON-serializable format
        historical_records = []
        for _, row in recent_data.iterrows():
            record = {
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'change': float(row['change']),
                'RSI': float(row.get('RSI', 50)),
                'MACD': float(row.get('MACD', 0)),
                'MACD_signal': float(row.get('MACD_signal', 0)),
                'MACD_diff': float(row.get('MACD_diff', 0)),
                'SMA_20': float(row.get('SMA_20', row['close'])),
                'SMA_50': float(row.get('SMA_50', row['close'])),
                'EMA_12': float(row.get('EMA_12', row['close'])),
                'EMA_26': float(row.get('EMA_26', row['close'])),
                'BB_upper': float(row.get('BB_upper', row['close'] * 1.02)),
                'BB_lower': float(row.get('BB_lower', row['close'] * 0.98)),
                'BB_middle': float(row.get('BB_middle', row['close'])),
                'BB_width': float(row.get('BB_width', 0.04)),
                'ATR': float(row.get('ATR', row['close'] * 0.01))
            }
            historical_records.append(record)
        
        # Add future predictions if available
        future_predictions = []
        if latest_prediction and 'multi_hour_predictions' in latest_prediction:
            future_predictions = latest_prediction['multi_hour_predictions']
        
        return JSONResponse(content={
            'historical_data': historical_records,
            'future_predictions': future_predictions,
            'total_records': len(historical_records),
            'last_updated': current_data['timestamp'].max().isoformat() if len(current_data) > 0 else None
        })
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get system status"""
    global current_data, latest_prediction, trading_system
    
    status = {
        'system_status': 'running',
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'model_loaded': trading_system is not None and trading_system.model is not None,
        'data_available': current_data is not None,
        'last_prediction': latest_prediction['timestamp'] if latest_prediction else None,
        'data_records': len(current_data) if current_data is not None else 0,
        'last_data_update': current_data['timestamp'].max().isoformat() if current_data is not None and len(current_data) > 0 else None,
        'prediction_method': 'ML' if (trading_system and trading_system.model) else 'Simple Trend',
        'config': CONFIG
    }
    
    return JSONResponse(content=status)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/manual-update")
async def manual_update():
    """Manually trigger data update and prediction"""
    asyncio.run(update_data_and_predict())
    return {"message": "Manual update triggered"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )