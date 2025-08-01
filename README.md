# Gold Trading Signal API

A FastAPI-based real-time gold price prediction and trading signal system with automated data pipeline, technical indicators, and ML-powered forecasting.

## Features

✅ **Real-time Gold Data**: Fetches XAU/USD spot prices with fallback to Yahoo Finance  
✅ **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA, EMA, ATR  
✅ **ML Predictions**: 5-hour ahead forecasts with confidence decay  
✅ **Trading Signals**: BUY/SELL/HOLD signals with strength ratings  
✅ **Automated Updates**: Hourly data updates with incremental processing  
✅ **REST API**: Clean JSON endpoints for external integration  
✅ **Economic Data**: Optional economic calendar integration  
✅ **Fallback System**: Graceful degradation when services are unavailable  

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/prediction` | GET | Current prediction with 5-hour forecasts |
| `/historical?hours=60` | GET | Last N hours of data with indicators |
| `/status` | GET | System status and configuration |
| `/health` | GET | Health check endpoint |
| `/manual-update` | POST | Manually trigger data update |

## Quick Start

### Local Development

1. **Clone and Setup**
```bash
git clone <your-repo-url>
cd gold-trading-api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run the Application**
```bash
python main.py
# Or using uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Test the API**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
curl http://localhost:8000/prediction
```

### Docker Deployment

1. **Build and Run**
```bash
docker build -t gold-trading-api .
docker run -p 8000:8000 gold-trading-api
```

2. **Using Docker Compose** (optional)
```yaml
version: '3.8'
services:
  gold-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `MODEL_URL` | - | URL to download model.h5 |
| `SCALER_URL` | - | URL to download scaler.pkl |

### Application Configuration

Edit the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    'data_file': 'gold_processed_data.csv',
    'model_path': 'model.h5',
    'scaler_path': 'scaler.pkl',
    'update_interval_hours': 1,
    'seq_len': 60,
    'prediction_hours': 5,
    'fallback_mode': True  # Use Yahoo Finance instead of Dukascopy
}
```

## API Response Examples

### Prediction Endpoint
```json
{
  "timestamp": "2025-01-01T12:00:00",
  "current_price": 2650.50,
  "predicted_change": 0.75,
  "predicted_price": 2670.25,
  "multi_hour_predictions": [
    {
      "hour_ahead": 1,
      "timestamp": "2025-01-01T13:00:00",
      "predicted_change": 0.75,
      "cumulative_change": 0.75,
      "predicted_price": 2670.25,
      "confidence_decay": 1.0
    }
  ],
  "signal": "BUY",
  "signal_strength": 6,
  "confidence": 80.0
}
```

### Historical Data Endpoint
```json
{
  "historical_data": [
    {
      "timestamp": "2025-01-01T11:00:00",
      "open": 2645.00,
      "high": 2655.00,
      "low": 2640.00,
      "close": 2650.50,
      "change": 0.21,
      "RSI": 65.5,
      "MACD": 2.1,
      "SMA_20": 2648.75,
      "BB_upper": 2665.00,
      "BB_lower": 2635.00
    }
  ],
  "future_predictions": [...],
  "total_records": 60,
  "last_updated": "2025-01-01T12:00:00"
}
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Data Pipeline  │    │  Trading System │
│                 │    │                  │    │                 │
│ • Yahoo Finance │───▶│ • Data Fetching  │───▶│ • ML Prediction │
│ • Dukascopy     │    │ • Technical Ind. │    │ • Signal Gen.   │
│ • Economic Data │    │ • Data Cleaning  │    │ • Trend Analysis│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Scheduler     │    │   Data Storage   │    │   FastAPI App   │
│                 │    │                  │    │                 │
│ • Hourly Updates│    │ • CSV Files      │    │ • REST Endpoints│
│ • Background    │    │ • Model Files    │    │ • CORS Support  │
│ • Error Handling│    │ • Incremental    │    │ • Health Checks │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Troubleshooting

### Common Issues

1. **TensorFlow Import Error**
   - The system will automatically fall back to simple trend analysis
   - Install TensorFlow: `pip install tensorflow==2.15.0`

2. **Data Fetching Failures**
   - System uses Yahoo Finance as fallback
   - Check internet connectivity
   - Verify API endpoints are accessible

3. **Model Loading Issues**
   - Ensure model files are accessible at configured URLs
   - Check file permissions and paths
   - System will use simple prediction if models fail to load

4. **Memory Issues**
   - Reduce `seq_len` in CONFIG for lower memory usage
   - Consider using smaller model files
   - Monitor system resources

### Logs and Monitoring

The application provides comprehensive logging:
- Startup and shutdown events
- Data fetching and processing status
- Prediction generation
- Error handling and fallbacks

Check logs for debugging:
```bash
# Local development
python main.py

# Docker
docker logs <container-id>

# Render/Heroku
Check platform-specific log viewers
```

## Performance Optimization

### For Production

1. **Caching**: Implement Redis for prediction caching
2. **Database**: Use PostgreSQL instead of CSV files
3. **Load Balancing**: Deploy multiple instances
4. **CDN**: Use CDN for static assets
5. **Monitoring**: Add APM tools like New Relic or DataDog

### Resource Requirements

- **Minimum**: 512MB RAM, 1 CPU core
- **Recommended**: 1GB RAM, 2 CPU cores
- **Storage**: 1GB for data and model files

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Create an issue on GitHub
4. Contact the development team

---

**Note**: This system is for educational and research purposes. Always validate predictions with additional analysis before making trading decisions.
