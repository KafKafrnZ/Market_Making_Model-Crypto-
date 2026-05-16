# Market Making Bot - Crypto (Enterprise Edition)

Production-ready cryptocurrency market making bot with LSTM prediction, advanced risk management, and beautiful dashboard.

## Features
- Real-time Binance WebSocket data
- LSTM + Technical Indicators for price prediction
- Dynamic spread & inventory skewing
- Professional risk & order management
- Modern Dash / Streamlit UI (coming in v2)

## Quick Start

```bash
pip install -e .
cp .env.example .env
# Edit .env with your keys
python -m market_maker.scripts.run_bot
```

## Structure
See `docs/architecture.md`