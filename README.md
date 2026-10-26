# Market making bot (prototype, 2025)

This is a learning project from early 2025, not a live trading system. The later commit only adds a package skeleton. The dashboard described below was never built. Do not point real Binance keys at it.

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