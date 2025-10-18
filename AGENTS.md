Develop a fully autonomous, profit-oriented trading agent leveraging machine learning, ATR-based momentum logic, and multi-agent reinforcement feedback to ensure consistent, risk-managed gains in volatile crypto markets.
This system aims to evolve into a self-optimizing trading AI — a true money-making machine that adapts in real time.

Core System Overview

Primary Engine: ai_bot.py
Exchange: Binance via CCXT
Mode: Live / Paper (toggle with LIVE_MODE)
Capital: USDC-based risk-managed portfolio

The system continuously:

Discovers top-volume USDC pairs.

Builds or updates per-symbol XGBoost models.

Calculates dynamic ATR-based stop-loss/take-profit thresholds.

Executes entries only when market trend alignment and probabilistic confidence exceed set thresholds.

Manages existing positions with real-time adaptive trailing logic and capital exposure control.

Reports all actions via Telegram alerts and live console dashboard.

Agent Architecture
1. Discovery Agent

File: ai_bot.py → discover_symbols()

Continuously scans Binance markets for high-volume USDC pairs.

Dynamically updates tracking list (TOP_N).

Filters pairs with sufficient liquidity (MIN_VOLUME).

2. Market Sentiment Agent

File: ai_bot.py → market_bullish()

Filters new entries using Bitcoin’s EMA trend correlation.

Prevents entries during bearish conditions.

3. Model Intelligence Agent

Class: SymbolModel

Builds symbol-specific predictive models using XGBoost.

Learns from OHLCV + engineered features (EMA, RSI, ATR, Volume Ratios).

Retrains hourly or on-demand.

Maintains local model persistence in /models.

4. Risk & Capital Agent

Controls exposure per trade (risk_percent).

Ensures total active capital ≤ MAX_CAPITAL_EXPOSURE.

Enforces position cap (MAX_OPEN_POSITIONS).

Calculates adaptive trade size based on effective capital and realized profit.

5. Execution Agent

Interfaces with Binance API via CCXT.

Executes live market orders (create_market_buy_order, create_market_sell_order).

Automatically updates state after exits.

Fully integrated with Telegram alerts for monitoring.

6. Profit & State Agent

Tracks cumulative and daily PnL via STATE_FILE.

Updates realized_profit, daily_profit, and last trade date.

Provides effective capital recalculation.

Writes persistent state logs for analysis and dashboard rendering.

7. Dashboard Agent

Displays real-time metrics (capital usage, open trades, next signal).

Includes next-symbol prediction and probability display.

Visual console refreshed each loop cycle.

Acts as human-monitoring interface for live feedback.

Learning Loop

The bot executes a continuous self-training cycle:

Fetch market data

Update symbol models

Predict next-hour momentum direction

Enter only if confidence > 0.65

Update stop-loss dynamically based on profit-lock logic

Exit automatically on TP/SL breach

Reinforce learning with new data

This allows the system to adapt to new volatility regimes, emerging patterns, and market anomalies.

Safety and Reliability

All API calls are rate-limited and exception-handled.

Sleep intervals protect against overload.

State recovery ensures safe restart after crash or disconnect.

Backward-compatible model structure enables seamless upgrade to v4+.

Next Evolution Goals

Integrate deep reinforcement learning agent for dynamic risk scaling.

Expand to multi-quote trading (BTC, ETH, BUSD).

Implement backtesting and simulation server.

Deploy cloud-based monitoring dashboard with performance metrics.

Add strategy mutation layer for automated strategy evolution.

Conclusion

The Adaptive ATR Momentum Bot is not just a trading bot — it’s the foundation for an autonomous trading intelligence network, designed to learn, trade, and grow capital over time with minimal human input.
