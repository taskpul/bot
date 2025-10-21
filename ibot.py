"""Trail Accumulator Bot with multi-symbol DCA support and live heartbeat logging."""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt.async_support as ccxt_async
import importlib
import importlib.util
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ---- Indicator backend ----
talib = None
ta = None
_talib_spec = importlib.util.find_spec("talib")
if _talib_spec is not None:
    talib = importlib.import_module("talib")
else:
    _ta_spec = importlib.util.find_spec("ta")
    if _ta_spec is None:
        raise ImportError("Either talib or ta package must be installed.")
    ta = importlib.import_module("ta")


# ---- Config & State ----
@dataclass
class Config:
    symbol: str = "BTC/USDC"
    timeframe: str = "1h"
    allocation_usdc: float = 400.0
    state_file: Path = Path("btc_usdc_state.json")
    poll_interval: int = 60
    candles_limit: int = 200
    live_mode: bool = False
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    paper_balance: float = 5000.0
    trade_symbols: List[str] = field(default_factory=lambda: ["BTC/USDC"])
    dca_drop_percent: float = 3.0
    profit_step_percent: float = 0.5
    base_currency: str = "USDC"


@dataclass
class TradeState:
    has_position: bool = False
    average_entry_price: float = 0.0
    stop_loss: float = 0.0
    amount: float = 0.0
    balance_usdc: float = 0.0
    total_invested_usdc: float = 0.0
    last_stop_update_ts: int = 0
    last_buy_price: float = 0.0
    profit_trailing_active: bool = False
    highest_price: float = 0.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TradeState":
        return cls(
            has_position=bool(payload.get("has_position", False)),
            average_entry_price=float(payload.get("average_entry_price") or payload.get("entry_price", 0.0)),
            stop_loss=float(payload.get("stop_loss", 0.0)),
            amount=float(payload.get("amount", 0.0)),
            balance_usdc=float(payload.get("balance_usdc") or payload.get("balance_usdt", 0.0)),
            total_invested_usdc=float(payload.get("total_invested_usdc", 0.0)),
            last_stop_update_ts=int(payload.get("last_stop_update_ts", 0)),
            last_buy_price=float(payload.get("last_buy_price", 0.0)),
            profit_trailing_active=bool(payload.get("profit_trailing_active", False)),
            highest_price=float(payload.get("highest_price", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---- Helpers ----
def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> Config:
    load_dotenv()
    config = Config()
    config.live_mode = parse_bool(os.getenv("LIVE_MODE"), False)
    config.api_key = os.getenv("BINANCE_API_KEY")
    config.api_secret = os.getenv("BINANCE_API_SECRET")
    config.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    config.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    config.allocation_usdc = float(os.getenv("TRADE_ALLOCATION_USDC", config.allocation_usdc))
    config.poll_interval = int(os.getenv("POLL_INTERVAL", config.poll_interval))
    config.candles_limit = int(os.getenv("CANDLES_LIMIT", config.candles_limit))
    config.paper_balance = float(os.getenv("PAPER_BALANCE_USDC", config.paper_balance))
    config.dca_drop_percent = float(os.getenv("DCA_DROP_PERCENT", config.dca_drop_percent))
    config.profit_step_percent = float(os.getenv("PROFIT_STEP_PERCENT", config.profit_step_percent))
    config.base_currency = os.getenv("BASE_CURRENCY", config.base_currency).strip().upper()
    symbols_env = os.getenv("TRADE_SYMBOLS", "")
    if symbols_env:
        config.trade_symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
        if config.trade_symbols:
            config.symbol = config.trade_symbols[0]
    return config


def _state_file_for_symbol(config: Config, symbol: str) -> Path:
    symbol_key = symbol.replace("/", "_").lower()
    return Path(f"{symbol_key}_state.json")


# ---- Core Bot ----
class TrailAccumulatorBot:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = self._load_state()
        if not self.state.balance_usdc:
            self.state.balance_usdc = config.paper_balance
        base, quote = config.symbol.split("/")
        self.base_asset, self.quote_asset = base, quote
        self.symbol_label = config.symbol
        self.profit_step_factor = config.profit_step_percent / 100.0
        self.exchange = self._create_exchange()
        self.last_candle_time = 0
        self.market: Optional[Dict[str, Any]] = None

    def _create_exchange(self):
        params = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        if self.config.live_mode:
            params.update({"apiKey": self.config.api_key, "secret": self.config.api_secret})
        return ccxt_async.binance(params)

    def _load_state(self) -> TradeState:
        fp = _state_file_for_symbol(self.config, self.config.symbol)
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    return TradeState.from_dict(json.load(f))
            except Exception:
                pass
        return TradeState()

    def _save_state(self):
        fp = _state_file_for_symbol(self.config, self.config.symbol)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    async def _fetch_candles(self):
        try:
            return await self.exchange.fetch_ohlcv(
                self.config.symbol, timeframe=self.config.timeframe, limit=self.config.candles_limit
            )
        except Exception as e:
            logging.error(f"Fetch error {self.symbol_label}: {e}")
            return None

    @staticmethod
    def _build_df(candles):
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def _calc_indicators(self, df):
        df["ema50"] = talib.EMA(df["close"], 50) if talib else ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["ema200"] = talib.EMA(df["close"], 200) if talib else ta.trend.EMAIndicator(df["close"], 200).ema_indicator()
        df["rsi14"] = talib.RSI(df["close"], 14) if talib else ta.momentum.RSIIndicator(df["close"], 14).rsi()
        df["atr14"] = talib.ATR(df["high"], df["low"], df["close"], 14) if talib else ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], 14).average_true_range()
        df["high20"] = df["high"].rolling(20).max()
        df["vol_mean20"] = df["vol"].rolling(20).mean()
        return df

    def _should_enter(self, r):
        return (
            r["ema50"] > r["ema200"]
            and r["close"] < r["high20"] * 0.98
            and r["rsi14"] < 35
            and r["close"] > r["open"]
            and r["vol"] > r["vol_mean20"]
        )

    async def _execute_buy(self, price, atr, is_dca=False):
        allocation = min(self.config.allocation_usdc, self.state.balance_usdc)
        if allocation <= 0:
            logging.info(f"{self.symbol_label}: no funds to buy")
            return False
        amount = allocation / price
        self.state.balance_usdc -= allocation
        prev_amt, prev_cost = self.state.amount, self.state.total_invested_usdc
        self.state.amount += amount
        self.state.total_invested_usdc += allocation
        self.state.average_entry_price = self.state.total_invested_usdc / self.state.amount
        self.state.last_buy_price = price
        self.state.has_position = True
        self.state.stop_loss = price - 1.5 * atr
        self._save_state()
        kind = "DCA Buy" if is_dca else "New Buy"
        logging.info(f"{self.symbol_label} {kind} at {price:.2f}, balance {self.state.balance_usdc:.2f}")
        return True

    async def _execute_sell(self, price):
        if self.state.amount <= 0:
            return
        proceeds = price * self.state.amount
        pnl = proceeds - self.state.total_invested_usdc
        self.state.balance_usdc += proceeds
        logging.info(f"{self.symbol_label} SOLD at {price:.2f}, PnL {pnl:.2f}, new balance {self.state.balance_usdc:.2f}")
        self.state = TradeState(balance_usdc=self.state.balance_usdc)
        self._save_state()

    async def _handle_position(self, row):
        if not self.state.has_position:
            return
        price, atr = row["close"], row["atr14"]
        if not self.state.profit_trailing_active and price >= self.state.average_entry_price * (1 + self.config.profit_step_percent / 100):
            self.state.profit_trailing_active = True
            self.state.highest_price = price
            self._save_state()
            logging.info(f"{self.symbol_label}: Profit trailing started at {price:.2f}")
        if self.state.profit_trailing_active and price > self.state.highest_price:
            self.state.highest_price = price
        new_stop = max(self.state.stop_loss, price - 1.5 * atr)
        if self.state.profit_trailing_active:
            new_stop = max(new_stop, self.state.highest_price * (1 - self.config.profit_step_percent / 100))
        if price <= new_stop:
            await self._execute_sell(price)
        else:
            self.state.stop_loss = new_stop
            self._save_state()

    async def _evaluate(self, df):
        if len(df) < 200:
            return
        ind = self._calc_indicators(df)
        r = ind.iloc[-1]
        vals = r[["ema50", "ema200", "rsi14", "atr14", "high20", "vol_mean20"]].astype(float).values
        if np.isnan(vals).any():
            return
        await self._handle_position(r)
        if not self.state.has_position and self._should_enter(r):
            await self._execute_buy(r["close"], r["atr14"])

    async def run(self):
        await self.exchange.load_markets()
        logging.info(f"Starting {self.symbol_label} Trail Accumulator Bot | Mode: {'Live' if self.config.live_mode else 'Paper'} | Balance: ${self.state.balance_usdc:,.2f}")
        try:
            while True:
                logging.info(f"Polling {self.symbol_label} candles...")
                candles = await self._fetch_candles()
                if not candles:
                    await asyncio.sleep(self.config.poll_interval)
                    continue
                logging.info(f"{self.symbol_label} fetched {len(candles)} candles. Last close = {candles[-1][4]:.2f}")
                df = self._build_df(candles)
                ts = int(df["ts"].iloc[-1].timestamp())
                if ts <= self.last_candle_time:
                    await asyncio.sleep(self.config.poll_interval)
                    continue
                self.last_candle_time = ts
                await self._evaluate(df)
                await asyncio.sleep(self.config.poll_interval)
        finally:
            await self.exchange.close()


# ---- Runner ----
async def _run_all(config: Config):
    bots = []
    for s in config.trade_symbols:
        sf = _state_file_for_symbol(config, s)
        sc = replace(config, symbol=s, state_file=sf)
        bots.append(TrailAccumulatorBot(sc))
    await asyncio.gather(*(b.run() for b in bots))
    for b in bots:
        await b.exchange.close()


def main():
    config = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    asyncio.run(_run_all(config))


if __name__ == "__main__":
    main()
