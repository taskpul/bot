"""Trail Accumulator Bot with multi-symbol DCA support.

This script manages Binance spot trades for one or more crypto pairs, buying
bullish dips, averaging into positions on configurable drawdowns, and
protecting gains with ATR- and percentage-based trailing stops. It supports
both paper and live trading, persists per-symbol trade state, and sends
Telegram notifications for key events.
"""

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


@dataclass
class Config:
    symbol: str = "BTC/USDT"
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
    trade_symbols: List[str] = field(default_factory=lambda: ["BTC/USDT"])
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
    allocation_env = os.getenv("TRADE_ALLOCATION_USDC") or os.getenv("TRADE_ALLOCATION_USDT")
    if allocation_env:
        config.allocation_usdc = float(allocation_env)
    poll_env = os.getenv("POLL_INTERVAL")
    if poll_env:
        config.poll_interval = max(30, int(poll_env))
    candles_env = os.getenv("CANDLES_LIMIT")
    if candles_env:
        config.candles_limit = max(120, int(candles_env))
    paper_balance_env = os.getenv("PAPER_BALANCE_USDC") or os.getenv("PAPER_BALANCE_USDT")
    if paper_balance_env:
        config.paper_balance = float(paper_balance_env)
    base_currency_env = os.getenv("BASE_CURRENCY")
    if base_currency_env:
        config.base_currency = base_currency_env.strip().upper()
    dca_env = os.getenv("DCA_DROP_PERCENT")
    if dca_env:
        config.dca_drop_percent = max(0.1, float(dca_env))
    profit_step_env = os.getenv("PROFIT_STEP_PERCENT")
    if profit_step_env:
        config.profit_step_percent = max(0.1, float(profit_step_env))
    symbols_env = os.getenv("TRADE_SYMBOLS")
    if symbols_env:
        symbols = [symbol.strip().upper() for symbol in symbols_env.split(",") if symbol.strip()]
        if symbols:
            config.trade_symbols = symbols
            config.symbol = symbols[0]
    if not config.trade_symbols:
        config.trade_symbols = [config.symbol]
    elif config.symbol not in config.trade_symbols:
        config.trade_symbols.insert(0, config.symbol)
    state_path_env = os.getenv("STATE_FILE")
    if state_path_env:
        config.state_file = Path(state_path_env)
    else:
        symbol_key = config.symbol.replace("/", "_").lower()
        config.state_file = Path(f"{symbol_key}_state.json")
    return config


def _state_file_for_symbol(config: Config, symbol: str) -> Path:
    base = config.state_file
    symbol_key = symbol.replace("/", "_").lower()
    if not base:
        return Path(f"{symbol_key}_state.json")
    if len(config.trade_symbols) <= 1:
        return base
    if base.suffix:
        return base.with_name(f"{base.stem}_{symbol_key}{base.suffix}")
    return base / f"{symbol_key}.json"


class BTCTrailAccumulatorBot:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = self._load_state()
        if not self.state.balance_usdc:
            self.state.balance_usdc = (
                self.config.paper_balance if not self.config.live_mode else 0.0
            )
        base, quote = self.config.symbol.split("/")
        self.base_asset = base
        self.quote_asset = quote
        self.symbol_label = self.config.symbol
        self.profit_step_factor = self.config.profit_step_percent / 100.0
        self.exchange = self._create_exchange()
        self.last_candle_time = 0
        self.market: Optional[Dict[str, Any]] = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _create_exchange(self) -> ccxt_async.binance:
        params: Dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if self.config.live_mode:
            params.update({
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
            })
        return ccxt_async.binance(params)

    def _load_state(self) -> TradeState:
        try:
            if self.config.state_file.exists():
                with open(self.config.state_file, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                return TradeState.from_dict(data)
        except (json.JSONDecodeError, OSError) as exc:
            logging.error("Failed to load state file: %s", exc)
        return TradeState()

    def _save_state(self) -> None:
        try:
            with open(self.config.state_file, "w", encoding="utf-8") as fp:
                json.dump(self.state.to_dict(), fp, indent=2)
        except OSError as exc:
            logging.error("Failed to save state: %s", exc)

    async def _load_markets(self) -> None:
        self.market = await self.exchange.load_markets()

    async def _fetch_balance(self) -> None:
        if not self.config.live_mode:
            return
        try:
            balance = await self.exchange.fetch_balance()
            free_balances = balance.get("free", {}) if balance else {}
            base_currency = self.config.base_currency
            if base_currency in free_balances:
                self.state.balance_usdc = float(free_balances[base_currency])
            else:
                quote_currency = self.quote_asset
                if quote_currency in free_balances:
                    self.state.balance_usdc = float(free_balances[quote_currency])
            self._save_state()
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("Failed to fetch balance: %s", exc)

    async def _fetch_candles(self) -> Optional[List[List[float]]]:
        try:
            return await self.exchange.fetch_ohlcv(
                self.config.symbol,
                timeframe=self.config.timeframe,
                limit=self.config.candles_limit,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("Failed to fetch candles: %s", exc)
            return None

    @staticmethod
    def _build_dataframe(candles: List[List[float]]) -> pd.DataFrame:
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        if talib is not None:
            return pd.Series(talib.EMA(series.to_numpy(), timeperiod=period), index=series.index)
        ema_indicator = ta.trend.EMAIndicator(close=series, window=period)
        return ema_indicator.ema_indicator()

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        if talib is not None:
            return pd.Series(talib.RSI(series.to_numpy(), timeperiod=period), index=series.index)
        rsi_indicator = ta.momentum.RSIIndicator(close=series, window=period)
        return rsi_indicator.rsi()

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        if talib is not None:
            return pd.Series(
                talib.ATR(high.to_numpy(), low.to_numpy(), close.to_numpy(), timeperiod=period),
                index=close.index,
            )
        atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=period)
        return atr_indicator.average_true_range()

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema50"] = self._ema(df["close"], 50)
        df["ema200"] = self._ema(df["close"], 200)
        df["rsi14"] = self._rsi(df["close"], 14)
        df["atr14"] = self._atr(df["high"], df["low"], df["close"], 14)
        df["high20"] = df["high"].rolling(window=20).max()
        df["vol_mean20"] = df["volume"].rolling(window=20).mean()
        return df

    def _format_usdc(self, value: float) -> str:
        return f"${value:,.2f}"

    def _send_telegram(self, message: str) -> None:
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            return
        url = (
            f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
        )
        payload = {"chat_id": self.config.telegram_chat_id, "text": message}
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code >= 400:
                logging.error("Telegram error %s: %s", response.status_code, response.text)
        except requests.RequestException as exc:
            logging.error("Telegram request failed: %s", exc)

    async def _execute_buy(self, price: float, atr: float, is_dca: bool = False) -> bool:
        allocation = self.config.allocation_usdc
        allocation_to_use = allocation
        if not self.config.live_mode:
            allocation_to_use = min(allocation, self.state.balance_usdc)
            if allocation_to_use <= 0:
                logging.info("Insufficient paper balance to execute buy for %s.", self.symbol_label)
                return False
        amount = round(allocation_to_use / price, 6)
        if amount <= 0:
            logging.info("Computed order size is zero for %s, skipping buy.", self.symbol_label)
            return False
        executed_price = price
        total_cost = allocation_to_use
        if self.config.live_mode:
            try:
                order = await self.exchange.create_market_buy_order(self.config.symbol, amount)
                fills = order.get("fills") or []
                if fills:
                    total_cost = sum(float(fill.get("cost", 0.0)) for fill in fills)
                    total_qty = sum(float(fill.get("qty", 0.0)) for fill in fills)
                    if total_qty:
                        executed_price = total_cost / total_qty
                else:
                    total_cost = float(order.get("cost") or total_cost)
                    executed_price = float(order.get("price") or price)
                await self._fetch_balance()
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Buy order failed for %s: %s", self.symbol_label, exc)
                return False
        else:
            self.state.balance_usdc -= allocation_to_use
        previous_amount = self.state.amount
        previous_cost = self.state.total_invested_usdc
        new_amount = previous_amount + amount
        if new_amount <= 0:
            logging.info("New position size would be zero for %s, skipping.", self.symbol_label)
            return False
        self.state.has_position = True
        self.state.amount = new_amount
        self.state.total_invested_usdc = previous_cost + total_cost
        self.state.average_entry_price = self.state.total_invested_usdc / self.state.amount
        self.state.last_buy_price = executed_price
        self.state.profit_trailing_active = False
        self.state.highest_price = 0.0
        atr_stop = executed_price - 1.5 * atr
        if self.state.stop_loss <= 0.0:
            self.state.stop_loss = atr_stop
        else:
            self.state.stop_loss = min(self.state.stop_loss, atr_stop)
        self.state.last_stop_update_ts = int(datetime.now(timezone.utc).timestamp())
        self._save_state()
        action = "DCA Buy Executed" if is_dca else "Dip Buy Executed"
        message = (
            f"{self.symbol_label} {action}\n"
            f"Price: {executed_price:.4f} {self.quote_asset}\n"
            f"Amount: {amount:.6f} {self.base_asset}\n"
            f"Average Entry: {self.state.average_entry_price:.4f} {self.quote_asset}\n"
            f"Allocated: {self.state.total_invested_usdc:.2f} {self.config.base_currency}\n"
            f"Stop: {self.state.stop_loss:.4f} {self.quote_asset}"
        )
        logging.info(message.replace("\n", " | "))
        self._send_telegram(message)
        return True

    async def _execute_sell(self, price: float) -> None:
        amount = self.state.amount
        if amount <= 0:
            return
        executed_price = price
        if self.config.live_mode:
            try:
                order = await self.exchange.create_market_sell_order(self.config.symbol, amount)
                fills = order.get("fills") or []
                if fills:
                    total_cost = sum(float(fill.get("cost", 0.0)) for fill in fills)
                    total_qty = sum(float(fill.get("qty", 0.0)) for fill in fills)
                    if total_cost and total_qty:
                        executed_price = total_cost / total_qty
                else:
                    executed_price = float(order.get("price") or price)
                await self._fetch_balance()
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Sell order failed: %s", exc)
                return
        else:
            proceeds = executed_price * amount
            self.state.balance_usdc += proceeds
        proceeds = executed_price * amount
        pnl = proceeds - self.state.total_invested_usdc
        message = (
            f"{self.symbol_label} Position Closed\n"
            f"Exit Price: {executed_price:.4f} {self.quote_asset}\n"
            f"Amount: {amount:.6f} {self.base_asset}\n"
            f"PnL: {pnl:.2f} {self.config.base_currency}\n"
            f"Balance: {self._format_usdc(self.state.balance_usdc)}"
        )
        logging.info(message.replace("\n", " | "))
        self._send_telegram(message)
        self.state.has_position = False
        self.state.amount = 0.0
        self.state.average_entry_price = 0.0
        self.state.stop_loss = 0.0
        self.state.total_invested_usdc = 0.0
        self.state.last_buy_price = 0.0
        self.state.profit_trailing_active = False
        self.state.highest_price = 0.0
        self.state.last_stop_update_ts = int(datetime.now(timezone.utc).timestamp())
        self._save_state()

    def _should_enter(self, row: pd.Series) -> bool:
        if row["ema50"] <= row["ema200"]:
            return False
        if row["close"] > row["high20"] * 0.98:
            return False
        if row["rsi14"] >= 35:
            return False
        if row["close"] <= row["open"]:
            return False
        if row["volume"] <= row["vol_mean20"]:
            return False
        return True

    async def _handle_position(self, row: pd.Series) -> None:
        if not self.state.has_position:
            return
        price = float(row["close"])
        atr = float(row["atr14"])
        if self.state.last_buy_price > 0 and not self.state.profit_trailing_active:
            trigger_price = self.state.last_buy_price * (1 - self.config.dca_drop_percent / 100.0)
            if price <= trigger_price + 1e-9:
                if await self._execute_buy(price, atr, is_dca=True):
                    return
        if not self.state.profit_trailing_active and self.state.average_entry_price > 0:
            recovery_price = self.state.average_entry_price * (1 + self.profit_step_factor)
            if price >= recovery_price:
                self.state.profit_trailing_active = True
                self.state.highest_price = price
                self.state.last_stop_update_ts = int(datetime.now(timezone.utc).timestamp())
                self._save_state()
                message = (
                    f"{self.symbol_label} Profit Trailing Activated\n"
                    f"Average Entry: {self.state.average_entry_price:.4f} {self.quote_asset}\n"
                    f"Trigger Price: {price:.4f} {self.quote_asset}"
                )
                logging.info(message.replace("\n", " | "))
                self._send_telegram(message)
        if self.state.profit_trailing_active:
            if price > self.state.highest_price:
                self.state.highest_price = price
        updated_stop = self.state.stop_loss
        atr_stop = price - 1.5 * atr
        if atr_stop > updated_stop:
            updated_stop = atr_stop
        if self.state.profit_trailing_active and self.state.highest_price > 0:
            trailing_stop = self.state.highest_price * (1 - self.profit_step_factor)
            if trailing_stop > updated_stop:
                updated_stop = trailing_stop
        if updated_stop > price:
            updated_stop = price * (1 - self.profit_step_factor)
        if updated_stop > self.state.stop_loss + 1e-6:
            self.state.stop_loss = updated_stop
            self.state.last_stop_update_ts = int(datetime.now(timezone.utc).timestamp())
            self._save_state()
            message = (
                f"{self.symbol_label} Trailing Stop Updated\n"
                f"New Stop: {self.state.stop_loss:.4f} {self.quote_asset}\n"
                f"Price: {price:.4f} {self.quote_asset}"
            )
            logging.info(message.replace("\n", " | "))
            self._send_telegram(message)
        if price <= self.state.stop_loss:
            await self._execute_sell(price)

    async def _evaluate(self, df: pd.DataFrame) -> None:
        if len(df) < 200:
            return
        indicators = self._calculate_indicators(df)
        latest = indicators.iloc[-1]
        if np.isnan(latest[["ema50", "ema200", "rsi14", "atr14", "high20", "vol_mean20"]]).any():
            return
        await self._handle_position(latest)
        if self.state.has_position:
            return
        if self._should_enter(latest):
            await self._execute_buy(float(latest["close"]), float(latest["atr14"]))

    async def run(self) -> None:
        await self._load_markets()
        if self.config.live_mode:
            await self._fetch_balance()
        logging.info(
            "Starting %s Trail Accumulator Bot | Mode: %s | Balance: %s",
            self.symbol_label,
            "Live" if self.config.live_mode else "Paper",
            self._format_usdc(self.state.balance_usdc),
        )
        try:
            while True:
                candles = await self._fetch_candles()
                if not candles:
                    await asyncio.sleep(self.config.poll_interval)
                    continue
                df = self._build_dataframe(candles)
                last_ts = int(df["timestamp"].iloc[-1].timestamp())
                if last_ts <= self.last_candle_time:
                    await asyncio.sleep(self.config.poll_interval)
                    continue
                self.last_candle_time = last_ts
                await self._evaluate(df)
                await asyncio.sleep(self.config.poll_interval)
        finally:
            await self.exchange.close()


async def _run_all(config: Config) -> None:
    bots: List[BTCTrailAccumulatorBot] = []
    for symbol in config.trade_symbols:
        state_file = _state_file_for_symbol(config, symbol)
        symbol_config = replace(config, symbol=symbol, state_file=state_file)
        bots.append(BTCTrailAccumulatorBot(symbol_config))
    await asyncio.gather(*(bot.run() for bot in bots))


def main() -> None:
    config = load_config()
    asyncio.run(_run_all(config))


if __name__ == "__main__":
    main()
