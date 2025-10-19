#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive ATR Momentum Bot – v3.6 (USDC Only)
------------------------------------------------------------
Now includes:
• Auto-import of all existing Binance holdings (wallet sync)
• Dynamic SL/TP and ATR trailing management
• Live dashboard, profit tracking, Telegram alerts
• Predictive XGBoost logic and self-learning updates
• Adaptive risk sizing and regime-aware probability thresholds
"""

import os, time, json, warnings, pandas as pd, numpy as np, ccxt, requests, concurrent.futures
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from colorama import Fore, Style, init as color_init
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
color_init(autoreset=True)
load_dotenv()


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ───────────────────────── CONFIG ─────────────────────────
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

QUOTE = "USDC"
TOP_N = 15
MIN_VOLUME = 100_000
MIN_HOLD_VALUE = 2.0
interval = 60
timeframe = "30m"

fast_ema, slow_ema, rsi_period, atr_period = 9, 21, 14, 14
ATR_SL_MULT, ATR_TP_MULT, TRAIL_FACTOR = 1.8, 4.0, 0.6
BASE_RISK_PERCENT = 0.015
MIN_RISK_PERCENT = 0.02
MAX_RISK_PERCENT = 0.03
RISK_STEP = 0.02
RISK_LOOKBACK = 12
BASE_CAPITAL = 1200.0
STATE_FILE = "adaptive_dynamic_dashboard_state.json"
MODEL_DIR = "models"
LIVE_MODE = env_bool("LIVE_MODE", False)
DISABLE_BTC_FILTER = env_bool("DISABLE_BTC_FILTER", False)
DEBUG_MODE = env_bool("DEBUG_MODE", False)
AGGRESSIVE_MODE = DISABLE_BTC_FILTER
MAX_OPEN_POSITIONS = 5
MAX_CAPITAL_EXPOSURE = 0.40
THREADS = 5
lock_factor = 0.5
PROB_THRESHOLD = 0.52 if DISABLE_BTC_FILTER else 0.60
HIGH_VOLATILITY_LEVEL = 0.025
LOW_VOLATILITY_LEVEL = 0.012
TREND_STRENGTH_BONUS = 0.04
FEE_BUFFER = 0.0015
PERFORMANCE_SUFFIX = "_meta.json"
FEE_RATE = float(os.getenv("FEE_RATE", "0.00075"))
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0005"))
MAX_SYMBOL_CORRELATION = float(os.getenv("MAX_SYMBOL_CORRELATION", "0.85"))
TARGET_PORTFOLIO_VOL = float(os.getenv("TARGET_PORTFOLIO_VOL", "0.018"))
PORTFOLIO_LOOKBACK = int(os.getenv("PORTFOLIO_LOOKBACK", "48"))
MIN_INCREMENTAL_BATCH = int(os.getenv("MIN_INCREMENTAL_BATCH", "24"))
BACKTEST_WINDOW = int(os.getenv("BACKTEST_WINDOW", "240"))
BACKTEST_STEP = int(os.getenv("BACKTEST_STEP", "24"))
LOG_DIR = os.getenv("LOG_DIR", "logs")
TRADE_LOG_DIR = Path(os.getenv("TRADE_LOG_DIR", "Logs"))
MAX_TRADE_LOG_DAYS = int(os.getenv("MAX_TRADE_LOG_DAYS", "60"))
CANDLE_STALENESS_TOLERANCE = int(os.getenv("CANDLE_STALENESS_TOLERANCE", "180"))

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)
exchange = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"adjustForTimeDifference": True}
})

try:
    exchange.load_markets()
except Exception as e:
    print(Fore.YELLOW + f"Market metadata load failed: {e}")

if LIVE_MODE and (not API_KEY or not API_SECRET):
    print(Fore.YELLOW + "LIVE_MODE disabled: missing API credentials.")
    LIVE_MODE = False

# ───────────────────────── UTILITIES ───────────────────────
def notify(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception:
        pass


def log_event(name, payload):
    try:
        payload = dict(payload)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        log_path = Path(LOG_DIR) / f"{name}.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def log_decision(symbol, context):
    context = dict(context)
    context.update({"symbol": symbol})
    log_event("decisions", context)


def _prune_trade_logs():
    try:
        logs = []
        for path in TRADE_LOG_DIR.glob("*.log"):
            try:
                log_date = datetime.strptime(path.stem, "%d-%m-%Y")
                logs.append((log_date, path))
            except ValueError:
                continue
        logs.sort(reverse=True)
        for _, old_path in logs[MAX_TRADE_LOG_DAYS:]:
            try:
                old_path.unlink()
            except FileNotFoundError:
                pass
    except Exception:
        pass


def log_trade(event_type, symbol, context):
    payload = dict(context)
    payload.update({"symbol": symbol, "event": event_type})
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    try:
        log_file = TRADE_LOG_DIR / f"{datetime.now(timezone.utc).strftime('%d-%m-%Y')}.log"
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
        _prune_trade_logs()
    except Exception:
        pass


def safe_fetch_balance_total():
    if not LIVE_MODE:
        return {}
    try:
        return exchange.fetch_balance().get("total", {}) or {}
    except ccxt.BaseError as e:
        print(Fore.YELLOW + f"Balance sync skipped: {e}")
    except Exception as e:
        print(Fore.YELLOW + f"Balance sync error: {e}")
    return {}


def safe_fetch_tickers():
    try:
        return exchange.fetch_tickers()
    except ccxt.BaseError as e:
        print(Fore.YELLOW + f"Ticker fetch failed: {e}")
    except Exception as e:
        print(Fore.YELLOW + f"Ticker fetch error: {e}")
    return {}


def fetch_symbol_ohlcv(sym, limit):
    try:
        raw = exchange.fetch_ohlcv(sym, timeframe, limit=limit)
        return pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol"])
    except ccxt.BaseError as e:
        print(Fore.LIGHTBLACK_EX + f"Data fetch failed for {sym}: {e}")
    except Exception as e:
        print(Fore.LIGHTBLACK_EX + f"Data fetch error {sym}: {e}")
    return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])


def candle_is_fresh(df):
    if df.empty:
        return False
    last_ts = float(df["ts"].iloc[-1]) / 1000.0
    return (time.time() - last_ts) <= CANDLE_STALENESS_TOLERANCE

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    if atr.isna().all():
        atr = (df["high"] - df["low"]).rolling(period).mean().fillna(0)
    return atr

def safe_sleep(seconds):
    for _ in range(seconds):
        time.sleep(1)

def load_state():
    if os.path.exists(STATE_FILE):
        s = json.load(open(STATE_FILE))
        s.setdefault("realized_profit", 0.0)
        s.setdefault("daily_profit", 0.0)
        s.setdefault("last_date", datetime.now().strftime("%Y-%m-%d"))
        s.setdefault("trade_history", [])
        s.setdefault("current_risk_percent", BASE_RISK_PERCENT)
        s.setdefault("expected_realized_history", [])
        s.setdefault("threshold_bias", 0.0)
        s.setdefault("holdings_snapshot", {})
        return s
    return {
        "realized_profit": 0.0,
        "daily_profit": 0.0,
        "last_date": datetime.now().strftime("%Y-%m-%d"),
        "trade_history": [],
        "current_risk_percent": BASE_RISK_PERCENT,
        "expected_realized_history": [],
        "threshold_bias": 0.0,
        "holdings_snapshot": {}
    }

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), indent=2)

state = load_state()


def _normalize_holding_entry(sym, entry, default_source="bot"):
    if not isinstance(entry, dict):
        return None
    try:
        amount = float(entry.get("amount", 0) or 0)
        price = float(entry.get("entry_price", 0) or 0)
    except (TypeError, ValueError):
        return None
    if amount <= 0 or price <= 0:
        return None
    if amount * price < MIN_HOLD_VALUE:
        return None
    normalized = {
        "amount": amount,
        "entry_price": price,
        "stop_loss": float(entry.get("stop_loss", price) or price),
        "take_profit": float(entry.get("take_profit", price) or price),
        "expected_return": float(entry.get("expected_return", 0.0) or 0.0),
        "expected_prob": float(entry.get("expected_prob", 0.0) or 0.0),
        "expected_pnl": float(entry.get("expected_pnl", 0.0) or 0.0),
        "entry_time": str(entry.get("entry_time", datetime.now(timezone.utc).isoformat())),
        "source": entry.get("source") or default_source,
    }
    return normalized


def persist_holdings(holdings):
    snapshot = {}
    for sym, entry in holdings.items():
        normalized = _normalize_holding_entry(sym, entry, entry.get("source", "bot") if isinstance(entry, dict) else "bot")
        if normalized:
            snapshot[sym] = normalized
    if state.get("holdings_snapshot") != snapshot:
        state["holdings_snapshot"] = snapshot
        save_state(state)

def effective_capital():
    return BASE_CAPITAL + state["realized_profit"]


def recent_trade_stats():
    history = state.get("trade_history", [])
    if not history:
        return {"win_rate": 0.5, "avg_pnl": 0.0}
    wins = sum(1 for p in history if p > 0)
    win_rate = wins / len(history)
    avg_pnl = float(np.mean(history))
    return {"win_rate": win_rate, "avg_pnl": avg_pnl}


def adjust_risk_from_history():
    stats = recent_trade_stats()
    current = state.get("current_risk_percent", BASE_RISK_PERCENT)
    new_risk = current
    if stats["avg_pnl"] > 0 and stats["win_rate"] > 0.6:
        new_risk = min(MAX_RISK_PERCENT, current + RISK_STEP)
    elif stats["avg_pnl"] < 0 and stats["win_rate"] < 0.4:
        new_risk = max(MIN_RISK_PERCENT, current - RISK_STEP)
    else:
        # mean-revert gently toward baseline when performance is mixed
        if current > BASE_RISK_PERCENT:
            new_risk = max(BASE_RISK_PERCENT, current - RISK_STEP / 2)
        elif current < BASE_RISK_PERCENT:
            new_risk = min(BASE_RISK_PERCENT, current + RISK_STEP / 2)
    state["current_risk_percent"] = round(new_risk, 4)
    save_state(state)
    return state["current_risk_percent"]


def record_trade_pnl(pnl):
    history = state.setdefault("trade_history", [])
    history.append(float(pnl))
    if len(history) > RISK_LOOKBACK:
        del history[0:len(history) - RISK_LOOKBACK]
    adjust_risk_from_history()
    save_state(state)


def record_expected_vs_realized(expected, realized):
    history = state.setdefault("expected_realized_history", [])
    history.append({"expected": float(expected), "realized": float(realized)})
    if len(history) > 100:
        del history[0:len(history) - 100]
    bias = state.get("threshold_bias", 0.0)
    diff = realized - expected
    bias_adjust = np.clip(-diff * 0.02, -0.02, 0.02)
    state["threshold_bias"] = float(np.clip(bias + bias_adjust, -0.08, 0.08))
    save_state(state)


def volatility_risk_modifier(btc_ctx):
    vol = btc_ctx.get("volatility", 0.0)
    momentum = btc_ctx.get("momentum", 0.0)
    if vol > HIGH_VOLATILITY_LEVEL * 1.6:
        return 0.6
    if vol > HIGH_VOLATILITY_LEVEL:
        return 0.8
    if vol < LOW_VOLATILITY_LEVEL / 2 and momentum > 0:
        return 1.15
    if vol < LOW_VOLATILITY_LEVEL:
        return 1.05
    return 1.0


def symbol_returns(df, lookback=PORTFOLIO_LOOKBACK):
    if df is None or df.empty:
        return np.array([])
    series = df["close"].pct_change().dropna()
    if series.empty:
        return np.array([])
    return series.iloc[-lookback:].to_numpy()


def portfolio_volatility(holdings, data_dict):
    if not holdings:
        return 0.0
    returns_list = []
    weights = []
    total_cap = max(effective_capital(), 1e-9)
    for sym, info in holdings.items():
        df = data_dict.get(sym)
        series = symbol_returns(df)
        if len(series) < 5:
            continue
        returns_list.append(series)
        value = info.get("entry_price", 0) * info.get("amount", 0)
        weights.append(value / total_cap)
    if not returns_list:
        return 0.0
    min_len = min(len(r) for r in returns_list)
    aligned = np.array([r[-min_len:] for r in returns_list])
    if aligned.shape[0] == 1:
        return float(np.std(aligned[0]))
    cov = np.cov(aligned)
    weights_arr = np.array(weights[:aligned.shape[0]])
    if weights_arr.sum() == 0:
        return 0.0
    weights_arr = weights_arr / weights_arr.sum()
    portfolio_var = float(weights_arr @ cov @ weights_arr)
    return float(np.sqrt(max(portfolio_var, 0.0)))


def portfolio_risk_adjustment(holdings, data_dict):
    vol = portfolio_volatility(holdings, data_dict)
    if vol == 0:
        return 1.0
    if vol > TARGET_PORTFOLIO_VOL:
        return float(np.clip(TARGET_PORTFOLIO_VOL / vol, 0.3, 1.0))
    if vol < TARGET_PORTFOLIO_VOL / 2:
        return float(np.clip(TARGET_PORTFOLIO_VOL / max(vol, 1e-6), 1.0, 1.4))
    return 1.0


def max_correlation_with_holdings(symbol, candidate_df, holdings, data_dict):
    candidate_returns = symbol_returns(candidate_df)
    if len(candidate_returns) < 5:
        return 0.0
    max_corr = 0.0
    for held_sym in holdings.keys():
        df = data_dict.get(held_sym)
        held_returns = symbol_returns(df)
        if len(held_returns) < 5:
            continue
        min_len = min(len(candidate_returns), len(held_returns))
        if min_len < 5:
            continue
        corr = np.corrcoef(candidate_returns[-min_len:], held_returns[-min_len:])[0, 1]
        if np.isnan(corr):
            continue
        max_corr = max(max_corr, abs(float(corr)))
    return max_corr

# ───────────────────────── MODEL ───────────────────────────
class SymbolModel:
    def __init__(self, sym):
        self.sym = sym
        self.model_path = os.path.join(MODEL_DIR, f"{sym.replace('/', '_')}.json")
        self.scaler_path = os.path.join(MODEL_DIR, f"{sym.replace('/', '_')}_scaler.json")
        self.meta_path = os.path.join(MODEL_DIR, f"{sym.replace('/', '_')}{PERFORMANCE_SUFFIX}")
        self.model = XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=3)
        self.scaler = StandardScaler()
        self.trained, self.last_train = False, 0
        self.last_train_rows = 0
        self.performance = {
            "precision": 0.5,
            "recall": 0.5,
            "avg_forward_return": 0.0,
            "threshold_bonus": 0.0,
            "walk_forward_precision": 0.0,
            "walk_forward_return": 0.0,
            "expected_forward_return": 0.0,
            "execution_cost": FEE_RATE * 2 + SLIPPAGE_RATE
        }
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.model_path):
                self.model.load_model(self.model_path)
                if os.path.exists(self.scaler_path):
                    p = json.load(open(self.scaler_path))
                    self.scaler.mean_ = np.array(p["mean"])
                    self.scaler.scale_ = np.array(p["scale"])
                self.trained = True
            if os.path.exists(self.meta_path):
                self.performance.update(json.load(open(self.meta_path)))
            self.last_train_rows = int(self.performance.get("last_train_rows", 0))
        except Exception:
            pass

    def _save(self):
        try:
            self.model.save_model(self.model_path)
            json.dump({"mean": self.scaler.mean_.tolist(),
                       "scale": self.scaler.scale_.tolist()}, open(self.scaler_path, "w"))
            self.performance["last_train_rows"] = int(self.last_train_rows)
            json.dump(self.performance, open(self.meta_path, "w"), indent=2)
        except Exception:
            pass

    def prepare_df(self, df):
        df = df.copy()
        df.loc[:, "ema_fast"] = df["close"].ewm(span=fast_ema).mean()
        df.loc[:, "ema_slow"] = df["close"].ewm(span=slow_ema).mean()
        df.loc[:, "rsi"] = calc_rsi(df["close"], rsi_period)
        df.loc[:, "atr"] = calc_atr(df, atr_period)
        df.loc[:, "body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
        df.loc[:, "vol_ratio"] = (df["vol"] / df["vol"].rolling(20).mean()).fillna(1)
        df.loc[:, "atr_change"] = df["atr"].pct_change().fillna(0)
        df.loc[:, "ema_ratio"] = (df["ema_fast"] / (df["ema_slow"] + 1e-9)) - 1
        df.loc[:, "price_momentum"] = df["close"].pct_change().rolling(5).mean().fillna(0)
        vol_mean = df["vol"].rolling(30).mean()
        vol_std = df["vol"].rolling(30).std().replace(0, np.nan)
        df.loc[:, "volume_zscore"] = ((df["vol"] - vol_mean) / vol_std).fillna(0)
        df.loc[:, "volatility_regime"] = (df["atr"] / (df["close"] + 1e-9)).rolling(20).mean().bfill().fillna(0)
        df.loc[:, "trend_slope"] = (df["ema_fast"] - df["ema_slow"]).diff().fillna(0)
        df.loc[:, "momentum_3"] = df["close"].pct_change(periods=3).fillna(0)
        df.loc[:, "rsi_slope"] = df["rsi"].diff().fillna(0)
        df.loc[:, "volume_trend"] = df["vol"].pct_change().rolling(3).mean().fillna(0)
        ma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std().replace(0, np.nan)
        df.loc[:, "bollinger_pos"] = ((df["close"] - ma20) / (2 * std20)).clip(-3, 3).fillna(0)
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        return df

    def update(self, df):
        df = self.prepare_df(df)
        feature_matrix = np.column_stack([
            df["ema_fast"] - df["ema_slow"],
            df["rsi"], df["atr"] / df["close"],
            df["body_ratio"], df["vol_ratio"], df["atr_change"],
            df["ema_ratio"], df["price_momentum"], df["volume_zscore"],
            df["volatility_regime"], df["trend_slope"], df["momentum_3"],
            df["rsi_slope"], df["volume_trend"], df["bollinger_pos"]
        ])
        forward_returns = (df["close"].shift(-1) / df["close"]) - 1
        execution_cost = self.performance.get("execution_cost", FEE_RATE * 2 + SLIPPAGE_RATE)
        adjusted_forward = forward_returns - execution_cost
        y = (adjusted_forward > 0).astype(int)
        if len(feature_matrix) <= 40:
            return
        try:
            X_full = feature_matrix[:-1]
            y_full = y[:-1]
            valid_mask = ~np.isnan(y_full)
            X_full = X_full[valid_mask]
            y_full = y_full[valid_mask]
            if len(X_full) <= 40:
                return
            new_rows = len(X_full) - self.last_train_rows
            if not self.trained or new_rows >= MIN_INCREMENTAL_BATCH:
                start_idx = max(0, len(X_full) - max(BACKTEST_WINDOW, MIN_INCREMENTAL_BATCH * 5))
                recent_X = X_full[start_idx:]
                recent_y = y_full[start_idx:]
                if len(recent_X) <= 30:
                    return
                val_size = max(10, int(len(recent_X) * 0.2))
                if len(recent_X) - val_size <= 20:
                    val_size = 0
                if val_size:
                    X_train_raw, X_val_raw = recent_X[:-val_size], recent_X[-val_size:]
                    y_train, y_val = recent_y[:-val_size], recent_y[-val_size:]
                else:
                    X_train_raw, y_train = recent_X, recent_y
                    X_val_raw, y_val = np.empty((0, recent_X.shape[1])), np.array([])

                self.scaler.fit(X_train_raw)
                X_train = self.scaler.transform(X_train_raw)
                if val_size:
                    X_val = self.scaler.transform(X_val_raw)
                else:
                    X_val = np.empty((0, X_train.shape[1]))

                try:
                    if self.trained:
                        self.model.fit(X_train, y_train, xgb_model=self.model.get_booster())
                    else:
                        self.model.fit(X_train, y_train)
                except Exception:
                    self.model = XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=3)
                    self.model.fit(X_train, y_train)
                self.trained = True
                self.last_train = time.time()
                self.last_train_rows = len(X_full)

                if val_size and len(np.unique(y_val)) > 1:
                    val_probs = self.model.predict_proba(X_val)[:, 1]
                    val_preds = (val_probs >= 0.5).astype(int)
                    precision = precision_score(y_val, val_preds, zero_division=0)
                    recall = recall_score(y_val, val_preds, zero_division=0)
                    avg_forward = float(np.nanmean(adjusted_forward[-val_size:]))
                    bonus = 0.0
                    if precision > 0.6:
                        bonus -= 0.05
                    elif precision < 0.45:
                        bonus += 0.05
                    if avg_forward > execution_cost * 2:
                        bonus -= 0.02
                    elif avg_forward < 0:
                        bonus += 0.02
                    self.performance.update({
                        "precision": round(float(precision), 4),
                        "recall": round(float(recall), 4),
                        "avg_forward_return": round(avg_forward, 6),
                        "threshold_bonus": float(np.clip(bonus, -0.08, 0.08)),
                        "execution_cost": execution_cost
                    })
                else:
                    self.performance.update({
                        "precision": 0.5,
                        "recall": 0.5,
                        "avg_forward_return": float(np.nanmean(adjusted_forward)) if len(adjusted_forward) else 0.0,
                        "threshold_bonus": 0.0,
                        "execution_cost": execution_cost
                    })
                wf = self.walk_forward_validate(df)
                if wf:
                    self.performance.update(wf)
                self._save()
        except Exception:
            pass

    def walk_forward_validate(self, df):
        try:
            prepared = self.prepare_df(df)
            returns = (prepared["close"].shift(-1) / prepared["close"]) - 1
            cost = self.performance.get("execution_cost", FEE_RATE * 2 + SLIPPAGE_RATE)
            metrics = []
            for start in range(0, len(prepared) - BACKTEST_WINDOW - BACKTEST_STEP, BACKTEST_STEP):
                train_slice = prepared.iloc[start:start + BACKTEST_WINDOW]
                test_slice = prepared.iloc[start + BACKTEST_WINDOW:start + BACKTEST_WINDOW + BACKTEST_STEP]
                if len(train_slice) < 50 or len(test_slice) < 5:
                    continue
                X_train = np.column_stack([
                    train_slice["ema_fast"] - train_slice["ema_slow"],
                    train_slice["rsi"], train_slice["atr"] / train_slice["close"],
                    train_slice["body_ratio"], train_slice["vol_ratio"], train_slice["atr_change"],
                    train_slice["ema_ratio"], train_slice["price_momentum"], train_slice["volume_zscore"],
                    train_slice["volatility_regime"], train_slice["trend_slope"], train_slice["momentum_3"],
                    train_slice["rsi_slope"], train_slice["volume_trend"], train_slice["bollinger_pos"]
                ])
                y_train = ((train_slice["close"].shift(-1) / train_slice["close"] - 1) - cost > 0).astype(int)[:-1]
                X_train = X_train[:-1]
                if len(np.unique(y_train)) < 2:
                    continue
                scaler = StandardScaler()
                scaler.fit(X_train)
                model = XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=3)
                model.fit(scaler.transform(X_train), y_train)

                X_test = np.column_stack([
                    test_slice["ema_fast"] - test_slice["ema_slow"],
                    test_slice["rsi"], test_slice["atr"] / test_slice["close"],
                    test_slice["body_ratio"], test_slice["vol_ratio"], test_slice["atr_change"],
                    test_slice["ema_ratio"], test_slice["price_momentum"], test_slice["volume_zscore"],
                    test_slice["volatility_regime"], test_slice["trend_slope"], test_slice["momentum_3"],
                    test_slice["rsi_slope"], test_slice["volume_trend"], test_slice["bollinger_pos"]
                ])
                probs = model.predict_proba(scaler.transform(X_test))[:, 1]
                forward = returns.iloc[start + BACKTEST_WINDOW:start + BACKTEST_WINDOW + BACKTEST_STEP] - cost
                forward_array = forward.to_numpy()
                expected = float(np.nanmean(forward_array))
                preds = probs >= 0.5
                if preds.any():
                    realized = float(np.nanmean(forward_array[preds]))
                else:
                    realized = 0.0
                precision = precision_score((forward_array > 0).astype(int), preds.astype(int), zero_division=0)
                metrics.append({
                    "precision": precision,
                    "expected": expected,
                    "realized": realized
                })
            if metrics:
                avg_precision = float(np.mean([m["precision"] for m in metrics]))
                avg_expected = float(np.mean([m["expected"] for m in metrics]))
                avg_realized = float(np.mean([m["realized"] for m in metrics]))
                return {
                    "walk_forward_precision": round(avg_precision, 4),
                    "walk_forward_return": round(avg_realized, 6),
                    "expected_forward_return": round(avg_expected, 6)
                }
        except Exception:
            return {}
        return {}

    def predict(self, df):
        if not self.trained:
            return 0.5
        try:
            if "ema_fast" not in df or "bollinger_pos" not in df:
                df = self.prepare_df(df)
            r = np.array([[
                df["ema_fast"].iloc[-1] - df["ema_slow"].iloc[-1],
                df["rsi"].iloc[-1], df["atr"].iloc[-1] / df["close"].iloc[-1],
                df["body_ratio"].iloc[-1], df["vol_ratio"].iloc[-1], df["atr_change"].iloc[-1],
                df["ema_ratio"].iloc[-1], df["price_momentum"].iloc[-1], df["volume_zscore"].iloc[-1],
                df["volatility_regime"].iloc[-1], df["trend_slope"].iloc[-1], df["momentum_3"].iloc[-1],
                df["rsi_slope"].iloc[-1], df["volume_trend"].iloc[-1], df["bollinger_pos"].iloc[-1]
            ]])
            r = self.scaler.transform(r)
            return float(self.model.predict_proba(r)[0, 1])
        except Exception:
            return 0.5

    def adjusted_threshold(self, base_threshold):
        bonus = self.performance.get("threshold_bonus", 0.0) if self.performance else 0.0
        return min(max(base_threshold + bonus, 0.5), 0.9)

    def risk_modifier(self):
        precision = self.performance.get("precision", 0.5)
        avg_ret = self.performance.get("avg_forward_return", 0.0)
        modifier = 0.9 + (precision - 0.5) * 0.6
        modifier += np.clip(avg_ret * 10, -0.1, 0.1)
        return float(np.clip(modifier, 0.5, 1.25))

# ───────────────────────── DISCOVERY & DASHBOARD ───────────
def discover_symbols():
    syms = []
    try:
        tick = safe_fetch_tickers()
        for s, t in tick.items():
            if not s.endswith("/" + QUOTE):
                continue
            v = t.get("quoteVolume", 0) or 0
            if v > MIN_VOLUME:
                syms.append((s, v))
        syms = [s for s, _ in sorted(syms, key=lambda x: x[1], reverse=True)[:TOP_N]]
    except Exception as e:
        print(Fore.RED + f"Discovery error {e}")
    print(Fore.CYAN + f"Tracking {len(syms)} high-volume {QUOTE} pairs")
    return syms

def fetch_btc_context():
    try:
        df = fetch_symbol_ohlcv("BTC/USDC", slow_ema + 120)
        if df.empty:
            raise ValueError("no btc data")
        df["atr"] = calc_atr(df, atr_period)
        ef = df["close"].ewm(span=fast_ema).mean()
        es = df["close"].ewm(span=slow_ema).mean()
        bullish = bool(ef.iloc[-1] > es.iloc[-1])
        vol_ratio = float(df["atr"].iloc[-1] / df["close"].iloc[-1])
        trend_strength = float((ef.iloc[-1] - es.iloc[-1]) / df["close"].iloc[-1])
        momentum = float(df["close"].pct_change().ewm(span=5).mean().iloc[-1])
        return {
            "bullish": bullish,
            "volatility": max(vol_ratio, 0.0),
            "trend_strength": trend_strength,
            "momentum": momentum
        }
    except Exception:
        return {"bullish": True, "volatility": 0.0, "trend_strength": 0.0, "momentum": 0.0}


def dynamic_entry_threshold(btc_ctx):
    threshold = PROB_THRESHOLD
    vol = btc_ctx.get("volatility", 0.0)
    trend_strength = btc_ctx.get("trend_strength", 0.0)
    if vol > HIGH_VOLATILITY_LEVEL:
        threshold += 0.05
    elif vol < LOW_VOLATILITY_LEVEL:
        threshold -= 0.03
    if trend_strength > 0.02:
        threshold -= TREND_STRENGTH_BONUS
    elif trend_strength < -0.02:
        threshold += TREND_STRENGTH_BONUS
    threshold += state.get("threshold_bias", 0.0)
    return min(max(threshold, 0.5), 0.9)

def execute_buy(sym, amt):
    if AGGRESSIVE_MODE:
        order = exchange.create_market_buy_order(sym, amt)
        print(Fore.GREEN + f"BUY {sym} {amt:.4f} (aggressive)")
        notify(f"BUY {sym} {amt:.4f} (aggressive)")
        return order
    try:
        for attempt in range(2):
            ticker = exchange.fetch_ticker(sym)
            current_price = float(ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask") or 0.0)
            if current_price <= 0:
                raise ValueError("Invalid price for buy order")
            target_price = current_price * 0.9995
            order = exchange.create_limit_buy_order(sym, amt, target_price)
            start = time.time()
            while time.time() - start < 60:
                try:
                    status = exchange.fetch_order(order["id"], sym)
                except Exception:
                    time.sleep(2)
                    continue
                order_status = str(status.get("status", "")).lower()
                filled = float(status.get("filled") or 0.0)
                amount = float(status.get("amount") or 0.0)
                if order_status == "closed" or (amount and filled >= amount):
                    print(Fore.GREEN + f"BUY {sym} {amt:.4f}")
                    notify(f"BUY {sym} {amt:.4f}")
                    return status
                time.sleep(2)
            try:
                exchange.cancel_order(order["id"], sym)
            except Exception:
                pass
        print(Fore.YELLOW + f"BUY skipped {sym} {amt:.4f} (limit not filled)")
    except Exception as e:
        print(Fore.RED + f"BUY error {sym}: {e}")
        return None

def execute_sell(sym, amt):
    if AGGRESSIVE_MODE:
        order = exchange.create_market_sell_order(sym, amt)
        print(Fore.GREEN + f"SELL {sym} {amt:.4f} (aggressive)")
        notify(f"SELL {sym} {amt:.4f} (aggressive)")
        return order
    try:
        for attempt in range(2):
            ticker = exchange.fetch_ticker(sym)
            current_price = float(ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask") or 0.0)
            if current_price <= 0:
                raise ValueError("Invalid price for sell order")
            target_price = current_price * 1.0005
            order = exchange.create_limit_sell_order(sym, amt, target_price)
            start = time.time()
            while time.time() - start < 60:
                try:
                    status = exchange.fetch_order(order["id"], sym)
                except Exception:
                    time.sleep(2)
                    continue
                order_status = str(status.get("status", "")).lower()
                filled = float(status.get("filled") or 0.0)
                amount = float(status.get("amount") or 0.0)
                if order_status == "closed" or (amount and filled >= amount):
                    print(Fore.GREEN + f"SELL {sym} {amt:.4f}")
                    notify(f"SELL {sym} {amt:.4f}")
                    return status
                time.sleep(2)
            try:
                exchange.cancel_order(order["id"], sym)
            except Exception:
                pass
        print(Fore.YELLOW + f"SELL skipped {sym} {amt:.4f} (limit not filled)")
    except Exception as e:
        print(Fore.RED + f"SELL error {sym}: {e}")
        return None

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def build_exchange_holding(sym, amt, price=None):
    try:
        amount = float(amt)
    except (TypeError, ValueError):
        return None
    if amount <= 0:
        return None
    markets = getattr(exchange, "markets", {})
    if sym not in markets:
        try:
            exchange.load_markets(reload=True)
            markets = getattr(exchange, "markets", {})
        except Exception:
            markets = {}
    if sym not in markets:
        return None
    try:
        ticker = exchange.fetch_ticker(sym) if price is None else {"last": price}
        last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
    except Exception as e:
        print(Fore.LIGHTBLACK_EX + f"Skip holding sync for {sym}: {e}")
        return None
    if last_price <= 0:
        return None
    value = amount * last_price
    if value < MIN_HOLD_VALUE:
        return None
    df = fetch_symbol_ohlcv(sym, slow_ema + 60)
    if df.empty:
        return None
    df["atr"] = calc_atr(df, atr_period)
    atr_series = df["atr"].replace([np.inf, -np.inf], np.nan).ffill()
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = last_price * 0.02
    atr = max(atr_value, 1e-9)
    entry = {
        "amount": amount,
        "entry_price": last_price,
        "stop_loss": last_price - ATR_SL_MULT * atr,
        "take_profit": last_price + ATR_TP_MULT * atr,
        "expected_return": 0.0,
        "expected_prob": 0.0,
        "expected_pnl": 0.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "source": "exchange"
    }
    return entry


def load_existing_holdings():
    holdings = {}
    snapshot = state.get("holdings_snapshot", {}) or {}
    for sym, entry in snapshot.items():
        normalized = _normalize_holding_entry(sym, entry, entry.get("source", "state") if isinstance(entry, dict) else "state")
        if normalized:
            value = normalized["amount"] * normalized["entry_price"]
            holdings[sym] = normalized
            print(Fore.CYAN + f"Recovered {sym} from snapshot: {normalized['amount']:.6f} (${value:.2f})")

    balances = safe_fetch_balance_total()
    if balances:
        for asset, amt in balances.items():
            if asset == QUOTE:
                continue
            sym = f"{asset}/{QUOTE}"
            entry = build_exchange_holding(sym, amt)
            if entry is None:
                continue
            holdings[sym] = entry
            print(Fore.CYAN + f"Loaded existing {sym}: {entry['amount']:.6f} (${entry['amount'] * entry['entry_price']:.2f})")

    if holdings:
        persist_holdings(holdings)
    return holdings


def sync_holdings_with_exchange(holdings):
    if not LIVE_MODE:
        if holdings:
            persist_holdings(holdings)
        return
    balances = safe_fetch_balance_total()
    if not balances:
        if holdings:
            persist_holdings(holdings)
        return
    seen = set()
    for asset, amt in balances.items():
        if asset == QUOTE:
            continue
        sym = f"{asset}/{QUOTE}"
        try:
            amount = float(amt)
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        existing_entry = holdings.get(sym)
        if existing_entry:
            try:
                entry_price = float(existing_entry.get("entry_price", 0) or 0)
            except (TypeError, ValueError):
                entry_price = 0.0
            if entry_price > 0 and entry_price * amount < MIN_HOLD_VALUE:
                print(Fore.LIGHTBLACK_EX + f"Removing synced holding {sym}: value below minimum (${entry_price * amount:.2f})")
                holdings.pop(sym, None)
                continue
        seen.add(sym)
        if sym in holdings:
            holdings[sym]["amount"] = amount
            if "source" not in holdings[sym]:
                holdings[sym]["source"] = "exchange"
        else:
            entry = build_exchange_holding(sym, amount)
            if entry is None:
                continue
            holdings[sym] = entry
            print(Fore.CYAN + f"Synced new holding {sym}: {entry['amount']:.6f} (${entry['amount'] * entry['entry_price']:.2f})")
    for sym in list(holdings.keys()):
        if holdings[sym].get("source") == "exchange" and sym not in seen:
            print(Fore.LIGHTBLACK_EX + f"Removing synced holding {sym}: no balance detected")
            holdings.pop(sym, None)
    persist_holdings(holdings)

def print_dashboard(tickers, holdings, state, interval, next_sym, next_prob, current_risk, threshold, btc_ctx):
    clear_console()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(Fore.MAGENTA + Style.BRIGHT + "══════════ ADAPTIVE ATR MOMENTUM BOT v3.6 DASHBOARD ══════════")
    print(Fore.CYAN + f"Last Updated: {now}")
    print(Fore.CYAN + f"Next Run In: {interval}s")
    print(Fore.WHITE + "-"*65)
    total_cap = effective_capital()
    used_cap = sum(h["entry_price"] * h["amount"] for h in holdings.values())
    free_cap = total_cap - used_cap
    profit_color = Fore.GREEN if state["daily_profit"] >= 0 else Fore.RED
    print(Fore.GREEN + f"Total Capital: ${total_cap:,.2f}")
    print(Fore.YELLOW + f"Used Capital:  ${used_cap:,.2f}")
    print(Fore.WHITE + f"Free Capital:  ${free_cap:,.2f}")
    stats = recent_trade_stats()
    print(Fore.WHITE + f"Dynamic Risk:  {current_risk*100:.1f}% (win {stats['win_rate']*100:.0f}% avgPnL {stats['avg_pnl']:+.2f})")
    bias = state.get("threshold_bias", 0.0)
    port_mod = state.get("last_portfolio_modifier", 1.0)
    print(Fore.WHITE + f"Entry Threshold: {threshold:.2f} (bias {bias:+.3f})  BTC vol {btc_ctx.get('volatility',0):.3f} trend {btc_ctx.get('trend_strength',0):+.3f}  PortMod {port_mod:.2f}")
    if state.get("expected_realized_history"):
        last_exp = state["expected_realized_history"][-1]
        print(Fore.WHITE + f"Last Exp/Realized: {last_exp['expected']:+.2f} / {last_exp['realized']:+.2f}")
    print(profit_color + f"Today's P/L:   {state['daily_profit']:+.2f} USDC")
    print(Fore.WHITE + "-"*65)
    print(Fore.CYAN + Style.BRIGHT + "CURRENT HOLDINGS")
    if not holdings:
        print(Fore.LIGHTBLACK_EX + "  None")
    else:
        for sym, h in holdings.items():
            if sym in tickers:
                price = tickers[sym]["last"]
                pnl = (price - h["entry_price"]) * h["amount"]
                color = Fore.GREEN if pnl >= 0 else Fore.RED
                print(color + f"  {sym:<10} Price: {price:.4f}  PnL: {pnl:+.2f}  SL: {h['stop_loss']:.4f}  TP: {h['take_profit']:.4f}")
    print(Fore.WHITE + "-"*65)
    if next_sym:
        print(Fore.BLUE + Style.BRIGHT + f"Next Target Symbol: {next_sym}  (Prob: {next_prob:.2f})")
    else:
        print(Fore.LIGHTBLACK_EX + "No qualifying entry signal yet.")
    print(Fore.WHITE + "-"*65)
    print(Fore.LIGHTBLACK_EX + "Press CTRL+C to exit.\n")


def print_debug_status(tickers, holdings, state, interval, next_sym, next_prob, current_risk, threshold, btc_ctx):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_cap = effective_capital()
    used_cap = sum(h["entry_price"] * h["amount"] for h in holdings.values())
    free_cap = total_cap - used_cap
    stats = recent_trade_stats()
    bias = state.get("threshold_bias", 0.0)
    port_mod = state.get("last_portfolio_modifier", 1.0)
    summary = (
        f"[{now}] Next run in {interval}s | Holdings: {len(holdings)} | Free cap: ${free_cap:,.2f} | "
        f"Risk: {current_risk*100:.1f}% (win {stats['win_rate']*100:.0f}% avgPnL {stats['avg_pnl']:+.2f}) | "
        f"Threshold: {threshold:.2f} (bias {bias:+.3f}) | BTC vol {btc_ctx.get('volatility',0):.3f} trend {btc_ctx.get('trend_strength',0):+.3f} | PortMod {port_mod:.2f}"
    )
    print(Fore.CYAN + summary)
    if next_sym:
        print(Fore.BLUE + f"  Next target: {next_sym} (prob {next_prob:.2f})")
    else:
        print(Fore.LIGHTBLACK_EX + "  No qualifying entry signal yet.")
    if holdings:
        print(Fore.CYAN + "  Holdings snapshot:")
        for sym, h in holdings.items():
            if sym not in tickers:
                continue
            price = tickers[sym]["last"]
            pnl = (price - h["entry_price"]) * h["amount"]
            color = Fore.GREEN if pnl >= 0 else Fore.RED
            print(color + f"    {sym:<10} price {price:.4f} pnl {pnl:+.2f} sl {h['stop_loss']:.4f} tp {h['take_profit']:.4f}")
    print(Fore.WHITE + "-"*65)


def render_status(*args, **kwargs):
    if DEBUG_MODE:
        print_debug_status(*args, **kwargs)
    else:
        print_dashboard(*args, **kwargs)

# ───────────────────────── MAIN LOOP ───────────────────────
symbols = discover_symbols()
models = {s: SymbolModel(s) for s in symbols}
holdings = load_existing_holdings()
print(Fore.MAGENTA + "Starting Adaptive ATR Momentum Bot v3.6 (USDC only)")
if DEBUG_MODE:
    print(Fore.YELLOW + "DEBUG_MODE enabled: streaming loop logs instead of dashboard UI.")
else:
    print(Fore.YELLOW + "Dashboard mode active. Set DEBUG_MODE=1 to view streaming logs.")

while True:
    try:
        sync_holdings_with_exchange(holdings)
        btc_ctx = fetch_btc_context()
        if not DISABLE_BTC_FILTER:
            if (not btc_ctx["bullish"]) and btc_ctx["trend_strength"] < -0.01:
                print(Fore.LIGHTBLACK_EX + "BTC downtrend — blocking entries.")
                safe_sleep(interval)
                continue

        symbols = discover_symbols()
        for s in symbols:
            models.setdefault(s, SymbolModel(s))

        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
            data = list(ex.map(lambda s: (s, fetch_symbol_ohlcv(s, slow_ema + 120)), symbols))
        tickers = safe_fetch_tickers()
        data_dict = {sym: df for sym, df in data}
        for held_sym in holdings.keys():
            if held_sym not in data_dict:
                data_dict[held_sym] = fetch_symbol_ohlcv(held_sym, slow_ema + 120)

        entry_threshold = 0.52 if AGGRESSIVE_MODE else dynamic_entry_threshold(btc_ctx)
        current_risk = state.get("current_risk_percent", BASE_RISK_PERCENT)
        portfolio_mod = portfolio_risk_adjustment(holdings, data_dict)
        state["last_portfolio_modifier"] = portfolio_mod

        best_sym, best_score, best_prob, best_price, best_atr = None, -np.inf, 0, 0, 0
        best_threshold, best_risk_mod, best_expected = entry_threshold, 1.0, 0.0
        for sym, df in data:
            if df.empty:
                log_decision(sym, {"reason": "no_data"})
                continue
            m = models[sym]
            if time.time() - m.last_train > 3600:
                m.update(df)
            if not candle_is_fresh(df):
                log_decision(sym, {"reason": "stale_candle"})
                continue
            prepared = m.prepare_df(df)
            p = m.predict(prepared)
            price, atr = prepared["close"].iloc[-1], prepared["atr"].iloc[-1]
            sym_threshold = m.adjusted_threshold(entry_threshold)
            risk_mod = m.risk_modifier()
            expected_forward = m.performance.get("expected_forward_return", m.performance.get("avg_forward_return", 0.0))
            execution_cost = m.performance.get("execution_cost", FEE_RATE * 2 + SLIPPAGE_RATE)
            net_expected = expected_forward - execution_cost
            corr = max_correlation_with_holdings(sym, df, holdings, data_dict)
            walk_precision = m.performance.get("walk_forward_precision", 0.0)
            decision_context = {
                "probability": round(float(p), 4),
                "threshold": round(float(sym_threshold), 4),
                "expected_forward": round(float(expected_forward), 6),
                "net_expected": round(float(net_expected), 6),
                "execution_cost": round(float(execution_cost), 6),
                "walk_precision": round(float(walk_precision), 4),
                "correlation": round(float(corr), 4)
            }
            if not AGGRESSIVE_MODE:
                if net_expected <= 0:
                    decision_context["status"] = "skip_low_expectation"
                    log_decision(sym, decision_context)
                    continue
                if walk_precision < 0.45:
                    decision_context["status"] = "skip_validation"
                    log_decision(sym, decision_context)
                    continue
                if corr >= MAX_SYMBOL_CORRELATION:
                    decision_context["status"] = "skip_correlation"
                    log_decision(sym, decision_context)
                    continue
            decision_context["status"] = "candidate"
            log_decision(sym, decision_context)
            score = float(p) if AGGRESSIVE_MODE else float(p * max(net_expected, 1e-6))
            if score > best_score:
                best_sym, best_score = sym, score
                best_prob, best_price, best_atr = p, price, atr
                best_threshold = 0.52 if AGGRESSIVE_MODE else sym_threshold
                best_risk_mod, best_expected = risk_mod, net_expected

        if best_sym and best_sym not in holdings and (AGGRESSIVE_MODE or best_prob > best_threshold):
            corr = max_correlation_with_holdings(best_sym, data_dict.get(best_sym), holdings, data_dict)
            if AGGRESSIVE_MODE or corr < MAX_SYMBOL_CORRELATION:
                used_cap = sum(h["entry_price"] * h["amount"] for h in holdings.values())
                if len(holdings) < MAX_OPEN_POSITIONS and used_cap < effective_capital() * MAX_CAPITAL_EXPOSURE:
                    vol_mod = 1.0 if AGGRESSIVE_MODE else volatility_risk_modifier(btc_ctx)
                    alloc = current_risk * best_risk_mod * vol_mod * portfolio_mod * effective_capital()
                    max_alloc = max(effective_capital() * MAX_CAPITAL_EXPOSURE - used_cap, 0)
                    alloc = min(alloc, max_alloc)
                    if alloc > 0:
                        amt = alloc / best_price
                        if LIVE_MODE:
                            execute_buy(best_sym, amt)
                        expected_pnl = alloc * best_expected
                        holdings[best_sym] = {
                            "amount": amt,
                            "entry_price": best_price,
                            "stop_loss": best_price - ATR_SL_MULT * best_atr,
                            "take_profit": best_price + ATR_TP_MULT * best_atr,
                            "expected_return": best_expected,
                            "expected_prob": best_prob,
                            "expected_pnl": expected_pnl,
                            "entry_time": datetime.now(timezone.utc).isoformat(),
                            "source": "bot"
                        }
                        log_trade("entry", best_sym, {
                            "amount": amt,
                            "price": best_price,
                            "prob": best_prob,
                            "expected_return": best_expected,
                            "expected_pnl": expected_pnl,
                            "allocation": alloc,
                            "threshold": best_threshold,
                            "risk_mod": best_risk_mod,
                            "portfolio_mod": portfolio_mod
                        })
                        persist_holdings(holdings)
            else:
                log_decision(best_sym, {"reason": "blocked_correlation", "correlation": corr})

        # Exit Management
        for sym, h in list(holdings.items()):
            if sym not in tickers:
                continue
            price = tickers[sym]["last"]
            pnl = (price - h["entry_price"]) * h["amount"]
            pnl_ratio = (price - h["entry_price"]) / h["entry_price"]

            if pnl > 0:
                new_stop = h["entry_price"] * (1 + pnl_ratio * lock_factor)
                h["stop_loss"] = max(h["stop_loss"], new_stop)

            if price <= h["stop_loss"] or price >= h["take_profit"]:
                pnl = (price - h["entry_price"]) * h["amount"]
                state["realized_profit"] += pnl
                today = datetime.now().strftime("%Y-%m-%d")
                if today != state.get("last_date"):
                    state["last_date"] = today
                    state["daily_profit"] = 0.0
                state["daily_profit"] += pnl
                record_trade_pnl(pnl)
                expected_pnl = h.get("expected_pnl", 0.0)
                record_expected_vs_realized(expected_pnl, pnl)
                exit_reason = "stop_loss" if price <= h["stop_loss"] else "take_profit"
                hold_duration = None
                if h.get("entry_time"):
                    try:
                        entry_dt = datetime.fromisoformat(h["entry_time"])
                        if entry_dt.tzinfo is None:
                            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                        hold_duration = (datetime.now(timezone.utc) - entry_dt).total_seconds()
                    except Exception:
                        hold_duration = None
                c = Fore.GREEN if pnl >= 0 else Fore.RED
                print(c + f"EXIT {sym} PnL ${pnl:.2f} Total ${state['realized_profit']:.2f}")
                notify(f"EXIT {sym} PnL ${pnl:.2f}")
                log_trade("exit", sym, {
                    "amount": h["amount"],
                    "exit_price": price,
                    "pnl": pnl,
                    "expected_pnl": expected_pnl,
                    "exit_reason": exit_reason,
                    "hold_seconds": hold_duration
                })
                if LIVE_MODE:
                    execute_sell(sym, h["amount"])
                del holdings[sym]
                persist_holdings(holdings)

        persist_holdings(holdings)
        render_status(tickers, holdings, state, interval, best_sym, best_prob, current_risk, best_threshold, btc_ctx)
        safe_sleep(interval)

    except ccxt.BaseError as e:
        print(Fore.RED + f"Exchange error: {e}")
        safe_sleep(60)
    except Exception as e:
        print(Fore.RED + f"Runtime error: {e}")
        safe_sleep(60)
