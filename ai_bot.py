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
from datetime import datetime

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
interval = 120
timeframe = "1h"

fast_ema, slow_ema, rsi_period, atr_period = 9, 21, 14, 14
ATR_SL_MULT, ATR_TP_MULT, TRAIL_FACTOR = 2.0, 3.0, 0.6
BASE_RISK_PERCENT = 0.10
MIN_RISK_PERCENT = 0.05
MAX_RISK_PERCENT = 0.20
RISK_STEP = 0.02
RISK_LOOKBACK = 12
BASE_CAPITAL = 500.0
STATE_FILE = "adaptive_dynamic_dashboard_state.json"
MODEL_DIR = "models"
LIVE_MODE = env_bool("LIVE_MODE", False)
DEBUG_MODE = env_bool("DEBUG_MODE", False)
MAX_OPEN_POSITIONS = 10
MAX_CAPITAL_EXPOSURE = 0.60
THREADS = 5
lock_factor = 0.5
PROB_THRESHOLD = 0.65
HIGH_VOLATILITY_LEVEL = 0.025
LOW_VOLATILITY_LEVEL = 0.012
TREND_STRENGTH_BONUS = 0.04
FEE_BUFFER = 0.0015
PERFORMANCE_SUFFIX = "_meta.json"

os.makedirs(MODEL_DIR, exist_ok=True)
exchange = ccxt.binance({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})

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
        return s
    return {
        "realized_profit": 0.0,
        "daily_profit": 0.0,
        "last_date": datetime.now().strftime("%Y-%m-%d"),
        "trade_history": [],
        "current_risk_percent": BASE_RISK_PERCENT
    }

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), indent=2)

state = load_state()

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
        self.performance = {
            "precision": 0.5,
            "recall": 0.5,
            "avg_forward_return": 0.0,
            "threshold_bonus": 0.0
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
        except Exception:
            pass

    def _save(self):
        try:
            self.model.save_model(self.model_path)
            json.dump({"mean": self.scaler.mean_.tolist(),
                       "scale": self.scaler.scale_.tolist()}, open(self.scaler_path, "w"))
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
        X = np.column_stack([
            df["ema_fast"] - df["ema_slow"],
            df["rsi"], df["atr"] / df["close"],
            df["body_ratio"], df["vol_ratio"], df["atr_change"],
            df["ema_ratio"], df["price_momentum"], df["volume_zscore"],
            df["volatility_regime"], df["trend_slope"], df["momentum_3"],
            df["rsi_slope"], df["volume_trend"], df["bollinger_pos"]
        ])
        forward_returns = (df["close"].shift(-1) / df["close"]) - 1
        y = (forward_returns > FEE_BUFFER).astype(int)
        if len(X) > 40:
            try:
                Xs = self.scaler.fit_transform(X[:-1])
                y_train_full = y[:-1]
                val_size = max(10, int(len(Xs) * 0.2))
                if len(Xs) - val_size <= 20:
                    val_size = 0
                if val_size:
                    X_train, X_val = Xs[:-val_size], Xs[-val_size:]
                    y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]
                else:
                    X_train, y_train = Xs, y_train_full
                    X_val, y_val = np.empty((0, Xs.shape[1])), np.array([])

                self.model.fit(X_train, y_train)
                self.trained = True
                self.last_train = time.time()

                if val_size and len(np.unique(y_val)) > 1:
                    val_probs = self.model.predict_proba(X_val)[:, 1]
                    val_preds = (val_probs >= 0.5).astype(int)
                    precision = precision_score(y_val, val_preds, zero_division=0)
                    recall = recall_score(y_val, val_preds, zero_division=0)
                    avg_forward = float(np.nanmean(forward_returns[-val_size:]))
                    bonus = 0.0
                    if precision > 0.58:
                        bonus -= 0.04
                    elif precision < 0.45:
                        bonus += 0.04
                    if avg_forward > FEE_BUFFER * 2:
                        bonus -= 0.02
                    elif avg_forward < FEE_BUFFER:
                        bonus += 0.02
                    self.performance.update({
                        "precision": round(float(precision), 4),
                        "recall": round(float(recall), 4),
                        "avg_forward_return": round(avg_forward, 6),
                        "threshold_bonus": float(np.clip(bonus, -0.06, 0.06))
                    })
                else:
                    self.performance.update({
                        "precision": 0.5,
                        "recall": 0.5,
                        "avg_forward_return": float(np.nanmean(forward_returns)) if len(forward_returns) else 0.0,
                        "threshold_bonus": 0.0
                    })
                self._save()
            except Exception:
                pass

    def predict(self, df):
        if not self.trained:
            return 0.5
        try:
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
    return min(max(threshold, 0.5), 0.85)

def execute_buy(sym, amt):
    try:
        o = exchange.create_market_buy_order(sym, amt)
        print(Fore.GREEN + f"BUY {sym} {amt:.4f}")
        notify(f"BUY {sym} {amt:.4f}")
        return o
    except Exception as e:
        print(Fore.RED + f"BUY error {sym}: {e}")
        return None

def execute_sell(sym, amt):
    try:
        o = exchange.create_market_sell_order(sym, amt)
        print(Fore.GREEN + f"SELL {sym} {amt:.4f}")
        notify(f"SELL {sym} {amt:.4f}")
        return o
    except Exception as e:
        print(Fore.RED + f"SELL error {sym}: {e}")
        return None

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def load_existing_holdings():
    holdings = {}
    balances = safe_fetch_balance_total()
    if not balances:
        return holdings
    for asset, amt in balances.items():
        if asset == QUOTE or amt <= 0:
            continue
        sym = f"{asset}/{QUOTE}"
        try:
            t = exchange.fetch_ticker(sym)
            price = t["last"]
            val = amt * price
            if val < MIN_HOLD_VALUE:
                continue
            df = fetch_symbol_ohlcv(sym, 50)
            if df.empty:
                continue
            df["atr"] = calc_atr(df, atr_period)
            atr = df["atr"].iloc[-1]
            holdings[sym] = {
                "amount": amt,
                "entry_price": price,
                "stop_loss": price - ATR_SL_MULT * atr,
                "take_profit": price + ATR_TP_MULT * atr
            }
            print(Fore.CYAN + f"Loaded existing {sym}: {amt:.6f} (${val:.2f})")
        except Exception as e:
            print(Fore.LIGHTBLACK_EX + f"Skip holding sync for {sym}: {e}")
            continue
    return holdings

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
    print(Fore.WHITE + f"Entry Threshold: {threshold:.2f}  BTC vol {btc_ctx.get('volatility',0):.3f} trend {btc_ctx.get('trend_strength',0):+.3f}")
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
    summary = (
        f"[{now}] Next run in {interval}s | Holdings: {len(holdings)} | Free cap: ${free_cap:,.2f} | "
        f"Risk: {current_risk*100:.1f}% (win {stats['win_rate']*100:.0f}% avgPnL {stats['avg_pnl']:+.2f}) | "
        f"Threshold: {threshold:.2f} | BTC vol {btc_ctx.get('volatility',0):.3f} trend {btc_ctx.get('trend_strength',0):+.3f}"
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
        btc_ctx = fetch_btc_context()
        if not btc_ctx["bullish"]:
            print(Fore.LIGHTBLACK_EX + "BTC filter blocking entries.")
            safe_sleep(interval)
            continue

        symbols = discover_symbols()
        for s in symbols:
            models.setdefault(s, SymbolModel(s))

        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
            data = list(ex.map(lambda s: (s, fetch_symbol_ohlcv(s, slow_ema + 120)), symbols))
        tickers = safe_fetch_tickers()

        entry_threshold = dynamic_entry_threshold(btc_ctx)
        current_risk = state.get("current_risk_percent", BASE_RISK_PERCENT)

        best_sym, best_prob, best_price, best_atr = None, 0, 0, 0
        best_threshold, best_risk_mod = entry_threshold, 1.0
        for sym, df in data:
            if df.empty:
                continue
            df = models[sym].prepare_df(df)
            m = models[sym]
            if time.time() - m.last_train > 3600:
                m.update(df)
            p = m.predict(df)
            price, atr = df["close"].iloc[-1], df["atr"].iloc[-1]
            sym_threshold = m.adjusted_threshold(entry_threshold)
            risk_mod = m.risk_modifier()
            if p > best_prob:
                best_sym, best_prob, best_price, best_atr = sym, p, price, atr
                best_threshold, best_risk_mod = sym_threshold, risk_mod

        if best_sym and best_prob > best_threshold and best_sym not in holdings:
            used_cap = sum(h["entry_price"] * h["amount"] for h in holdings.values())
            if len(holdings) < MAX_OPEN_POSITIONS and used_cap < effective_capital() * MAX_CAPITAL_EXPOSURE:
                vol_mod = volatility_risk_modifier(btc_ctx)
                alloc = current_risk * best_risk_mod * vol_mod * effective_capital()
                max_alloc = max(effective_capital() * MAX_CAPITAL_EXPOSURE - used_cap, 0)
                alloc = min(alloc, max_alloc)
                if alloc > 0:
                    amt = alloc / best_price
                    if LIVE_MODE:
                        execute_buy(best_sym, amt)
                    holdings[best_sym] = {
                        "amount": amt,
                        "entry_price": best_price,
                        "stop_loss": best_price - ATR_SL_MULT * best_atr,
                        "take_profit": best_price + ATR_TP_MULT * best_atr
                    }

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
                c = Fore.GREEN if pnl >= 0 else Fore.RED
                print(c + f"EXIT {sym} PnL ${pnl:.2f} Total ${state['realized_profit']:.2f}")
                notify(f"EXIT {sym} PnL ${pnl:.2f}")
                if LIVE_MODE:
                    execute_sell(sym, h["amount"])
                del holdings[sym]

        render_status(tickers, holdings, state, interval, best_sym, best_prob, current_risk, best_threshold, btc_ctx)
        safe_sleep(interval)

    except ccxt.BaseError as e:
        print(Fore.RED + f"Exchange error: {e}")
        safe_sleep(60)
    except Exception as e:
        print(Fore.RED + f"Runtime error: {e}")
        safe_sleep(60)
