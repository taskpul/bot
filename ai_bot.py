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
"""

import os, time, json, warnings, pandas as pd, numpy as np, ccxt, requests, concurrent.futures
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from colorama import Fore, Style, init as color_init
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
color_init(autoreset=True)
load_dotenv()

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
risk_percent = 0.10
BASE_CAPITAL = 500.0
STATE_FILE = "adaptive_dynamic_dashboard_state.json"
MODEL_DIR = "models"
LIVE_MODE = True
MAX_OPEN_POSITIONS = 10
MAX_CAPITAL_EXPOSURE = 0.60
THREADS = 5
lock_factor = 0.5

os.makedirs(MODEL_DIR, exist_ok=True)
exchange = ccxt.binance({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})

# ───────────────────────── UTILITIES ───────────────────────
def notify(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception:
        pass

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
        return s
    return {"realized_profit": 0.0, "daily_profit": 0.0, "last_date": datetime.now().strftime("%Y-%m-%d")}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), indent=2)

state = load_state()

def effective_capital():
    return BASE_CAPITAL + state["realized_profit"]

# ───────────────────────── MODEL ───────────────────────────
class SymbolModel:
    def __init__(self, sym):
        self.sym = sym
        self.model_path = os.path.join(MODEL_DIR, f"{sym.replace('/', '_')}.json")
        self.scaler_path = os.path.join(MODEL_DIR, f"{sym.replace('/', '_')}_scaler.json")
        self.model = XGBClassifier(eval_metric="logloss", n_estimators=100, max_depth=3)
        self.scaler = StandardScaler()
        self.trained, self.last_train = False, 0
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
        except Exception:
            pass

    def _save(self):
        try:
            self.model.save_model(self.model_path)
            json.dump({"mean": self.scaler.mean_.tolist(),
                       "scale": self.scaler.scale_.tolist()}, open(self.scaler_path, "w"))
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
        return df

    def update(self, df):
        df = self.prepare_df(df)
        X = np.column_stack([
            df["ema_fast"] - df["ema_slow"],
            df["rsi"], df["atr"] / df["close"],
            df["body_ratio"], df["vol_ratio"], df["atr_change"]
        ])
        y = (df["close"].shift(-1) > df["close"]).astype(int)
        if len(X) > 40:
            try:
                Xs = self.scaler.fit_transform(X[:-1])
                self.model.fit(Xs, y[:-1])
                self.trained = True
                self.last_train = time.time()
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
                df["body_ratio"].iloc[-1], df["vol_ratio"].iloc[-1], df["atr_change"].iloc[-1]
            ]])
            r = self.scaler.transform(r)
            return float(self.model.predict_proba(r)[0, 1])
        except Exception:
            return 0.5

# ───────────────────────── DISCOVERY & DASHBOARD ───────────
def discover_symbols():
    syms = []
    try:
        tick = exchange.fetch_tickers()
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

def market_bullish():
    try:
        df = pd.DataFrame(exchange.fetch_ohlcv("BTC/USDC", timeframe, limit=slow_ema + 20),
                          columns=["ts","open","high","low","close","vol"])
        ef = df["close"].ewm(span=fast_ema).mean().iloc[-1]
        es = df["close"].ewm(span=slow_ema).mean().iloc[-1]
        return ef > es
    except Exception:
        return True

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

def print_dashboard(tickers, holdings, state, interval, next_sym, next_prob):
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

# ───────────────────────── MAIN LOOP ───────────────────────
symbols = discover_symbols()
models = {s: SymbolModel(s) for s in symbols}
holdings = {}
print(Fore.MAGENTA + "Starting Adaptive ATR Momentum Bot v3.6 (USDC only)")

# Auto-load all Binance holdings
balances = exchange.fetch_balance().get("total", {})
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
        df = pd.DataFrame(exchange.fetch_ohlcv(sym, timeframe, limit=50),
                          columns=["ts","open","high","low","close","vol"])
        df["atr"] = calc_atr(df, atr_period)
        atr = df["atr"].iloc[-1]
        holdings[sym] = {
            "amount": amt,
            "entry_price": price,
            "stop_loss": price - ATR_SL_MULT * atr,
            "take_profit": price + ATR_TP_MULT * atr
        }
        print(Fore.CYAN + f"Loaded existing {sym}: {amt:.6f} (${val:.2f})")
    except Exception:
        continue

while True:
    try:
        if not market_bullish():
            print(Fore.LIGHTBLACK_EX + "BTC filter blocking entries.")
            safe_sleep(interval)
            continue

        symbols = discover_symbols()
        for s in symbols:
            models.setdefault(s, SymbolModel(s))

        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
            data = list(ex.map(lambda s: (s, pd.DataFrame(exchange.fetch_ohlcv(s, timeframe, limit=slow_ema + 120),
                                                         columns=["ts","open","high","low","close","vol"])), symbols))
        tickers = exchange.fetch_tickers()

        best_sym, best_prob, best_price, best_atr = None, 0, 0, 0
        for sym, df in data:
            if df.empty:
                continue
            df = models[sym].prepare_df(df)
            m = models[sym]
            if time.time() - m.last_train > 3600:
                m.update(df)
            p = m.predict(df)
            price, atr = df["close"].iloc[-1], df["atr"].iloc[-1]
            if p > best_prob:
                best_sym, best_prob, best_price, best_atr = sym, p, price, atr

        if best_sym and best_prob > 0.65 and best_sym not in holdings:
            used_cap = sum(h["entry_price"] * h["amount"] for h in holdings.values())
            if len(holdings) < MAX_OPEN_POSITIONS and used_cap < effective_capital() * MAX_CAPITAL_EXPOSURE:
                alloc = risk_percent * effective_capital()
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
                save_state(state)
                c = Fore.GREEN if pnl >= 0 else Fore.RED
                print(c + f"EXIT {sym} PnL ${pnl:.2f} Total ${state['realized_profit']:.2f}")
                notify(f"EXIT {sym} PnL ${pnl:.2f}")
                if LIVE_MODE:
                    execute_sell(sym, h["amount"])
                del holdings[sym]

        print_dashboard(tickers, holdings, state, interval, best_sym, best_prob)
        safe_sleep(interval)

    except ccxt.BaseError as e:
        print(Fore.RED + f"Exchange error: {e}")
        safe_sleep(60)
    except Exception as e:
        print(Fore.RED + f"Runtime error: {e}")
        safe_sleep(60)
