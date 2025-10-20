"""Telegram control dashboard for Adaptive ATR Momentum Bot."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.constants import ParseMode
    from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                              ContextTypes)
    from telegram.helpers import escape_markdown
except Exception:  # pragma: no cover - optional dependency guard
    InlineKeyboardButton = InlineKeyboardMarkup = Update = ParseMode = Application = None  # type: ignore
    CallbackQueryHandler = CommandHandler = ContextTypes = None  # type: ignore
    escape_markdown = None  # type: ignore

try:  # pragma: no cover - bot module may be absent during tests
    import adaptive_bot

    state = adaptive_bot.state
    state_lock = adaptive_bot.state_lock
    momentum_holdings = adaptive_bot.momentum_holdings
    momentum_holdings_lock = adaptive_bot.momentum_holdings_lock
    dip_holdings = adaptive_bot.dip_holdings
    dip_holdings_lock = adaptive_bot.dip_holdings_lock
    effective_capital = adaptive_bot.effective_capital
    momentum_used_capital = adaptive_bot.momentum_used_capital
    dip_used_capital = adaptive_bot.dip_used_capital
    pause_engines = adaptive_bot.pause_engines
    resume_engines = adaptive_bot.resume_engines
    is_running = adaptive_bot.is_running
    pause_status = adaptive_bot.pause_status
    request_restart = adaptive_bot.request_restart
    safe_fetch_tickers = adaptive_bot.safe_fetch_tickers
except Exception:
    adaptive_bot = None  # type: ignore
    state = {}
    state_lock = threading.RLock()
    momentum_holdings = {}
    momentum_holdings_lock = threading.RLock()
    dip_holdings = {}
    dip_holdings_lock = threading.RLock()

    def effective_capital() -> float:
        return 0.0

    def momentum_used_capital(_: Dict[str, Any]) -> float:
        return 0.0

    def dip_used_capital(_: Dict[str, Any]) -> float:
        return 0.0

    def pause_engines(reason: Optional[str] = None) -> None:  # type: ignore
        logging.warning("pause_engines called but adaptive_bot unavailable: %s", reason)

    def resume_engines() -> None:  # type: ignore
        logging.warning("resume_engines called but adaptive_bot unavailable")

    def is_running() -> bool:  # type: ignore
        return False

    def pause_status() -> Optional[str]:  # type: ignore
        return "Bot offline"

    def request_restart(delay: float = 2.0) -> None:  # type: ignore
        logging.warning("Restart requested but adaptive_bot unavailable (delay=%s)", delay)

    def safe_fetch_tickers() -> Dict[str, Any]:  # type: ignore
        return {}


TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
AUTHORIZED_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PTB_AVAILABLE = InlineKeyboardButton is not None and Application is not None and escape_markdown is not None

_logger = logging.getLogger(__name__)
_application_lock = threading.Lock()
_application_thread: Optional[threading.Thread] = None
_application_ready = threading.Event()


def _build_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh"),
            InlineKeyboardButton("📈 Performance", callback_data="performance"),
            InlineKeyboardButton("💼 Positions", callback_data="positions"),
        ],
        [
            InlineKeyboardButton("▶️ Start", callback_data="start"),
            InlineKeyboardButton("⏸ Stop", callback_data="stop"),
            InlineKeyboardButton("♻️ Restart", callback_data="restart"),
        ],
    ])


def _authorized(update: Update) -> bool:
    chat = update.effective_chat
    if chat is None:
        return False
    if not AUTHORIZED_CHAT_ID:
        return False
    return str(chat.id) == AUTHORIZED_CHAT_ID


def _escape(text: str) -> str:
    if not PTB_AVAILABLE:
        return text
    return escape_markdown(text, version=2)


def _format_usd(value: float) -> str:
    return f"${value:,.2f}"


def _format_percent(value: float) -> str:
    return f"{value:+.2f}%"


def _current_price(symbol: str, tickers: Dict[str, Any]) -> Optional[float]:
    ticker = tickers.get(symbol) if tickers else None
    if not ticker:
        return None
    for key in ("last", "close", "bid", "ask"):
        price = ticker.get(key)
        if isinstance(price, (int, float)) and price > 0:
            return float(price)
    return None


def _snapshot_state() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    with state_lock:
        state_snapshot = deepcopy(state)
    with momentum_holdings_lock:
        momentum_snapshot = deepcopy(momentum_holdings)
    with dip_holdings_lock:
        dip_snapshot = deepcopy(dip_holdings)
    return state_snapshot, momentum_snapshot, dip_snapshot


def _performance_metrics(trade_history: Iterable[float]) -> Tuple[float, float, float]:
    history = [float(x) for x in trade_history if isinstance(x, (int, float))]
    if not history:
        return 0.0, 0.0, 0.0
    wins = sum(1 for x in history if x > 0)
    win_rate = wins / len(history)
    avg_pnl = mean(history)
    sharpe = 0.0
    if len(history) >= 2:
        dispersion = pstdev(history)
        if dispersion > 0:
            sharpe = (avg_pnl / dispersion) * (len(history) ** 0.5)
    return win_rate, avg_pnl, sharpe


def _format_positions(momentum_snapshot: Dict[str, Any], dip_snapshot: Dict[str, Any], tickers: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for sym, info in sorted(momentum_snapshot.items()):
        try:
            entry = float(info.get("entry_price", 0.0) or 0.0)
            stop_loss = float(info.get("stop_loss", 0.0) or 0.0)
            trail_high = float(info.get("trail_high", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        price = _current_price(sym, tickers)
        change = 0.0
        if price and entry:
            change = ((price - entry) / entry) * 100
        lines.append(
            f"{sym:<12} { _format_percent(change):>8} @ {price or entry:.4f} | SL {stop_loss:.4f} | TH {trail_high:.4f}"
        )
    for sym, info in sorted(dip_snapshot.items()):
        layers = info.get("layers", {}) if isinstance(info, dict) else {}
        try:
            total_amount = float(info.get("total_amount", info.get("amount", 0.0)) or 0.0)
            avg_entry = float(info.get("avg_entry", info.get("entry_price", 0.0)) or 0.0)
            trail_high = float(info.get("trail_high", avg_entry) or avg_entry)
            trail_stop = float(info.get("trail_stop", avg_entry * 0.98) or avg_entry * 0.98)
        except (TypeError, ValueError):
            continue
        price = _current_price(sym, tickers)
        change = 0.0
        if price and avg_entry:
            change = ((price - avg_entry) / avg_entry) * 100
        layer_desc = ",".join(sorted(layers.keys())) if layers else "-"
        lines.append(
            f"{sym:<12} { _format_percent(change):>8} @ {price or avg_entry:.4f} | TL {trail_stop:.4f} | Layers {layer_desc}"
        )
    if not lines:
        lines.append("No open positions")
    return lines


def _build_report(view: str = "report") -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    state_snapshot, momentum_snapshot, dip_snapshot = _snapshot_state()
    total_cap = effective_capital()
    momentum_cap = momentum_used_capital(momentum_snapshot)
    dip_cap = dip_used_capital(dip_snapshot)
    used_total = momentum_cap + dip_cap
    daily_pnl = float(state_snapshot.get("daily_profit", 0.0) or 0.0)
    trade_history = state_snapshot.get("trade_history", [])
    win_rate, avg_pnl, sharpe = _performance_metrics(trade_history)

    tickers: Dict[str, Any] = {}
    if view in {"report", "positions"}:
        tickers = safe_fetch_tickers() or {}

    header = "📊 *Adaptive ATR Momentum Dashboard*"
    divider = "──────────────────────────────"
    paused_banner = ""
    pause_msg = pause_status()
    if pause_msg and not is_running():
        paused_banner = _escape(f"⚠️ {pause_msg}")

    blocks: List[str] = []

    if view == "performance":
        perf_lines = [
            "Performance Metrics",
            f"Rolling PnL : {_format_usd(float(state_snapshot.get('realized_profit', 0.0) or 0.0))}",
            f"Daily PnL   : {_format_usd(daily_pnl)}",
            f"Win Rate    : {win_rate * 100:.1f}%",
            f"Avg PnL     : {avg_pnl:+.4f}",
            f"Sharpe      : {sharpe:.2f}",
            f"Trades      : {len(trade_history)}",
        ]
        block = "```\n" + "\n".join(perf_lines) + "\n```"
        blocks.append(block)
    elif view == "positions":
        position_lines = ["Open Positions"] + _format_positions(momentum_snapshot, dip_snapshot, tickers)
        block = "```\n" + "\n".join(position_lines) + "\n```"
        blocks.append(block)
    else:
        lines = [
            f"💰 Total Capital : {_format_usd(total_cap)}",
            f"⚙️ Used Capital  : {_format_usd(used_total)}",
            f"    Momentum     : {_format_usd(momentum_cap)}",
            f"    Dip-Buy      : {_format_usd(dip_cap)}",
            f"📈 Daily PnL    : {_format_usd(daily_pnl)}",
            f"📊 Win Rate     : {win_rate * 100:.1f}% | AvgPnL {avg_pnl:+.4f} | Sharpe {sharpe:.2f}",
        ]
        block = "```\n" + "\n".join(lines) + "\n```"
        blocks.append(block)
        position_lines = ["Open Positions"] + _format_positions(momentum_snapshot, dip_snapshot, tickers)
        blocks.append("```\n" + "\n".join(position_lines) + "\n```")

    footer = _escape(f"⏱ Updated: {now}")

    message_parts = [header, divider]
    if paused_banner:
        message_parts.append(paused_banner)
    message_parts.extend(blocks)
    message_parts.append(divider)
    message_parts.append(footer)
    return "\n".join(message_parts)


async def _send_report(update: Update, context: ContextTypes.DEFAULT_TYPE, view: str) -> None:
    if not _authorized(update):
        return
    context.chat_data["dashboard_view"] = view
    text = await asyncio.to_thread(_build_report, view)
    reply_markup = _build_keyboard()
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        try:
            await query.edit_message_text(text=text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup)
        except Exception as exc:
            _logger.debug("Edit message failed: %s", exc)
    elif update.effective_message:
        await update.effective_message.reply_text(text=text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    resume_engines()
    if update.callback_query:
        await update.callback_query.answer()
    text = _escape("▶️ Engines resumed")
    await update.effective_message.reply_text(text=text, parse_mode=ParseMode.MARKDOWN_V2)


async def handle_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    pause_engines("Paused via Telegram")
    if update.callback_query:
        await update.callback_query.answer()
    text = _escape("⏸ Engines paused")
    await update.effective_message.reply_text(text=text, parse_mode=ParseMode.MARKDOWN_V2)


async def handle_restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    request_restart()
    if update.callback_query:
        await update.callback_query.answer()
    text = _escape("♻️ Restart requested")
    await update.effective_message.reply_text(text=text, parse_mode=ParseMode.MARKDOWN_V2)


async def handle_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_report(update, context, "report")


async def handle_positions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_report(update, context, "positions")


async def handle_performance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_report(update, context, "performance")


async def handle_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        if update.callback_query:
            await update.callback_query.answer()
        return
    view = context.chat_data.get("dashboard_view", "report")
    await _send_report(update, context, view)


async def handle_root_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    resume_engines()
    await _send_report(update, context, "report")


def _register_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", handle_root_start))
    application.add_handler(CommandHandler("report", handle_report))
    application.add_handler(CommandHandler("positions", handle_positions))
    application.add_handler(CommandHandler("performance", handle_performance))
    application.add_handler(CommandHandler("stop", handle_stop))
    application.add_handler(CommandHandler("restart", handle_restart))
    application.add_handler(CommandHandler("resume", handle_start))
    application.add_handler(CallbackQueryHandler(handle_refresh, pattern="^refresh$"))
    application.add_handler(CallbackQueryHandler(handle_performance, pattern="^performance$"))
    application.add_handler(CallbackQueryHandler(handle_positions, pattern="^positions$"))
    application.add_handler(CallbackQueryHandler(handle_start, pattern="^start$"))
    application.add_handler(CallbackQueryHandler(handle_stop, pattern="^stop$"))
    application.add_handler(CallbackQueryHandler(handle_restart, pattern="^restart$"))


async def _run_application() -> None:
    if not PTB_AVAILABLE or not TOKEN or not AUTHORIZED_CHAT_ID:
        _logger.warning("Telegram dashboard disabled (PTB=%s, token=%s, chat_id=%s)", PTB_AVAILABLE, bool(TOKEN), bool(AUTHORIZED_CHAT_ID))
        return
    application = Application.builder().token(TOKEN).build()
    _register_handlers(application)
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    _application_ready.set()
    try:
        await application.updater.idle()
    finally:
        await application.shutdown()
        await application.stop()


def _thread_target() -> None:
    try:
        asyncio.run(_run_application())
    except Exception as exc:  # pragma: no cover - background thread guard
        _logger.error("Telegram dashboard crashed: %s", exc)


def ensure_started() -> None:
    if not PTB_AVAILABLE or not TOKEN or not AUTHORIZED_CHAT_ID:
        return
    global _application_thread
    with _application_lock:
        if _application_thread and _application_thread.is_alive():
            return
        logging.getLogger("telegram").setLevel(logging.INFO)
        _application_thread = threading.Thread(target=_thread_target, name="telegram-dashboard", daemon=True)
        _application_thread.start()


__all__ = [
    "ensure_started",
]
