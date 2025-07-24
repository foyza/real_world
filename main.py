import asyncio
import os
import logging
import sqlite3
from datetime import datetime
from telegram.ext import Updater, CommandHandler

import pandas as pd
import httpx
import ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters
)

from dotenv import load_dotenv
load_dotenv()

# ⛓ Настройки
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TWELVE_API_KEY = os.getenv("TWELVEDATA_API_KEY")
DB_FILE = "users.sqlite"

# 📊 Индикаторы по умолчанию
DEFAULT_STRATEGY = "All"
DEFAULT_TIMEFRAME = "15min"
DEFAULT_SYMBOL = "BTC/USD"
ML_ENABLED_USERS = set()

# 🧠 Настройки базы
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY, strategy TEXT, timeframe TEXT, symbol TEXT)''')
conn.commit()

# 🔧 Хелперы

def save_user_settings(user_id, strategy, timeframe, symbol):
    c.execute("INSERT OR REPLACE INTO users (user_id, strategy, timeframe, symbol) VALUES (?, ?, ?, ?)",
              (user_id, strategy, timeframe, symbol))
    conn.commit()

def get_user_settings(user_id):
    c.execute("SELECT strategy, timeframe, symbol FROM users WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    return row if row else (DEFAULT_STRATEGY, DEFAULT_TIMEFRAME, DEFAULT_SYMBOL)

# 📥 Получение данных
async def get_data_twelvedata(symbol: str, interval: str = "15min", outputsize: int = 100):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol.replace('/', '')}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_API_KEY}&format=JSON"
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})
    df = df.sort_values("time")
    return df

# 📈 Индикаторы и фильтры
def apply_indicators(df):
    df["ema21"] = ta.trend.ema_indicator(df["Close"], 21)
    df["ema50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["ema200"] = ta.trend.ema_indicator(df["Close"], 200)
    df["rsi"] = ta.momentum.rsi(df["Close"])
    df["macd"] = ta.trend.macd_diff(df["Close"])
    df["adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"])
    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
    return df

def filter_conditions(df):
    last = df.iloc[-1]
    conditions = {
        "trend": last["adx"] > 20 and last["Close"] > last["ema50"],
        "overbought": last["rsi"] > 70,
        "oversold": last["rsi"] < 30,
        "price_action": any([
            abs(last["Open"] - last["Close"]) < (last["High"] - last["Low"]) * 0.2,  # Doji
            (last["Close"] < last["Open"]) and (last["High"] - last["Close"] < (last["High"] - last["Low"]) * 0.3),  # PinBar
        ])
    }
    return conditions

def generate_signal(df):
    df = apply_indicators(df)
    conditions = filter_conditions(df)
    if conditions["trend"] and conditions["oversold"]:
        return "📈 LONG"
    elif conditions["trend"] and conditions["overbought"]:
        return "📉 SHORT"
    else:
        return "⏳ WAIT"

# 🧠 ML модуль
def ml_enhance(signal: str, df: pd.DataFrame):
    # Простая заглушка ML-модуля
    return f"{signal} 🤖" if signal in ["📈 LONG", "📉 SHORT"] else signal

# 🎯 TP/SL расчет
def adaptive_tp_sl(df):
    atr = df["atr"].iloc[-1]
    entry = df["Close"].iloc[-1]
    tp = round(entry + atr * 1.5, 2)
    sl = round(entry - atr * 1.0, 2)
    return tp, sl

# 🔄 Telegram команды

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    save_user_settings(user.id, DEFAULT_STRATEGY, DEFAULT_TIMEFRAME, DEFAULT_SYMBOL)
    await update.message.reply_text("🎬 Добро пожаловать. Пора выбраться из матрицы.\nНажми “🔄 Получить сигнал”")

async def signal_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    strategy, timeframe, symbol = get_user_settings(user_id)
    df = await get_data_twelvedata(symbol, timeframe)
    if df is None:
        await update.message.reply_text("⚠️ Ошибка получения данных")
        return
    signal = generate_signal(df)
    if user_id in ML_ENABLED_USERS:
        signal = ml_enhance(signal, df)
    tp, sl = adaptive_tp_sl(df)
    await update.message.reply_text(f"🔔 Сигнал: {signal}\n🎯 TP: {tp}\n🛡 SL: {sl}")

async def strategy_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if args:
        strategy = args[0]
        _, tf, sym = get_user_settings(user_id)
        save_user_settings(user_id, strategy, tf, sym)
        await update.message.reply_text(f"✅ Стратегия обновлена: {strategy}")
    else:
        await update.message.reply_text("⚙️ Используй: /strategy [название]")

async def ml_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    ML_ENABLED_USERS.add(user_id)
    await update.message.reply_text("🤖 ML-модуль включён")

async def ml_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    ML_ENABLED_USERS.discard(user_id)
    await update.message.reply_text("🧠 ML-модуль отключён")

async def set_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Позже: интерактивные настройки через кнопки
    await update.message.reply_text("⚙️ Настройки пока в разработке")

# ▶️ Запуск бота
def main():
    logging.basicConfig(level=logging.INFO)

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("strategy", strategy_handler))
    dp.add_handler(CommandHandler("ml_on", ml_on))
    dp.add_handler(CommandHandler("ml_off", ml_off))
    dp.add_handler(CommandHandler("set", set_handler))
    dp.add_handler(CommandHandler("signal", signal_handler))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
