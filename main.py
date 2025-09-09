import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from dotenv import load_dotenv
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# === CONFIG ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

ASSETS = ['BTC/USD', 'XAU/USD', 'ETH/USD']

logging.basicConfig(level=logging.INFO)

dp = Dispatcher()
bot = Bot(token=TOKEN, default=types.DefaultBotProperties(parse_mode=ParseMode.HTML))

user_settings = {}  # {uid: {"asset": ..., "mute": False}}

# === ML MODEL ===
model = None
ml_trained = False

# === NLP SENTIMENT ===
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# === UI ===
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Получить сигнал")],
            [KeyboardButton(text="BTC/USD"), KeyboardButton(text="XAU/USD"), KeyboardButton(text="ETH/USD")],
            [KeyboardButton(text="🕒 Расписание")],
            [KeyboardButton(text="🔇 Mute"), KeyboardButton(text="🔊 Unmute")]
        ],
        resize_keyboard=True
    )

# === DATA ===
async def get_twelvedata(asset, interval="1h", count=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": asset,
        "interval": interval,
        "outputsize": count,
        "apikey": TWELVEDATA_API_KEY,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            if "values" not in data:
                raise ValueError(f"TwelveData API error: {data.get('message', 'no data')}")
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")

            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df.dropna()

async def get_news_sentiment(asset: str):
    query = "bitcoin" if "BTC" in asset else "gold" if "XAU" in asset else "ethereum"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            data = await r.json()
            if "articles" not in data:
                return 0
            scores = []
            for art in data["articles"][:5]:
                text = art["title"] + " " + (art.get("description") or "")
                sentiment = sia.polarity_scores(text)
                scores.append(sentiment["compound"])
            return np.mean(scores) if scores else 0

# === INDICATORS ===
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def compute_obv(df):
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    return obv

def add_indicators(df):
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df)
    df["obv"] = compute_obv(df)
    df["return"] = df["close"].pct_change()
    df["bollinger_up"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["bollinger_down"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
    return df.dropna()

# === RULE-BASED STRATEGY ===
def rule_based_signal(df):
    latest = df.iloc[-1]
    ema_signal = "buy" if latest["ema10"] > latest["ema50"] else "sell"
    rsi_signal = "buy" if latest["rsi"] < 30 else "sell" if latest["rsi"] > 70 else "neutral"
    macd_signal = "buy" if latest["macd"] > 0 else "sell"
    signals = [ema_signal, rsi_signal, macd_signal]
    direction = "buy" if signals.count("buy") >= 2 else "sell" if signals.count("sell") >= 2 else "neutral"
    acc = round((signals.count(direction) / 3) * 100, 2) if direction != "neutral" else 50.0
    return direction, acc

# === ML TRAINING ===
async def train_model(asset="BTC/USD"):
    global ml_trained, model
    df = await get_twelvedata(asset, count=500)
    if df is None:
        return
    df = add_indicators(df)

    df["target"] = np.where(df["close"].shift(-3) > df["close"] * 1.002, 1,
                            np.where(df["close"].shift(-3) < df["close"] * 0.998, 0, np.nan))
    df = df.dropna()

    features = ["ema10", "ema50", "rsi", "macd", "atr", "obv", "return", "bollinger_up", "bollinger_down"]
    X = df[features]
    y = df["target"]

    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_score = 0

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

        params = {"objective": "binary", "metric": "binary_error", "verbosity": -1}
        tmp_model = lgb.train(params, lgb_train, valid_sets=[lgb_test], num_boost_round=200,
                              early_stopping_rounds=20, verbose_eval=False)

        acc = 1 - tmp_model.best_score["valid_0"]["binary_error"]
        if acc > best_score:
            best_score = acc
            best_model = tmp_model

    model = best_model
    ml_trained = True
    logging.info(f"✅ ML модель обучена, точность {best_score:.2%}")

def ml_predict(latest_row):
    if not ml_trained or model is None:
        return "neutral", 50.0
    X = latest_row[["ema10", "ema50", "rsi", "macd", "atr", "obv", "return", "bollinger_up", "bollinger_down"]].values.reshape(1, -1)
    prob = model.predict(X)[0]
    if prob > 0.7:
        return "buy", round(prob * 100, 2)
    elif prob < 0.3:
        return "sell", round((1 - prob) * 100, 2)
    return "neutral", 50.0

# === SIGNAL ===
async def send_signal(user_id, asset):
    if user_settings.get(user_id, {}).get("mute", False):
        return

    df = await get_twelvedata(asset, count=200)
    if df is None or len(df) < 50:
        await bot.send_message(user_id, f"⚠️ Нет данных для {asset}")
        return

    df = add_indicators(df)
    direction_rule, acc_rule = rule_based_signal(df)
    direction_ml, acc_ml = ml_predict(df.iloc[-1])
    news_score = await get_news_sentiment(asset)

    direction = "neutral"
    accuracy = 50.0

    if direction_rule == direction_ml and direction_rule != "neutral":
        direction = direction_rule
        accuracy = round((acc_rule + acc_ml) / 2, 2)
    else:
        direction, accuracy = direction_rule, acc_rule

    if news_score > 0.1 and direction != "sell":
        direction = "buy"
        accuracy += 5
    elif news_score < -0.1 and direction != "buy":
        direction = "sell"
        accuracy += 5

    accuracy = min(100, accuracy)
    price = df["close"].iloc[-1]
    tp_pct, sl_pct = 2.0, 1.0
    tp_price = round(price * (1 + tp_pct / 100), 2) if direction == "buy" else round(price * (1 - tp_pct / 100), 2)
    sl_price = round(price * (1 - sl_pct / 100), 2) if direction == "buy" else round(price * (1 + sl_pct / 100), 2)

    if direction == "neutral":
        msg = f"⚠️ Недостаточно сигнала по {asset}"
    else:
        msg = (
            f"📢 Сигнал для <b>{asset}</b>\n"
            f"Направление: <b>{direction.upper()}</b>\n"
            f"Цена: {price}\n"
            f"🟢 TP: {tp_price} (+{tp_pct}%)\n"
            f"🔴 SL: {sl_price} (-{sl_pct}%)\n"
            f"📊 Точность: {accuracy:.2f}%\n"
            f"📰 Новости: {'позитив' if news_score>0 else 'негатив' if news_score<0 else 'нейтрально'}"
        )
    await bot.send_message(user_id, msg)

# === HANDLERS ===
@dp.message(CommandStart())
async def start(message: types.Message):
    user_settings[message.from_user.id] = {"asset": "BTC/USD", "mute": False}
    await message.answer("🚀 Trading bot готов к работе!", reply_markup=get_main_keyboard())

@dp.message()
async def handle_buttons(message: types.Message):
    uid = message.from_user.id
    text = message.text
    if uid not in user_settings:
        user_settings[uid] = {"asset": "BTC/USD", "mute": False}

    if text == "🔄 Получить сигнал":
        await send_signal(uid, user_settings[uid]["asset"])
    elif text in ASSETS:
        user_settings[uid]["asset"] = text
        await message.answer(f"✅ Актив установлен: {text}")
    elif text == "🕒 Расписание":
        await message.answer("Сигналы проверяются каждые 15 минут автоматически.")
    elif text == "🔇 Mute":
        user_settings[uid]["mute"] = True
        await message.answer("🔇 Авто-сигналы отключены")
    elif text == "🔊 Unmute":
        user_settings[uid]["mute"] = False
        await message.answer("🔊 Авто-сигналы включены")

# === AUTO LOOP ===
async def auto_signal_loop():
    while True:
        for uid, settings in user_settings.items():
            if not settings.get("mute", False):
                await send_signal(uid, settings["asset"])
        await asyncio.sleep(900)  # каждые 15 минут

async def main():
    await train_model("BTC/USD")
    loop = asyncio.get_event_loop()
    loop.create_task(auto_signal_loop())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
