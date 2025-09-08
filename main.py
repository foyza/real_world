import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# === CONFIG ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

ASSETS = ['BTC/USD', 'XAU/USD', 'NDX']
TWELVE_SYMBOLS = {
    "BTC/USD": "BTC/USD",
    "XAU/USD": "XAU/USD",
    "NDX": "NDX",
}

logging.basicConfig(level=logging.INFO)

dp = Dispatcher()
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)

user_settings = {}  # {uid: {"asset": ...}}

# === ML MODEL ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()
ml_trained = False

# === NLP SENTIMENT ===
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()


# === UI ===
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Получить сигнал")],
            [KeyboardButton(text="BTC/USD"), KeyboardButton(text="XAU/USD"), KeyboardButton(text="NDX")],
            [KeyboardButton(text="🕒 Расписание")]
        ],
        resize_keyboard=True
    )


# === DATA ===
async def get_twelvedata(asset, interval="1h", count=50):
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
            
            # Приводим к float только те колонки, что есть
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            return df



async def get_news_sentiment(asset: str):
    """Анализируем новости через NewsAPI + Vader NLP"""
    query = "bitcoin" if "BTC" in asset else "gold" if "XAU" in asset else "nasdaq"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            data = await r.json()
            if "articles" not in data:
                return 0  # нейтрально
            scores = []
            for art in data["articles"][:5]:  # последние 5 новостей
                text = art["title"] + " " + (art.get("description") or "")
                sentiment = sia.polarity_scores(text)
                scores.append(sentiment["compound"])
            if not scores:
                return 0
            avg = np.mean(scores)
            return avg  # >0 = позитив, <0 = негатив


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


def add_indicators(df):
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    return df.dropna()


# === RULE-BASED STRATEGY ===
def rule_based_signal(df):
    latest = df.iloc[-1]
    ema_signal = "buy" if latest["ema10"] > latest["ema50"] else "sell"
    rsi_signal = "buy" if latest["rsi"] < 30 else "sell" if latest["rsi"] > 70 else "neutral"
    macd_signal = "buy" if latest["macd"] > 0 else "sell"
    signals = [ema_signal, rsi_signal, macd_signal]
    direction = "buy" if signals.count("buy") >= 2 else "sell" if signals.count("sell") >= 2 else "neutral"
    accuracy = int((signals.count(direction) / 3) * 100) if direction != "neutral" else 66
    return direction, accuracy


# === ML TRAINING ===
async def train_model(asset="BTC/USD"):
    global ml_trained, model, scaler
    df = await get_twelvedata(asset, count=500)
    if df is None:
        return
    df = add_indicators(df)
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    features = df[["ema10", "ema50", "rsi", "macd"]]
    labels = df["target"].dropna()
    features = features.iloc[:len(labels)]
    X = scaler.fit_transform(features)
    y = labels
    model.fit(X, y)
    ml_trained = True
    logging.info("✅ ML модель обучена")


def ml_predict(latest_row):
    if not ml_trained:
        return "neutral", 50
    X = np.array([[latest_row["ema10"], latest_row["ema50"], latest_row["rsi"], latest_row["macd"]]])
    X = scaler.transform(X)
    prob = model.predict_proba(X)[0]
    if prob[1] > 0.55:
        return "buy", int(prob[1] * 100)
    elif prob[0] > 0.55:
        return "sell", int(prob[0] * 100)
    return "neutral", 50


# === SIGNAL ===
async def send_signal(user_id, asset):
    df = await get_twelvedata(asset, count=50)
    if df is None or len(df) < 50:
        await bot.send_message(user_id, f"⚠️ Нет данных для {asset}")
        return

    df = add_indicators(df)
    direction_rule, acc_rule = rule_based_signal(df)
    direction_ml, acc_ml = ml_predict(df.iloc[-1])
    news_score = await get_news_sentiment(asset)

    # Комбинируем 3 источника
    direction = "neutral"
    if direction_rule == direction_ml and direction_rule != "neutral":
        direction = direction_rule
        accuracy = int((acc_rule + acc_ml) / 2)
    else:
        direction, accuracy = direction_rule, acc_rule

    if news_score > 0.1:
        if direction != "sell":
            direction = "buy"
            accuracy += 10
    elif news_score < -0.1:
        if direction != "buy":
            direction = "sell"
            accuracy += 10

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
            f"📊 Точность: {accuracy}%\n"
            f"📰 Новости: {'позитив' if news_score>0 else 'негатив' if news_score<0 else 'нейтрально'}"
        )
    await bot.send_message(user_id, msg)


# === HANDLERS ===
@dp.message(CommandStart())
async def start(message: types.Message):
    user_settings[message.from_user.id] = {"asset": "BTC/USD"}
    await message.answer("Escape the matrix",
                         reply_markup=get_main_keyboard())


@dp.message()
async def handle_buttons(message: types.Message):
    uid = message.from_user.id
    text = message.text
    if uid not in user_settings:
        user_settings[uid] = {"asset": "BTC/USD"}
    if text == "🔄 Получить сигнал":
        await send_signal(uid, user_settings[uid]["asset"])
    elif text in ASSETS:
        user_settings[uid]["asset"] = text
        await message.answer(f"✅ Актив установлен: {text}")
    elif text == "🕒 Расписание":
        await message.answer("Сигналы проверяются каждые 5 минут автоматически.")


# === AUTO LOOP ===
async def auto_signal_loop():
    while True:
        for uid, settings in user_settings.items():
            await send_signal(uid, settings["asset"])
        await asyncio.sleep(300)


async def main():
    await train_model("BTC/USD")
    loop = asyncio.get_event_loop()
    loop.create_task(auto_signal_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
