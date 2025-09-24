import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# === CONFIG ===
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ASSETS = ['BTC/USD', 'XAU/USD', 'ETH/USD']

logging.basicConfig(level=logging.INFO)
dp = Dispatcher()
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)
user_settings = {}  # {uid: {"asset": ... , "muted": False}}

# === ML + LSTM ===
model_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
scaler = StandardScaler()
model_lstm = None
ml_trained = False

# === NLP ===
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# === UI ===
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Получить сигнал")],
            [KeyboardButton(text="BTC/USD"), KeyboardButton(text="XAU/USD"), KeyboardButton(text="ETH/USD")],
            [KeyboardButton(text="🔕 Mute"), KeyboardButton(text="🔔 Unmute")]
        ], 
        resize_keyboard=True
    )


# === DATA ===
async def get_twelvedata(asset, interval="1h", count=150):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": asset, "interval": interval, "outputsize": count, "apikey": TWELVEDATA_API_KEY}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
            if "values" not in data:
                return None
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                else:
                    df[col] = 0  # Заглушка
            return df

async def get_news_sentiment(asset):
    # фильтр только сильных новостей
    query = "bitcoin" if "BTC" in asset else "gold" if "XAU" in asset else "ethereum"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            data = await r.json()
            if "articles" not in data:
                return 0
            scores = []
            for art in data["articles"][:5]:
                title = art.get("title","").lower()
                description = art.get("description","").lower()
                if any(word in title+description for word in ["fed","cpi","interest","regulation","etf"]):
                    text = art.get("title","") + " " + art.get("description","")
                    scores.append(sia.polarity_scores(text)["compound"])
            return float(np.mean(scores)) if scores else 0

# === INDICATORS ===
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def compute_atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(close, volume):
    obv = [0]
    for i in range(1,len(close)):
        obv.append(obv[-1]+volume.iloc[i] if close.iloc[i]>close.iloc[i-1] else obv[-1]-volume.iloc[i] if close.iloc[i]<close.iloc[i-1] else obv[-1])
    return pd.Series(obv, index=close.index)

def compute_bollinger(series, period=20, dev=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + dev*std, ma - dev*std

def add_indicators(df):
    df = df.copy()
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df["high"], df["low"], df["close"])
    df["bb_upper"], df["bb_lower"] = compute_bollinger(df["close"])
    if "volume" in df.columns:
        df["obv"] = compute_obv(df["close"], df["volume"])
    else:
        df["obv"] = 0
    return df.dropna()

# === ML TRAINING ===
async def train_models(asset="BTC/USD"):
    global ml_trained, model_gb, scaler, model_lstm
    df = await get_twelvedata(asset, count=500)
    if df is None: return
    df = add_indicators(df)
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    features = df[["ema10","ema50","rsi","macd","atr","obv"]].iloc[:-3]
    labels = df["target"].iloc[:-3]
    X = scaler.fit_transform(features)
    y = labels
    model_gb.fit(X, y)
    
    # LSTM
    X_lstm = np.expand_dims(X, axis=1)
    model_lstm = Sequential([
        LSTM(32,input_shape=(X_lstm.shape[1],X_lstm.shape[2])),
        Dense(1,activation="sigmoid")
    ])
    model_lstm.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    model_lstm.fit(X_lstm, y, epochs=3, verbose=0)
    
    ml_trained = True
    logging.info("✅ ML + LSTM модели обучены")

# === SIGNAL ===
async def send_signal(uid, asset):
    df = await get_twelvedata(asset)
    if df is None or len(df)<50:
        await bot.send_message(uid, f"⚠️ Нет данных для {asset}")
        return
    df = add_indicators(df)
    dir_ml, acc_ml = "neutral", 50
    if ml_trained:
        latest = df[["ema10","ema50","rsi","macd","atr","obv"]].iloc[-1]
        X = scaler.transform([latest])
        prob_gb = model_gb.predict_proba(X)[0]
        prob_lstm = model_lstm.predict(np.expand_dims(X,axis=1))[0][0]
        prob = (prob_gb[1]+prob_lstm)/2
        if prob>0.55: dir_ml="buy"
        elif prob<0.45: dir_ml="sell"
        acc_ml = int(prob*100)
    
    news_score = await get_news_sentiment(asset)
    direction = dir_ml
    accuracy = acc_ml
    if news_score>0.15 and direction!="sell":
        direction="buy"
        accuracy=min(100,accuracy+10)
    elif news_score<-0.15 and direction!="buy":
        direction="sell"
        accuracy=min(100,accuracy+10)
    
    price = df["close"].iloc[-1]
    atr = df["atr"].iloc[-1]
    tp_price = round(price + atr*2 if direction=="buy" else price - atr*2,2)
    sl_price = round(price - atr*1 if direction=="buy" else price + atr*1,2)
    
    msg = f"📢 Сигнал для <b>{asset}</b>\nНаправление: <b>{direction.upper()}</b>\nЦена: {price}\n🟢 TP: {tp_price}\n🔴 SL: {sl_price}\n📊 Точность: {accuracy}%\n📰 Новости: {'позитив' if news_score>0 else 'негатив' if news_score<0 else 'нейтрально'}"
    muted = user_settings.get(uid,{}).get("muted",False)
    await bot.send_message(uid,msg,disable_notification=muted)

# === HANDLERS ===
@dp.message(CommandStart())
async def start(message: types.Message):
    user_settings[message.from_user.id] = {"asset":"BTC/USD","muted":False}
    await message.answer("Escape the matrix.",reply_markup=get_main_keyboard())

@dp.message()
async def handle_buttons(message: types.Message):
    uid = message.from_user.id
    text = message.text
    if uid not in user_settings: user_settings[uid] = {"asset":"BTC/USD","muted":False}
    if text=="🔄 Получить сигнал":
        await send_signal(uid,user_settings[uid]["asset"])
    elif text in ASSETS:
        user_settings[uid]["asset"] = text
        await message.answer(f"✅ Актив установлен: {text}")
    elif text=="🔕 Mute":
        user_settings[uid]["muted"]=True
        await message.answer("🔕 Уведомления отключены")
    elif text=="🔔 Unmute":
        user_settings[uid]["muted"]=False
        await message.answer("🔔 Уведомления включены")

# === AUTO LOOP ===
async def auto_signal_loop():
    while True:
        for uid,settings in user_settings.items():
            await send_signal(uid,settings["asset"])
        await asyncio.sleep(900)

async def main():
    await train_models("BTC/USD")
    loop = asyncio.get_event_loop()
    loop.create_task(auto_signal_loop())
    await dp.start_polling(bot)

if __name__=="__main__":
    asyncio.run(main())
