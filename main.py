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
import joblib
import time

# ========== CONFIG ==========
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

if not TOKEN or not TWELVEDATA_API_KEY or not NEWSAPI_KEY:
    raise ValueError("API ключи не установлены в .env")

ASSETS = ["BTC/USD", "XAU/USD", "ETH/USD"]

# Logging
logger = logging.getLogger("bot")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console = logging.StreamHandler()
console.setFormatter(formatter)
file_handler = logging.FileHandler("bot.log")
file_handler.setFormatter(formatter)
logger.addHandler(console)
logger.addHandler(file_handler)

# aiogram setup
dp = Dispatcher()
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)

user_settings: dict = {}  # {user_id: {"asset": ..., "muted": False}}

# ========== ML models ==========
model_file = "rf_model.joblib"
scaler_file = "scaler.joblib"

model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
scaler = StandardScaler()
ml_trained = False

# ========== NLP ==========
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ========== UI keyboard ==========
def get_main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("🔄 Получить сигнал")],
            [KeyboardButton("BTC/USD"), KeyboardButton("XAU/USD"), KeyboardButton("ETH/USD")],
            [KeyboardButton("🔕 Mute"), KeyboardButton("🔔 Unmute")],
            [KeyboardButton("🕒 Расписание")]
        ],
        resize_keyboard=True,
    )

# ========== Data helpers ==========
async def fetch_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    return await resp.json()
        except Exception as e:
            logger.warning("Request failed (%d/%d): %s", attempt + 1, retries, e)
            await asyncio.sleep(1)
    return None

async def get_twelvedata(asset: str, interval: str = "1h", count: int = 1000) -> pd.DataFrame | None:
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": asset, "interval": interval, "outputsize": count, "apikey": TWELVEDATA_API_KEY}
    data = await fetch_json(url, params=params)
    if not data or "values" not in data:
        logger.warning("No data returned for %s", asset)
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

async def get_news_sentiment(asset: str) -> float:
    query = "bitcoin" if "BTC" in asset else "gold" if "XAU" in asset else "ethereum"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en&pageSize=5"
    data = await fetch_json(url)
    if not data or "articles" not in data:
        return 0.0
    scores = []
    for art in data["articles"][:5]:
        text = (art.get("title") or "") + " " + (art.get("description") or "")
        if text.strip():
            scores.append(sia.polarity_scores(text)["compound"])
    return float(np.mean(scores)) if scores else 0.0

# ========== Indicators ==========
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    sign = np.sign(df["close"].diff()).fillna(0)
    obv = (sign * df["volume"]).cumsum().fillna(0)
    return obv

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series) -> pd.Series:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "close" not in df.columns:
        raise ValueError("close column missing")
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"] = compute_macd(df["close"])
    df["atr"] = compute_atr(df)
    df["obv"] = compute_obv(df)
    ma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["momentum3"] = df["close"] - df["close"].shift(3)
    df["volatility14"] = df["close"].rolling(14).std()
    df["ret1"] = df["close"].pct_change(1)
    df = df.dropna().reset_index(drop=True)
    return df

# ========== Labels ==========
def make_labels(df: pd.DataFrame, horizon: int = 3, thr: float = 0.008) -> pd.Series:
    future_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    labels = pd.Series(0, index=df.index)
    labels[future_ret > thr] = 1
    labels[future_ret < -thr] = -1
    return labels

# ========== Rule-based ==========
def rule_based_signal(df: pd.DataFrame):
    latest = df.iloc[-1]
    ema_sig = 1 if latest["ema10"] > latest["ema50"] else -1
    rsi_sig = 1 if latest["rsi"] < 30 else -1 if latest["rsi"] > 70 else 0
    macd_sig = 1 if latest["macd"] > 0 else -1
    votes = [ema_sig, rsi_sig, macd_sig]
    s = sum(votes)
    if s > 0:
        return "buy", abs(s) / 3.0
    if s < 0:
        return "sell", abs(s) / 3.0
    return "neutral", 0.33

# ========== Training ==========
async def train_model_all(assets=ASSETS, count=1000, horizon=3, thr=0.008):
    global ml_trained, model, scaler
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        ml_trained = True
        logger.info("✅ ML model loaded from disk")
        return

    dfs = []
    for a in assets:
        df = await get_twelvedata(a, count=count)
        if df is None or len(df) < 200:
            logger.warning("Not enough data for %s", a)
            continue
        df = add_indicators(df)
        labels = make_labels(df, horizon=horizon, thr=thr)
        df = df.iloc[: len(labels)]
        df["label"] = labels.values
        df["asset"] = a
        dfs.append(df)
    if not dfs:
        logger.error("No data for training")
        return
    df_all = pd.concat(dfs, ignore_index=True).dropna()
    feature_cols = [
        "ema10", "ema50", "rsi", "macd",
        "atr", "obv", "bb_upper", "bb_lower",
        "momentum3", "volatility14", "ret1"
    ]
    X = df_all[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    y = df_all["label"].values
    model.fit(X_scaled, y)
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    ml_trained = True
    logger.info("✅ ML trained on %d samples", len(y))

# ========== ML prediction ==========
def ml_predict_row(row: pd.Series):
    if not ml_trained:
        return "neutral", 0.5
    feature_cols = [
        "ema10", "ema50", "rsi", "macd",
        "atr", "obv", "bb_upper", "bb_lower",
        "momentum3", "volatility14", "ret1"
    ]
    X = np.array([row[feature_cols].values])
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[0]
    classes = model.classes_
    prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
    best_class = max(prob_map.items(), key=lambda x: x[1])[0]
    conf = prob_map[best_class]
    if best_class == 1:
        return "buy", conf
    if best_class == -1:
        return "sell", conf
    return "neutral", conf

# ========== Combine scores ==========
def combine_scores(ml_dir, ml_conf, rule_dir, rule_conf, news_score):
    w_ml, w_rule, w_news = 0.5, 0.3, 0.2
    def dir_to_num(d, conf):
        return conf if d=="buy" else -conf if d=="sell" else 0
    total = w_ml*dir_to_num(ml_dir, ml_conf) + w_rule*dir_to_num(rule_dir, rule_conf) + w_news*np.clip(news_score, -1, 1)
    thresh = 0.35
    if total >= thresh:
        return "buy", float(total)
    if total <= -thresh:
        return "sell", float(abs(total))
    return "neutral", float(abs(total))

def compute_tp_sl(price: float, atr: float, max_tp_atr=2.0, max_sl_atr=1.0):
    tp = price + max_tp_atr*atr
    sl = price - max_sl_atr*atr
    tp_pct = (tp/price-1)*100
    sl_pct = (1 - sl/price)*100
    return round(tp,6), round(sl,6), round(tp_pct,2), round(sl_pct,2)

# ========== Send signal ==========
async def send_signal(uid: int, asset: str):
    try:
        df = await get_twelvedata(asset, count=150)
        if df is None or len(df) < 60:
            await bot.send_message(uid, f"⚠️ Недостаточно данных для {asset}")
            return
        df = add_indicators(df)
        rule_dir, rule_conf = rule_based_signal(df)
        ml_dir, ml_conf = ml_predict_row(df.iloc[-1])
        news = await get_news_sentiment(asset)
        final_dir, final_conf = combine_scores(ml_dir, ml_conf, rule_dir, rule_conf, news)
        send_threshold = 0.55
        if final_dir=="neutral" or final_conf<send_threshold:
            await bot.send_message(uid, f"⚠️ Нет уверенного сигнала для {asset} (score {final_conf:.2f})")
            return
        price = float(df["close"].iloc[-1])
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else price*0.01
        tp_price, sl_price, tp_pct, sl_pct = compute_tp_sl(price, atr)
        news_txt = "позитив" if news>0.05 else "негатив" if news<-0.05 else "нейтрально"
        accuracy_pct = min(99.99, round(final_conf*100,2))
        msg = (
            f"📢 Сигнал — <b>{asset}</b>\n"
            f"Направление: <b>{final_dir.upper()}</b>\n"
            f"Цена входа: <b>{price}</b>\n"
            f"🟢 TP: {tp_price} (+{tp_pct}%)\n"
            f"🔴 SL: {sl_price} (-{sl_pct}%)\n"
            f"📊 Надёжность: <b>{accuracy_pct}%</b>\n"
            f"📰 Новости: {news_txt}\n\n"
            f"🧾 ML={ml_dir}({ml_conf:.2f}), Rule={rule_dir}({rule_conf:.2f}), News={news:.2f}"
        )
        muted = user_settings.get(uid, {}).get("muted", False)
        await bot.send_message(uid, msg, disable_notification=muted)
    except Exception as e:
        logger.exception("send_signal failed")
        await bot.send_message(uid, f"⚠️ Ошибка при формировании сигнала: {e}")

# ========== Handlers ==========
@dp.message(CommandStart())
async def start_handler(msg: types.Message):
    user_settings[msg.from_user.id] = {"asset": "BTC/USD", "muted": False}
    await msg.answer("🤖 Бот готов. Выберите актив и запросите сигнал.", reply_markup=get_main_keyboard())

@dp.message()
async def all_messages_handler(msg: types.Message):
    uid = msg.from_user.id
    text = (msg.text or "").strip()
    if uid not in user_settings:
        user_settings[uid] = {"asset": "BTC/USD", "muted": False}
    lower = text.lower()
    if "mute" in lower or text.startswith("🔕") or text.startswith("🔇"):
        user_settings[uid]["muted"] = True
        await msg.answer("🔕 Автосигналы отключены.")
        return
    if "unmute" in lower or text.startswith("🔔") or text.startswith("🔊"):
        user_settings[uid]["muted"] = False
        await msg.answer("🔔 Автосигналы включены.")
        return
    if text=="🔄 Получить сигнал":
        await send_signal(uid, user_settings[uid]["asset"])
        return
    if text in ASSETS:
        user_settings[uid]["asset"] = text
        await msg.answer(f"✅ Актив установлен: {text}")
        return
    if text=="🕒 Расписание":
        await msg.answer("Авто-проверка каждые 15 минут.")
        return

# ========== Auto loop ==========
async def auto_signal_loop():
    while True:
        try:
            tasks = [send_signal(uid, s["asset"]) for uid,s in user_settings.items() if not s.get("muted",False)]
            if tasks:
                await asyncio.gather(*tasks)
        except Exception:
            logger.exception("auto loop error")
        await asyncio.sleep(900)

# ========== Main ==========
async def main():
    await train_model_all()
    asyncio.create_task(auto_signal_loop())
    await dp.start_polling(bot)

if __name__=="__main__":
    asyncio.run(main())
