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

# ========== CONFIG ==========
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

ASSETS = ["BTC/USD", "XAU/USD", "ETH/USD"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# aiogram setup
dp = Dispatcher()
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)

user_settings: dict = {}  # {user_id: {"asset": ..., "muted": False}}

# ========== ML models ==========
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
async def get_twelvedata(asset: str, interval: str = "1h", count: int = 1000) -> pd.DataFrame | None:
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": asset, "interval": interval, "outputsize": count, "apikey": TWELVEDATA_API_KEY}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=30) as resp:
                data = await resp.json()
    except Exception as e:
        logger.exception("TwelveData request failed")
        return None

    if not isinstance(data, dict) or "values" not in data:
        logger.warning("TwelveData returned no values: %s", data)
        return None

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    # numeric cast only for present columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

async def get_news_sentiment(asset: str) -> float:
    # return VADER compound average on last 5 articles (english)
    query = "bitcoin" if "BTC" in asset else "gold" if "XAU" in asset else "ethereum"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&language=en&pageSize=5"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=20) as r:
                data = await r.json()
    except Exception:
        return 0.0
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
    # handle missing columns gracefully
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
    # thr = 0.8% default for 1h horizon
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
        return "buy", abs(s) / 3.0  # normalized confidence 0..1
    if s < 0:
        return "sell", abs(s) / 3.0
    return "neutral", 0.33

# ========== Training ==========
async def train_model_all(assets=ASSETS, count=1000, horizon=3, thr=0.008):
    global ml_trained, model, scaler
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
    # train
    model.fit(X_scaled, y)
    ml_trained = True
    logger.info("✅ ML trained on %d samples", len(y))

# ========== Prediction helper ==========
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
    # build dictionary class->prob
    prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
    buy_p = prob_map.get(1, 0.0)
    sell_p = prob_map.get(-1, 0.0)
    neutral_p = prob_map.get(0, 0.0)
    # choose highest
    best_class = max(prob_map.items(), key=lambda x: x[1])[0]
    conf = prob_map[best_class]
    if best_class == 1:
        return "buy", conf
    if best_class == -1:
        return "sell", conf
    return "neutral", conf

# ========== Compose final decision and TP/SL ==========
def combine_scores(ml_dir: str, ml_conf: float, rule_dir: str, rule_conf: float, news_score: float):
    # ml_conf in 0..1, rule_conf 0..1, news_score ~ [-1,1]
    w_ml = 0.5
    w_rule = 0.3
    w_news = 0.2

    # map directions to numeric
    def dir_to_num(d, conf):
        if d == "buy":
            return conf
        if d == "sell":
            return -conf
        return 0.0

    ml_val = dir_to_num(ml_dir, ml_conf)
    rule_val = dir_to_num(rule_dir, rule_conf)
    news_val = np.clip(news_score, -1.0, 1.0)  # already -1..1

    total = w_ml * ml_val + w_rule * rule_val + w_news * news_val
    # final direction threshold
    thresh = 0.35  # tuned: higher -> fewer signals but higher precision
    if total >= thresh:
        return "buy", float(total)
    if total <= -thresh:
        return "sell", float(abs(total))
    return "neutral", float(abs(total))

def compute_tp_sl(price: float, atr: float, max_tp_atr=2.0, max_sl_atr=1.0):
    # TP = price + max_tp_atr * atr  (for buy)
    # SL = price - max_sl_atr * atr
    tp = price + max_tp_atr * atr
    sl = price - max_sl_atr * atr
    tp_pct = (tp / price - 1.0) * 100
    sl_pct = (1.0 - sl / price) * 100
    return round(tp, 6), round(sl, 6), round(tp_pct, 2), round(sl_pct, 2)

# ========== Send signal ==========
async def send_signal(uid: int, asset: str):
    try:
        df = await get_twelvedata(asset, count=150)
        if df is None or len(df) < 60:
            await bot.send_message(uid, f"⚠️ Не удалось получить достаточные данные для {asset}")
            return
        df = add_indicators(df)
        # rule
        rule_dir, rule_conf = rule_based_signal(df)
        # ml
        ml_dir, ml_conf = ml_predict_row(df.iloc[-1])
        # news
        news = await get_news_sentiment(asset)  # [-1..1]
        # combine
        final_dir, final_conf = combine_scores(ml_dir, ml_conf, rule_dir, rule_conf, news)
        # require ML minimum confidence and combined confidence
        send_threshold = 0.55  # require >55% combined-ish to send
        if final_dir == "neutral" or final_conf < send_threshold:
            # don't spam: only notify when clear
            await bot.send_message(uid, f"⚠️ По {asset} нет уверенного сигнала (score {final_conf:.2f}).")
            return

        price = float(df["close"].iloc[-1])
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else (0.0)
        # fallback ATR-> percent if zero: use volatility14
        if not atr or np.isnan(atr) or atr <= 0:
            atr = float(df["volatility14"].iloc[-1]) if "volatility14" in df.columns else price * 0.01

        tp_price, sl_price, tp_pct, sl_pct = compute_tp_sl(price, atr, max_tp_atr=2.0, max_sl_atr=1.0)
        accuracy_pct = min(99.99, round(final_conf * 100, 2))
        news_txt = "позитив" if news > 0.05 else "негатив" if news < -0.05 else "нейтрально"

        msg = (
            f"📢 Сигнал — <b>{asset}</b>\n"
            f"Направление: <b>{final_dir.upper()}</b>\n"
            f"Цена входа: <b>{price}</b>\n"
            f"🟢 TP: {tp_price} (+{tp_pct}%)\n"
            f"🔴 SL: {sl_price} (-{sl_pct}%)\n"
            f"📊 Надёжность (оценка): <b>{accuracy_pct}%</b>\n"
            f"📰 Новости: {news_txt}\n\n"
            f"🧾 Дополнительно: ML={ml_dir}({ml_conf:.2f}), Rule={rule_dir}({rule_conf:.2f}), News={news:.2f}"
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
    # initialize
    if uid not in user_settings:
        user_settings[uid] = {"asset": "BTC/USD", "muted": False}
    lower = text.lower()
    # robust mute/unmute detection
    if "mute" in lower or text.startswith("🔕") or text.startswith("🔇"):
        user_settings[uid]["muted"] = True
        await msg.answer("🔕 Автосигналы отключены.")
        return
    if "unmute" in lower or text.startswith("🔔") or text.startswith("🔊"):
        user_settings[uid]["muted"] = False
        await msg.answer("🔔 Автосигналы включены.")
        return
    if text == "🔄 Получить сигнал":
        await send_signal(uid, user_settings[uid]["asset"])
        return
    if text in ASSETS:
        user_settings[uid]["asset"] = text
        await msg.answer(f"✅ Актив установлен: {text}")
        return
    if text == "🕒 Расписание":
        await msg.answer("Авто-проверка выполняется каждые 15 минут.")
        return
    # fallback: ignore other messages
    return

# ========== Auto loop ==========
async def auto_signal_loop():
    while True:
        try:
            for uid, s in list(user_settings.items()):
                if not s.get("muted", False):
                    await send_signal(uid, s["asset"])
        except Exception:
            logger.exception("auto loop error")
        await asyncio.sleep(900)

# ========== Main ==========
async def main():
    # train on all assets (may take some seconds)
    await train_model_all()
    # start auto loop
    asyncio.create_task(auto_signal_loop())
    # start polling
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
