"""
⚙️ Konfigürasyon Modülü
Tüm proje ayarlarını merkezi olarak yönetir.
"""

import os
from pathlib import Path
from zoneinfo import ZoneInfo
import datetime as _dt

# ─── Dizin Ayarları ───────────────────────────────────────────────
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = BASE_DIR / "nesine_futbol.db"
MODEL_DIR = BASE_DIR / "models_cache"
LOG_DIR = BASE_DIR / "logs"

# Gerekli dizinleri oluştur
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ─── Veritabanı Ayarları ──────────────────────────────────────────
# PostgreSQL için environment variable'lardan oku, yoksa SQLite fallback
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "nesine_futbol")

if POSTGRES_USER and POSTGRES_PASSWORD:
    DATABASE_URL = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    DB_ENGINE = "postgresql"
else:
    # Lokal geliştirme için SQLite fallback
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
    DB_ENGINE = "sqlite"

# ─── Scraper Ayarları ─────────────────────────────────────────────
NESINE_URL = "https://www.nesine.com/iddaa?et=1&le=2"
SCRAPER_TIMEOUT = 15
DEFAULT_MATCH_COUNT = 20

# ─── ML Model Ayarları ────────────────────────────────────────────
MIN_TRAINING_SAMPLES = 50       # Cold-start eşiği: bunun altında Poisson kullanılır
MIN_TRAINING_SAMPLES_XGBOOST = 200  # XGBoost için minimum örnek
COLD_START_MODE = "poisson"     # "poisson" veya "static"
MODEL_RETRAIN_INTERVAL = 24     # Saat cinsinden yeniden eğitim aralığı

# Poisson Ayarları
POISSON_MAX_GOALS = 7           # Poisson dağılımında maksimum gol sayısı
LEAGUE_AVG_GOALS = 2.65         # Lig ortalaması gol beklentisi

# ML Feature Ayarları
FORM_WINDOW = 5                 # Son N maç formu
H2H_WINDOW = 10                # H2H son N maç\n\n# Feature sayısı (feature_engineering.FEATURE_NAMES ile senkron tutulmalı)\nN_FEATURES = 96                # v3.1: 85 base + 11 season dampening

# Sezon Başı Sönümleme (Bayesian Smoothing) Ayarları
TYPICAL_SEASON_LENGTH = 34      # Tipik lig sezonu maç sayısı
BAYESIAN_PRIOR_MATCHES = 5     # Bayesian prior sabiti (C) — "ortalama takım" varsayımı
EARLY_SEASON_THRESHOLD = 0.20  # Erken sezon eşiği (0.20 ≈ ilk ~7 maç)

# ─── Value Bet Ayarları ───────────────────────────────────────────
VALUE_BET_MIN_EDGE = 5.0        # Minimum edge yüzdesi (%)
VALUE_BET_MIN_CONFIDENCE = 60   # Minimum güven puanı (%)
MAX_REPORT_MATCHES = 30         # Raporda gösterilecek maks maç

# ─── Ağırlık Sabitleri (Fallback / Cold-Start) ───────────────────
FALLBACK_WEIGHTS = {
    'form': 0.20,
    'hakem': 0.15,
    'h2h': 0.15,
    'value': 0.20,
    'lig': 0.15,
    'eksik': 0.15,
}

# ─── Timezone ─────────────────────────────────────────────────────
TZ_ISTANBUL = ZoneInfo("Europe/Istanbul")

def now_istanbul() -> _dt.datetime:
    """Timezone-aware şu anki Istanbul saatini döndürür."""
    return _dt.datetime.now(tz=TZ_ISTANBUL)

# ─── Reproducibility ─────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ─── Loglama ──────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
