"""
🧪 Smoke & Unit Tests — v3.1 (audit-fixes branch)

Kapsamı:
  • Feature Engineering: FEATURE_NAMES boyutu, parse fonksiyonları, vektör doğruluğu
  • Predictor: Initialize, MODEL_VERSION, predict (DB verisi varsa)
  • Config: RANDOM_SEED, N_FEATURES, TZ_ISTANBUL
  • Poisson: PoissonModel basit çıktı kontrolü

Çalıştırma:
  python -m pytest tests/test_smoke_v31.py -v
  veya
  python tests/test_smoke_v31.py  (standalone)
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Proje kök dizinini path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from config import (
    N_FEATURES,
    RANDOM_SEED,
    TZ_ISTANBUL,
    now_istanbul,
)
from feature_engineering import (
    FeatureExtractor,
    _form_to_points,
    _form_trend,
    _parse_turkish_date,
    _resolve_match_datetime,
    build_training_dataset,
)


# ═══════════════════════════════════════════════════════════════════
#  Config tests
# ═══════════════════════════════════════════════════════════════════

class TestConfig:
    def test_n_features_matches_feature_names(self):
        """N_FEATURES config sabiti, FeatureExtractor.FEATURE_NAMES ile eşleşmeli."""
        assert N_FEATURES == len(FeatureExtractor.FEATURE_NAMES), (
            f"Config N_FEATURES={N_FEATURES}, "
            f"FeatureExtractor.FEATURE_NAMES={len(FeatureExtractor.FEATURE_NAMES)}"
        )

    def test_n_features_is_96(self):
        """v3.1 feature sayısı 96 olmalı."""
        assert len(FeatureExtractor.FEATURE_NAMES) == 96

    def test_random_seed_default(self):
        """Varsayılan RANDOM_SEED = 42."""
        assert RANDOM_SEED == 42

    def test_now_istanbul_is_aware(self):
        """now_istanbul() timezone-aware olmalı."""
        dt = now_istanbul()
        assert dt.tzinfo is not None
        assert str(dt.tzinfo) == "Europe/Istanbul"


# ═══════════════════════════════════════════════════════════════════
#  Feature Engineering — parse fonksiyonları
# ═══════════════════════════════════════════════════════════════════

class TestTurkishDateParse:
    """_parse_turkish_date() kapsamlı testleri."""

    def test_turkish_month_short(self):
        dt = _parse_turkish_date("31 Oca", ref_year=2025)
        assert dt.year == 2025 and dt.month == 1 and dt.day == 31

    def test_turkish_month_december(self):
        dt = _parse_turkish_date("5 Ara", ref_year=2024)
        assert dt.year == 2024 and dt.month == 12 and dt.day == 5

    def test_dot_format(self):
        dt = _parse_turkish_date("02.08.2025")
        assert dt.year == 2025 and dt.month == 8 and dt.day == 2

    def test_all_returns_aware(self):
        """Tüm formatlar timezone-aware datetime dönmeli."""
        dt1 = _parse_turkish_date("31 Oca", ref_year=2025)
        dt2 = _parse_turkish_date("02.08.2025")
        dt3 = _parse_turkish_date("Bugün")
        for dt in (dt1, dt2, dt3):
            assert dt is not None
            assert dt.tzinfo is not None, f"datetime should be aware: {dt}"

    def test_none_returns_none(self):
        assert _parse_turkish_date(None) is None

    def test_empty_returns_none(self):
        assert _parse_turkish_date("") is None

    def test_bugun(self):
        dt = _parse_turkish_date("Bugün")
        assert dt is not None
        # Tarih, bugüne yakın olmalı (timezone farkı hesaba katılır)
        today = now_istanbul().date()
        assert abs((dt.date() - today).days) <= 1

    def test_yarin(self):
        dt = _parse_turkish_date("Yarın")
        assert dt is not None
        tomorrow = (now_istanbul() + timedelta(days=1)).date()
        assert abs((dt.date() - tomorrow).days) <= 1


class TestFormCalculations:
    """_form_to_points() ve _form_trend() testleri."""

    def test_all_wins(self):
        assert _form_to_points("GGGGG") == 100.0

    def test_all_losses(self):
        assert _form_to_points("MMMMM") == 0.0

    def test_mixed_form(self):
        result = _form_to_points("GBMBG")
        assert 40.0 < result < 65.0  # yaklaşık 53

    def test_empty_form(self):
        result = _form_to_points("")
        assert result == 0.0 or result == 50.0  # boş: 0 veya orta (impl'a bağlı)

    def test_trend_upward(self):
        trend = _form_trend("MGBGG")
        assert trend > 0, f"Upward trend expected, got {trend}"

    def test_trend_downward(self):
        trend = _form_trend("GGBMM")
        assert trend < 0, f"Downward trend expected, got {trend}"


# ═══════════════════════════════════════════════════════════════════
#  Feature Engineering — FeatureExtractor
# ═══════════════════════════════════════════════════════════════════

class TestFeatureExtractor:
    """FeatureExtractor vektör çıktı kontrolleri (DB bağımlı)."""

    def test_feature_names_unique(self):
        """Tüm feature isimleri benzersiz olmalı."""
        names = FeatureExtractor.FEATURE_NAMES
        assert len(names) == len(set(names)), "Duplicate feature names detected"

    def test_feature_names_v31_set(self):
        """v3.1 ile eklenen Bayesian dampening feature'ları mevcut olmalı."""
        expected_v31 = [
            "season_progress",
            "season_confidence",
            "dampened_home_rank",
            "dampened_away_rank",
            "relative_market_strength_home",
            "relative_market_strength_away",
            "early_season_reliability",
        ]
        for fname in expected_v31:
            assert fname in FeatureExtractor.FEATURE_NAMES, (
                f"v3.1 feature eksik: {fname}"
            )

    @pytest.fixture
    def db_session(self):
        """Yalnızca DB testi çalıştırılırsa oturum döndürür."""
        from database import get_session, init_db
        init_db()
        with get_session() as session:
            yield session

    def test_extract_vector_dimensions(self, db_session):
        """Gerçek bir maç varsa vektör boyutu 96 olmalı."""
        from models import Match
        match = db_session.query(Match).first()
        if match is None:
            pytest.skip("DB boş — maç bulunamadı")
        extractor = FeatureExtractor(db_session)
        vector = extractor.extract_vector(match)
        assert len(vector) == 96, f"Expected 96-dim vector, got {len(vector)}"

    def test_extract_dict_keys(self, db_session):
        """extract() sözlük anahtarları FEATURE_NAMES ile birebir eşleşmeli."""
        from models import Match
        match = db_session.query(Match).first()
        if match is None:
            pytest.skip("DB boş — maç bulunamadı")
        extractor = FeatureExtractor(db_session)
        features = extractor.extract(match)
        assert set(features.keys()) == set(FeatureExtractor.FEATURE_NAMES)


# ═══════════════════════════════════════════════════════════════════
#  Training Dataset
# ═══════════════════════════════════════════════════════════════════

class TestBuildTrainingDataset:
    @pytest.fixture
    def db_session(self):
        from database import get_session, init_db
        init_db()
        with get_session() as session:
            yield session

    def test_returns_arrays(self, db_session):
        X, y = build_training_dataset(db_session)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_feature_dimension(self, db_session):
        X, y = build_training_dataset(db_session)
        if len(X) == 0:
            pytest.skip("DB'de eğitim verisi yok")
        assert X.shape[1] == 96, f"Expected 96, got {X.shape[1]}"


# ═══════════════════════════════════════════════════════════════════
#  Predictor
# ═══════════════════════════════════════════════════════════════════

class TestPredictor:
    @pytest.fixture
    def db_session(self):
        from database import get_session, init_db
        init_db()
        with get_session() as session:
            yield session

    def test_model_version(self):
        from predictor import MatchPredictor
        assert MatchPredictor.MODEL_VERSION == "v3.1"

    def test_initialize(self, db_session):
        from predictor import MatchPredictor
        predictor = MatchPredictor(db_session)
        status = predictor.initialize()
        assert isinstance(status, str)

    def test_predict_output(self, db_session):
        from models import Match
        from predictor import MatchPredictor
        match = db_session.query(Match).first()
        if match is None:
            pytest.skip("DB boş — maç bulunamadı")
        predictor = MatchPredictor(db_session)
        predictor.initialize()
        result = predictor.predict(match)
        assert result.prediction in ("1", "X", "2")
        assert 0 <= result.confidence <= 100
        assert result.risk_level in (
            "🟢 Düşük Risk", "🟡 Orta Risk", "🔴 Yüksek Risk",
        )


# ═══════════════════════════════════════════════════════════════════
#  Poisson Model
# ═══════════════════════════════════════════════════════════════════

class TestPoissonModel:
    def test_basic_prediction(self):
        from poisson_model import PoissonModel
        pm = PoissonModel()
        result = pm.predict(
            home_attack=1.2,
            home_defense=0.9,
            away_attack=1.0,
            away_defense=1.1,
        )
        assert hasattr(result, "prob_home")
        assert hasattr(result, "prob_draw")
        assert hasattr(result, "prob_away")
        total = result.prob_home + result.prob_draw + result.prob_away
        assert abs(total - 100.0) < 1.0, f"Probabilities don't sum to 100: {total}"


# ═══════════════════════════════════════════════════════════════════
#  Standalone çalıştırma
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
