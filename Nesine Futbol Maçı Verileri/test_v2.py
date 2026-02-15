"""Feature Engineering v2.1 + Predictor entegrasyon testi."""
from datetime import datetime
from feature_engineering import (
    FeatureExtractor, _parse_turkish_date,
    _form_to_points, _form_trend, _resolve_match_datetime,
    build_training_dataset,
)

N_FEATURES = 61  # v2.1

# ── Test 1: FEATURE_NAMES boyutu ──
assert len(FeatureExtractor.FEATURE_NAMES) == N_FEATURES, \
    f"Expected {N_FEATURES}, got {len(FeatureExtractor.FEATURE_NAMES)}"
print(f"✓ Test 1: FEATURE_NAMES = {len(FeatureExtractor.FEATURE_NAMES)} feature")

# ── Test 2: Türkçe tarih parse ──
dt1 = _parse_turkish_date("31 Oca", ref_year=2025)
assert dt1 == datetime(2025, 1, 31), f"Expected Jan 31, got {dt1}"
print(f'✓ Test 2a: "31 Oca" → {dt1}')

dt2 = _parse_turkish_date("02.08.2025")
assert dt2 == datetime(2025, 8, 2), f"Expected Aug 2, got {dt2}"
print(f'✓ Test 2b: "02.08.2025" → {dt2}')

dt3 = _parse_turkish_date("5 Ara", ref_year=2024)
assert dt3 == datetime(2024, 12, 5), f"Expected Dec 5, got {dt3}"
print(f'✓ Test 2c: "5 Ara" → {dt3}')

assert _parse_turkish_date(None) is None
print("✓ Test 2d: None → None")

# ── Test 2e: Relative dates (v2.1) ──
dt_bugun = _parse_turkish_date("Bugün")
assert dt_bugun is not None, "Bugün → None olmamalı"
assert dt_bugun.date() == datetime.now().date(), f"Bugün parse hatası: {dt_bugun}"
print(f'✓ Test 2e: "Bugün" → {dt_bugun.date()}')

dt_yarin = _parse_turkish_date("Yarın")
assert dt_yarin is not None
from datetime import timedelta
assert dt_yarin.date() == (datetime.now() + timedelta(days=1)).date()
print(f'✓ Test 2f: "Yarın" → {dt_yarin.date()}')

# ── Test 3: Form hesaplamaları ──
assert _form_to_points("GGGGG") == 100.0
assert _form_to_points("MMMMM") == 0.0
assert abs(_form_to_points("GBMBG") - 53.33) < 1.0
print("✓ Test 3: form_to_points çalışıyor")

# ── Test 4: Form trend ──
trend_up = _form_trend("MGBGG")
assert trend_up > 0, f"Expected positive trend, got {trend_up}"
trend_down = _form_trend("GGBMM")
assert trend_down < 0, f"Expected negative trend, got {trend_down}"
print(f"✓ Test 4: form_trend (up={trend_up:.2f}, down={trend_down:.2f})")

# ── Test 5-10: DB entegrasyonu ──
from database import get_session, init_db
from models import Match
init_db()  # tabloları oluştur (yoksa)

with get_session() as session:
    match = session.query(Match).first()
    if match:
        extractor = FeatureExtractor(session)
        features = extractor.extract(match)
        vector = extractor.extract_vector(match)
        assert len(vector) == N_FEATURES, f"Expected {N_FEATURES}-dim, got {len(vector)}"
        assert len(features) == N_FEATURES, f"Expected {N_FEATURES} keys, got {len(features)}"
        print(f"✓ Test 5a: {match.display_name} → {len(features)} feature")

        # v2.1 feature'larını kontrol et
        v21_features = [
            # v2.0'dan gelen
            "league_position_composite", "ref_home_bias", "ref_over_tendency",
            "ref_kg_var_pct", "h2h_recent_trend", "h2h_avg_goals",
            "h2h_odds_accuracy", "home_critical_injury_count",
            "away_critical_injury_count", "total_injury_importance",
            "form_adjusted_home_score", "form_adjusted_away_score",
            "home_strength_composite", "away_strength_composite",
            "strength_diff",
            # v2.1'de eklenen
            "away_win_rate",
            "ref_alignment_score",
            "injury_normalized_score",
            "referee_tahmin_uyumu",
            "h2h_tahmin_uyumu",
        ]
        for nf in v21_features:
            assert nf in features, f"Missing feature: {nf}"
        print(f"✓ Test 5b: {len(v21_features)} feature mevcut (v2.1 dahil)")
    else:
        print("⚠ Test 5: DB boş, maç bulunamadı")

    # ── Test 6: build_training_dataset ──
    X, y = build_training_dataset(session)
    print(f"✓ Test 6a: build_training_dataset → X.shape={X.shape}, y.shape={y.shape}")
    if len(X) > 0:
        assert X.shape[1] == N_FEATURES, \
            f"Feature boyutu {N_FEATURES} olmalı, {X.shape[1]} bulundu"
        print(f"✓ Test 6b: Feature boyutu doğru ({N_FEATURES})")

    # ── Test 7: Predictor import & initialize ──
    from predictor import MatchPredictor
    predictor = MatchPredictor(session)
    status = predictor.initialize()
    print(f"✓ Test 7a: Predictor başlatıldı → {status}")
    assert predictor.MODEL_VERSION == "v2.1", \
        f"Expected v2.1, got {predictor.MODEL_VERSION}"
    print("✓ Test 7b: MODEL_VERSION = v2.1")

    # ── Test 8: Predictor predict ──
    if match:
        result = predictor.predict(match)
        assert result.prediction in ("1", "X", "2")
        assert result.model_version.startswith(
            ("poisson_v2.1", "hybrid_v2.1", "ml_v2.1")
        ), f"Unexpected version: {result.model_version}"
        print(f"✓ Test 8: predict() → {result.prediction} "
              f"(conf={result.confidence:.1f}%, engine={result.engine_used})")
        print(f"   Açıklama: {result.explanation[:140]}...")

    # ── Test 9: _resolve_match_datetime ──
    if match:
        rdt = _resolve_match_datetime(match)
        assert isinstance(rdt, datetime)
        print(f"✓ Test 9: _resolve_match_datetime → {rdt}")

    # ── Test 10: Risk & explanation kalite testi ──
    if match:
        result = predictor.predict(match)
        assert result.risk_level in (
            "🟢 Düşük Risk", "🟡 Orta Risk", "🔴 Yüksek Risk"
        ), f"Unexpected risk: {result.risk_level}"
        assert len(result.explanation) > 20, "Açıklama çok kısa"
        print(f"✓ Test 10: Risk={result.risk_level}, Açıklama uzunluğu={len(result.explanation)}")

print()
print("=" * 50)
print("✅ TÜM TESTLER BAŞARILI — v2.1 hazır!")
