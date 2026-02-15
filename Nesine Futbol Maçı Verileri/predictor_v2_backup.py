"""
🤖 Hibrit Tahmin Motoru v2.1 (MatchPredictor)
Poisson + ML (XGBoost/RandomForest) ile maç sonucu tahmini.

Akış:
  1. Veritabanındaki sonuçlanmış maç sayısını kontrol et
  2. Yeterli veri yoksa → Poisson fallback
  3. Yeterli veri varsa  → ML modeli eğit & tahmin yap
  4. Her iki motorun çıktılarını birleştir (ensemble)

Cold-Start Mekanizması:
  • < 50  maç  → Saf Poisson
  • 50-200 maç → Poisson (%60) + ML (%40)  hibrit
  • > 200 maç  → ML ağırlıklı (%70) + Poisson (%30) doğrulama

v2.1 Değişiklikleri (v2.0 üzerinden):
  ✓ 61-feature vektörü desteği (feature_engineering v2.1)
  ✓ MODEL_VERSION = "v2.1" + otomatik cache invalidation
  ✓ _determine_risk: Zengin risk faktörleri (sakatlık normalize skor,
    hakem-tahmin çelişkisi, form trendi tutarsızlığı)
  ✓ _generate_explanation: value_bet_analyzer kalitesinde insan-okunur
    açıklama (lig pozisyonu, ref alignment, H2H trend, sakatlık detay)
  ✓ Tüm fonksiyonlarda tip hint'leri
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from config import (
    MIN_TRAINING_SAMPLES,
    MIN_TRAINING_SAMPLES_XGBOOST,
    MODEL_DIR,
    VALUE_BET_MIN_EDGE,
    VALUE_BET_MIN_CONFIDENCE,
)
from models import Match, Odds, Prediction
from feature_engineering import FeatureExtractor, build_training_dataset
from poisson_model import PoissonModel, PoissonResult

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Tahmin Sonucu Veri Sınıfı
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class PredictionResult:
    """Tek bir maç için birleşik tahmin sonucu."""

    match_id: int
    match_display: str
    engine_used: str          # "poisson" | "ml" | "hybrid"
    model_version: str

    # Olasılıklar (%)
    prob_home: float = 0.0
    prob_draw: float = 0.0
    prob_away: float = 0.0
    prob_over_25: float = 0.0
    prob_under_25: float = 0.0

    # Poisson beklentileri
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0
    top_scores: List[Tuple[str, float]] = field(default_factory=list)

    # Final tahmin
    prediction: str = ""       # "1" | "X" | "2"
    confidence: float = 0.0
    value_edge: float = 0.0
    is_value_bet: bool = False
    risk_level: str = ""

    explanation: str = ""

    def to_prediction_model(self) -> Dict[str, object]:
        """``Prediction`` ORM modeli için dict döndürür."""
        return {
            "engine_used": self.engine_used,
            "model_version": self.model_version,
            "prob_home": self.prob_home,
            "prob_draw": self.prob_draw,
            "prob_away": self.prob_away,
            "prob_over_25": self.prob_over_25,
            "prob_under_25": self.prob_under_25,
            "expected_home_goals": self.expected_home_goals,
            "expected_away_goals": self.expected_away_goals,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "value_edge": self.value_edge,
            "is_value_bet": self.is_value_bet,
            "risk_level": self.risk_level,
            "explanation": self.explanation,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ana Tahmin Sınıfı
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MatchPredictor:
    """Hibrit tahmin motoru.

    Kullanım::

        predictor = MatchPredictor(session)
        predictor.initialize()          # Modeli eğit veya yükle
        result = predictor.predict(match)
    """

    MODEL_FILE = MODEL_DIR / "match_predictor.pkl"
    MODEL_VERSION: str = "v2.1"

    def __init__(self, session: Session) -> None:
        self.session: Session = session
        self.extractor: FeatureExtractor = FeatureExtractor(session)
        self.poisson: PoissonModel = PoissonModel()
        self.ml_model: object | None = None
        self.training_samples: int = 0
        self._mode: str = "poisson"  # "poisson" | "hybrid" | "ml"

    # ─── Başlatma / Eğitim ────────────────────────────────────────

    def initialize(self) -> str:
        """Modeli başlatır. Veri miktarına göre mod seçer.

        Returns
        -------
        str
            Aktif mod açıklaması.
        """
        finished: int = (
            self.session.query(Match)
            .filter(Match.is_finished == True)  # noqa: E712
            .count()
        )
        self.training_samples = finished

        if finished < MIN_TRAINING_SAMPLES:
            self._mode = "poisson"
            logger.info(
                "📊 Cold-Start modu: Poisson (%d/%d maç)",
                finished, MIN_TRAINING_SAMPLES,
            )
            return f"Poisson (Cold-Start: {finished} maç)"

        # Kayıtlı model var mı kontrol et
        if self._load_model():
            if finished >= MIN_TRAINING_SAMPLES_XGBOOST:
                self._mode = "ml"
            else:
                self._mode = "hybrid"
            logger.info("📊 Model yüklendi: %s mod", self._mode)
            return f"{self._mode} (kayıtlı model, {finished} maç)"

        # Yeni model eğit
        return self._train_model(finished)

    def _train_model(self, finished: int) -> str:
        """ML modelini eğitir.

        v2.1:
          • 61-feature vektörü (feature_engineering v2.1)
          • Kronolojik train/test split (son %20 test) — data leakage fix
          • Eski pickle → otomatik silme (feature boyutu değişimi)
        """
        logger.info("🔧 ML modeli eğitiliyor (v2.1 — temporal split)...")

        # Eski modeli temizle (feature boyutu değişmiş olabilir)
        if self.MODEL_FILE.exists():
            self.MODEL_FILE.unlink()
            logger.info("🗑  Eski model cache silindi (feature vektörü güncellendi)")

        X, y = build_training_dataset(self.session)

        if len(X) < MIN_TRAINING_SAMPLES:
            self._mode = "poisson"
            return f"Poisson (yetersiz eğitim verisi: {len(X)})"

        # XGBoost veya RandomForest seçimi
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
            )
            model_name = "XGBoost"
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
            model_name = "RandomForest"

        # ── Kronolojik Eğitim / Test bölünmesi (DATA LEAKAGE FIX) ──
        split_idx: int = int(len(X) * 0.80)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # NaN kontrolü
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        model.fit(X_train, y_train)

        accuracy: float = model.score(X_test, y_test)
        logger.info(
            "✓ %s eğitildi: Doğruluk = %.2f%% (temporal split)",
            model_name, accuracy * 100,
        )

        self.ml_model = model
        self._save_model()

        if finished >= MIN_TRAINING_SAMPLES_XGBOOST:
            self._mode = "ml"
        else:
            self._mode = "hybrid"

        return f"{self._mode} ({model_name}, doğruluk: {accuracy:.2%}, {finished} maç)"

    def retrain(self) -> str:
        """Modeli yeniden eğitir."""
        finished: int = (
            self.session.query(Match)
            .filter(Match.is_finished == True)  # noqa: E712
            .count()
        )
        self.training_samples = finished
        return self._train_model(finished)

    # ─── Model Kaydetme / Yükleme ────────────────────────────────

    def _save_model(self) -> None:
        """Modeli diske kaydeder."""
        if self.ml_model is None:
            return
        try:
            with open(self.MODEL_FILE, "wb") as fp:
                pickle.dump(
                    {
                        "model": self.ml_model,
                        "version": self.MODEL_VERSION,
                        "samples": self.training_samples,
                        "n_features": len(FeatureExtractor.FEATURE_NAMES),
                    },
                    fp,
                )
            logger.info("✓ Model kaydedildi: %s", self.MODEL_FILE)
        except Exception as e:
            logger.error("Model kaydetme hatası: %s", e)

    def _load_model(self) -> bool:
        """Kaydedilmiş modeli yükler.

        v2.1: Versiyon + feature boyutu uyumsuzluğu kontrolü.
        """
        if not self.MODEL_FILE.exists():
            return False
        try:
            with open(self.MODEL_FILE, "rb") as fp:
                data: dict = pickle.load(fp)  # noqa: S301
            saved_version: str = data.get("version", "v1.0")
            saved_n_features: int = data.get("n_features", 0)
            expected_n: int = len(FeatureExtractor.FEATURE_NAMES)

            # Versiyon veya feature boyutu uyumsuzluğu → yeniden eğit
            if saved_version != self.MODEL_VERSION or saved_n_features != expected_n:
                logger.info(
                    "⚠️  Model uyumsuz (v=%s→%s, feat=%d→%d), yeniden eğitilecek",
                    saved_version, self.MODEL_VERSION,
                    saved_n_features, expected_n,
                )
                self.MODEL_FILE.unlink()
                return False

            self.ml_model = data["model"]
            logger.info("✓ Model yüklendi (versiyon: %s)", saved_version)
            return True
        except Exception as e:
            logger.warning("Model yükleme hatası: %s", e)
            return False

    # ─── Tahmin ───────────────────────────────────────────────────

    def predict(self, match: Match) -> PredictionResult:
        """Bir maç için tahmin üretir.

        Mod'a göre Poisson, ML veya hibrit kullanır.
        """
        features: Dict[str, float] = self.extractor.extract(match)
        feature_vector: np.ndarray = self.extractor.extract_vector(match)

        # Poisson tahmini (her zaman hesapla)
        poisson_result: PoissonResult = self.poisson.predict_from_features(features)

        # Oran bilgisi
        odds: Optional[Odds] = (
            self.session.query(Odds).filter_by(match_id=match.id).first()
        )

        if self._mode == "poisson":
            return self._build_poisson_prediction(
                match, poisson_result, features, odds,
            )
        elif self._mode == "hybrid":
            return self._build_hybrid_prediction(
                match, poisson_result, feature_vector, features, odds,
            )
        else:  # ml
            return self._build_ml_prediction(
                match, poisson_result, feature_vector, features, odds,
            )

    def predict_batch(self, matches: List[Match]) -> List[PredictionResult]:
        """Birden fazla maç için toplu tahmin."""
        results: List[PredictionResult] = []
        for match in matches:
            try:
                results.append(self.predict(match))
            except Exception as e:
                logger.error("Tahmin hatası (%s): %s", match.display_name, e)
        return results

    # ─── Tahmin Oluşturucuları ────────────────────────────────────

    def _build_poisson_prediction(
        self,
        match: Match,
        pr: PoissonResult,
        features: Dict[str, float],
        odds: Optional[Odds],
    ) -> PredictionResult:
        """Saf Poisson tahmini."""
        prediction: str = pr.prediction
        probs: Dict[str, float] = {
            "1": pr.prob_home, "X": pr.prob_draw, "2": pr.prob_away,
        }
        confidence: float = probs[prediction]

        edge, is_value = self._calc_value_edge(prediction, probs, odds)
        risk: str = self._determine_risk(confidence, features, prediction)
        explanation: str = self._generate_explanation(
            features, pr, prediction, confidence, "poisson",
        )

        return PredictionResult(
            match_id=match.id,
            match_display=match.display_name,
            engine_used="poisson",
            model_version=f"poisson_{self.MODEL_VERSION}",
            prob_home=pr.prob_home,
            prob_draw=pr.prob_draw,
            prob_away=pr.prob_away,
            prob_over_25=pr.prob_over_25,
            prob_under_25=pr.prob_under_25,
            expected_home_goals=pr.expected_home_goals,
            expected_away_goals=pr.expected_away_goals,
            top_scores=pr.top_scores,
            prediction=prediction,
            confidence=confidence,
            value_edge=edge,
            is_value_bet=is_value,
            risk_level=risk,
            explanation=explanation,
        )

    def _build_ml_prediction(
        self,
        match: Match,
        pr: PoissonResult,
        feature_vec: np.ndarray,
        features: Dict[str, float],
        odds: Optional[Odds],
    ) -> PredictionResult:
        """ML ağırlıklı tahmin (Poisson doğrulaması ile)."""
        if self.ml_model is None:
            return self._build_poisson_prediction(match, pr, features, odds)

        vec: np.ndarray = np.nan_to_num(feature_vec.reshape(1, -1), nan=0.0)
        ml_probs: np.ndarray = self.ml_model.predict_proba(vec)[0]

        labels: List[str] = ["1", "X", "2"]
        ml_prob_dict: Dict[str, float] = {
            labels[i]: ml_probs[i] * 100 for i in range(len(labels))
        }

        # ML (%70) + Poisson (%30) ensemble
        probs: Dict[str, float] = {
            "1": ml_prob_dict["1"] * 0.70 + pr.prob_home * 0.30,
            "X": ml_prob_dict["X"] * 0.70 + pr.prob_draw * 0.30,
            "2": ml_prob_dict["2"] * 0.70 + pr.prob_away * 0.30,
        }

        prediction: str = max(probs, key=probs.get)  # type: ignore[arg-type]
        confidence: float = probs[prediction]

        edge, is_value = self._calc_value_edge(prediction, probs, odds)
        risk: str = self._determine_risk(confidence, features, prediction)
        explanation: str = self._generate_explanation(
            features, pr, prediction, confidence, "ml",
        )

        return PredictionResult(
            match_id=match.id,
            match_display=match.display_name,
            engine_used="ml",
            model_version=f"ml_{self.MODEL_VERSION}",
            prob_home=probs["1"],
            prob_draw=probs["X"],
            prob_away=probs["2"],
            prob_over_25=pr.prob_over_25,
            prob_under_25=pr.prob_under_25,
            expected_home_goals=pr.expected_home_goals,
            expected_away_goals=pr.expected_away_goals,
            top_scores=pr.top_scores,
            prediction=prediction,
            confidence=confidence,
            value_edge=edge,
            is_value_bet=is_value,
            risk_level=risk,
            explanation=explanation,
        )

    def _build_hybrid_prediction(
        self,
        match: Match,
        pr: PoissonResult,
        feature_vec: np.ndarray,
        features: Dict[str, float],
        odds: Optional[Odds],
    ) -> PredictionResult:
        """Hibrit tahmin: Poisson (%60) + ML (%40)."""
        if self.ml_model is None:
            return self._build_poisson_prediction(match, pr, features, odds)

        vec: np.ndarray = np.nan_to_num(feature_vec.reshape(1, -1), nan=0.0)
        ml_probs: np.ndarray = self.ml_model.predict_proba(vec)[0]

        labels: List[str] = ["1", "X", "2"]
        ml_prob_dict: Dict[str, float] = {
            labels[i]: ml_probs[i] * 100 for i in range(len(labels))
        }

        probs: Dict[str, float] = {
            "1": pr.prob_home * 0.60 + ml_prob_dict["1"] * 0.40,
            "X": pr.prob_draw * 0.60 + ml_prob_dict["X"] * 0.40,
            "2": pr.prob_away * 0.60 + ml_prob_dict["2"] * 0.40,
        }

        prediction: str = max(probs, key=probs.get)  # type: ignore[arg-type]
        confidence: float = probs[prediction]

        edge, is_value = self._calc_value_edge(prediction, probs, odds)
        risk: str = self._determine_risk(confidence, features, prediction)
        explanation: str = self._generate_explanation(
            features, pr, prediction, confidence, "hybrid",
        )

        return PredictionResult(
            match_id=match.id,
            match_display=match.display_name,
            engine_used="hybrid",
            model_version=f"hybrid_{self.MODEL_VERSION}",
            prob_home=probs["1"],
            prob_draw=probs["X"],
            prob_away=probs["2"],
            prob_over_25=pr.prob_over_25,
            prob_under_25=pr.prob_under_25,
            expected_home_goals=pr.expected_home_goals,
            expected_away_goals=pr.expected_away_goals,
            top_scores=pr.top_scores,
            prediction=prediction,
            confidence=confidence,
            value_edge=edge,
            is_value_bet=is_value,
            risk_level=risk,
            explanation=explanation,
        )

    # ─── Yardımcı Fonksiyonlar ────────────────────────────────────

    def _calc_value_edge(
        self,
        prediction: str,
        probs: Dict[str, float],
        odds: Optional[Odds],
    ) -> Tuple[float, bool]:
        """Value edge hesaplar.

        ``edge = model_prob − implied_prob``.
        ``edge ≥ VALUE_BET_MIN_EDGE`` → value bet.
        """
        if not odds:
            return 0.0, False

        odd_map: Dict[str, Optional[float]] = {
            "1": odds.ms_1, "X": odds.ms_x, "2": odds.ms_2,
        }
        odd: Optional[float] = odd_map.get(prediction)

        if not odd or odd <= 1.0:
            return 0.0, False

        implied: float = (1.0 / odd) * 100
        model_prob: float = probs.get(prediction, 0)
        edge: float = model_prob - implied

        return round(edge, 2), edge >= VALUE_BET_MIN_EDGE

    def _determine_risk(
        self,
        confidence: float,
        features: Dict[str, float],
        prediction: str,
    ) -> str:
        """Risk seviyesi belirler.

        Risk faktörleri (v2.1):
          1. Güven seviyesi (confidence < 40/50/55)
          2. H2H veri eksikliği (< 3 maç)
          3. Toplam sakatlık etkisi (> 20)
          4. Kritik eksik oyuncu sayısı (≥ 1 / ≥ 3)
          5. Hakem bias'ı güçlü (|bias| > 15)
          6. Hakem-tahmin çelişkisi (alignment < 25)
          7. Form trend tutarsızlığı (trend ↑ ama tahmin "2" gibi)
          8. Sakatlık normalize skoru aşırı tek taraflı (<25 veya >75)

        Seviyeler:
          ≥ 6 → 🔴 Yüksek Risk
          ≥ 3 → 🟡 Orta Risk
          < 3 → 🟢 Düşük Risk
        """
        risk_score: int = 0

        # 1 — Güven seviyesi
        if confidence < 40:
            risk_score += 3
        elif confidence < 50:
            risk_score += 2
        elif confidence < 55:
            risk_score += 1

        # 2 — H2H verisi eksikliği
        h2h_total: float = features.get("h2h_total", 0)
        if h2h_total < 3:
            risk_score += 1

        # 3 — Toplam sakatlık etkisi
        total_injury: float = features.get("total_injury_importance", 0)
        if total_injury > 20:
            risk_score += 1

        # 4 — Kritik eksik oyuncu sayısı
        critical_total: float = (
            features.get("home_critical_injury_count", 0)
            + features.get("away_critical_injury_count", 0)
        )
        if critical_total >= 3:
            risk_score += 2
        elif critical_total >= 1:
            risk_score += 1

        # 5 — Güçlü hakem bias'ı
        ref_bias: float = features.get("ref_home_bias", 0)
        if abs(ref_bias) > 15:
            risk_score += 1

        # 6 — Hakem-tahmin çelişkisi (v2.1)
        ref_alignment: float = features.get("ref_alignment_score", 50)
        if ref_alignment < 25:
            risk_score += 1

        # 7 — Form trend tutarsızlığı (v2.1)
        form_trend_diff: float = features.get("form_trend_diff", 0)
        if prediction == "1" and form_trend_diff < -0.5:
            # Ev tahmini ama form trendi deplasmanı işaret ediyor
            risk_score += 1
        elif prediction == "2" and form_trend_diff > 0.5:
            risk_score += 1

        # 8 — Sakatlık normalize skoru aşırı tek taraflı (v2.1)
        inj_norm: float = features.get("injury_normalized_score", 50)
        if inj_norm < 25 and prediction == "1":
            risk_score += 1
        elif inj_norm > 75 and prediction == "2":
            risk_score += 1

        # ── Seviye belirleme ──
        if risk_score >= 6:
            return "🔴 Yüksek Risk"
        elif risk_score >= 3:
            return "🟡 Orta Risk"
        return "🟢 Düşük Risk"

    def _generate_explanation(
        self,
        features: Dict[str, float],
        pr: PoissonResult,
        prediction: str,
        confidence: float,
        engine: str,
    ) -> str:
        """İnsan okunabilir açıklama üretir.

        v2.1: value_bet_analyzer.generate_explanation kalitesinde
        dinamik ve zengin açıklamalar. Her feature grubu koşullu
        olarak dahil edilir.
        """
        parts: List[str] = []

        # ── Motor bilgisi ──
        engine_labels: Dict[str, str] = {
            "poisson": "📊 Poisson",
            "hybrid": "🔀 Hibrit (Poisson+ML)",
            "ml": "🤖 ML Model",
        }
        parts.append(f"[{engine_labels.get(engine, engine)}]")

        # ── Poisson beklentileri ──
        parts.append(
            f"⚽ Beklenen: {pr.expected_home_goals:.1f}-{pr.expected_away_goals:.1f}"
        )

        # ── Form + trend ──
        fd: float = features.get("form_diff", 0)
        trend_d: float = features.get("form_trend_diff", 0)
        if fd > 20:
            trend_icon: str = "↑↑" if trend_d > 0.3 else ("↗" if trend_d > 0 else "→")
            parts.append(f"📈 Ev sahibi formda {trend_icon}")
        elif fd < -20:
            trend_icon = "↑↑" if trend_d < -0.3 else ("↗" if trend_d < 0 else "→")
            parts.append(f"📈 Deplasman formda {trend_icon}")

        # ── Lig pozisyonu (v2.1 — value_bet_analyzer kalitesinde) ──
        h_rank: float = features.get("home_rank", 10)
        a_rank: float = features.get("away_rank", 10)
        league_comp: float = features.get("league_position_composite", 50)
        if abs(h_rank - a_rank) > 3:
            lider = "Ev sahibi" if h_rank < a_rank else "Deplasman"
            parts.append(
                f"📈 {lider} ligde üstün "
                f"({int(min(h_rank, a_rank))}. vs {int(max(h_rank, a_rank))}.)"
            )

        # ── Güç composite'i ──
        strength_diff: float = features.get("strength_diff", 0)
        if abs(strength_diff) > 15:
            favori: str = "Ev sahibi" if strength_diff > 0 else "Deplasman"
            parts.append(f"💪 {favori} güçlü (+{abs(strength_diff):.0f})")

        # ── Hakem bias + alignment (v2.1) ──
        ref_bias: float = features.get("ref_home_bias", 0)
        ref_align: float = features.get("ref_alignment_score", 50)
        if abs(ref_bias) > 10:
            bias_label: str = "evci" if ref_bias > 0 else "deplasmanı destekler"
            align_note: str = ""
            if ref_align > 45:
                align_note = " ✓ tahminle uyumlu"
            elif ref_align < 30:
                align_note = " ⚠ tahminle çelişiyor"
            parts.append(f"👨‍⚖️ Hakem {bias_label} ({ref_bias:+.1f}){align_note}")

        # ── Eksik oyuncular + kritik (v2.1 — detaylı) ──
        h_inj: float = features.get("home_injury_penalty", 0)
        a_inj: float = features.get("away_injury_penalty", 0)
        h_crit: int = int(features.get("home_critical_injury_count", 0))
        a_crit: int = int(features.get("away_critical_injury_count", 0))
        inj_norm: float = features.get("injury_normalized_score", 50)

        if h_inj > 5 or h_crit > 0:
            crit_txt: str = f" ({h_crit} kritik)" if h_crit else ""
            parts.append(f"🏥 Ev {h_inj:.0f}p eksik{crit_txt}")
        if a_inj > 5 or a_crit > 0:
            crit_txt = f" ({a_crit} kritik)" if a_crit else ""
            parts.append(f"🏥 Dep {a_inj:.0f}p eksik{crit_txt}")
        if abs(inj_norm - 50) > 15:
            avantaj: str = "ev sahibi" if inj_norm > 50 else "deplasman"
            parts.append(f"🏥 Sağlık avantajı: {avantaj}")

        # ── H2H + trend (v2.1 — detaylı) ──
        h2h_total: float = features.get("h2h_total", 0)
        if h2h_total >= 3:
            h2h_rate: float = features.get("h2h_home_win_rate", 33)
            h2h_trend: float = features.get("h2h_recent_trend", 0)
            h2h_uyum: float = features.get("h2h_tahmin_uyumu", 33.3)
            trend_str: str = ""
            if h2h_trend > 0.3:
                trend_str = " ↑ ev trendi"
            elif h2h_trend < -0.3:
                trend_str = " ↑ dep trendi"
            uyum_str: str = ""
            if h2h_uyum > 50:
                uyum_str = " ✓ tahminle uyumlu"
            parts.append(
                f"📜 H2H: {int(h2h_total)} maç, ev %{h2h_rate:.0f}"
                f"{trend_str}{uyum_str}"
            )

        # ── En olası skor ──
        if pr.top_scores:
            top = pr.top_scores[0]
            parts.append(f"🎯 En olası skor: {top[0]} (%{top[1]:.1f})")

        return " | ".join(parts)

    # ─── Model Doğrulama ─────────────────────────────────────────

    def validate_past_predictions(self) -> Dict[str, object]:
        """Geçmiş tahminlerin doğruluğunu hesaplar."""
        preds: List[Prediction] = (
            self.session.query(Prediction)
            .filter(Prediction.actual_result.isnot(None))
            .all()
        )

        if not preds:
            return {"total": 0, "correct": 0, "accuracy": 0.0}

        correct: int = sum(1 for p in preds if p.is_correct)
        total: int = len(preds)

        by_engine: Dict[str, Dict[str, int | float]] = {}
        for p in preds:
            eng: str = p.engine_used or "unknown"
            if eng not in by_engine:
                by_engine[eng] = {"total": 0, "correct": 0}
            by_engine[eng]["total"] += 1  # type: ignore[operator]
            if p.is_correct:
                by_engine[eng]["correct"] += 1  # type: ignore[operator]

        for eng in by_engine:
            t: int = by_engine[eng]["total"]  # type: ignore[assignment]
            c: int = by_engine[eng]["correct"]  # type: ignore[assignment]
            by_engine[eng]["accuracy"] = (c / t * 100) if t > 0 else 0.0

        return {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total * 100) if total > 0 else 0.0,
            "by_engine": by_engine,
        }
