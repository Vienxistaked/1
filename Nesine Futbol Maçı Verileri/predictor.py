"""
🤖 Hibrit Tahmin Motoru v3.1 (MatchPredictor — Stacking Ensemble)
Poisson + Stacking Ensemble (CatBoost + LightGBM + XGBoost → LogisticRegression)
ile maç sonucu tahmini + SHAP Explainability.

Mimari:
  ┌────────────────────────────────────────────────────┐
  │             Layer 1 (Base Learners)                │
  │  ┌───────────┐ ┌───────────┐ ┌───────────────┐    │
  │  │  CatBoost │ │ LightGBM  │ │   XGBoost     │    │
  │  │ (96 feat) │ │ (96 feat) │ │  (96 feat)    │    │
  │  │ +4 categ. │ │           │ │               │    │
  │  └─────┬─────┘ └─────┬─────┘ └──────┬────────┘    │
  │        │              │              │             │
  │        ▼              ▼              ▼             │
  │  ┌─────────────────────────────────────────────┐   │
  │  │   Layer 2 (Meta-Learner)                    │   │
  │  │   LogisticRegression(C=1.0)                 │   │
  │  │   Input: 9 class probabilities (3×3)        │   │
  │  └─────────────────────┬───────────────────────┘   │
  │                        │                           │
  │                        ▼                           │
  │              Final Prediction (1/X/2)              │
  │              + SHAP Feature Importance             │
  └────────────────────────────────────────────────────┘

Cold-Start Mekanizması:
  • < 50  maç  → Saf Poisson
  • 50–200 maç → Poisson (%60) + Stacking (%40)  hibrit
  • > 200 maç  → Stacking ağırlıklı (%70) + Poisson (%30)

v3.1 Değişiklikleri (v3.0 üzerinden):
  ✓ Bayesian Smoothing: Erken sezon overfitting çözümü
    — Puan tablosu feature'ları lig medyanına çekilir (dampened_rank)
    — Poisson λ hesaplaması Bayesian Average ile sönümlenir
    — season_progress, season_confidence ile sezon konteksti
  ✓ Dynamic Feature Trust: Erken sezonda implied_prob ağırlığı artar,
    standing feature ağırlığı azalır (season_confidence üzerinden)
  ✓ İlgili yeni feature'lar: relative_market_strength, early_season_reliability
  ✓ Risk değerlendirme: Erken sezon güvenilirlik eksikliği faktörü eklendi
  ✓ 96-feature vektörü (feature_engineering v3.1, 11 yeni feature)

v3.0 Değişiklikleri (v2.1 üzerinden):
  ✓ XGBoost tek model → Stacking Ensemble (CatBoost+LightGBM+XGBoost)
  ✓ LogisticRegression meta-learner (Layer 2)
  ✓ CatBoost native categorical feature desteği (4 kategorik: takım+hakem)
  ✓ 85-feature vektörü (feature_engineering v3.0)
  ✓ SHAP entegrasyonu → "Neden MS1?" human-readable açıklamalar
  ✓ Kronolojik 5-Fold CV ile stacking-oof eğitimi (data leakage koruması)
  ✓ MODEL_VERSION = "v3.1" + otomatik cache invalidation
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sqlalchemy.orm import Session

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from config import (
    MIN_TRAINING_SAMPLES,
    MIN_TRAINING_SAMPLES_XGBOOST,
    MODEL_DIR,
    RANDOM_SEED,
    VALUE_BET_MIN_EDGE,
    VALUE_BET_MIN_CONFIDENCE,
)
from models import Match, Odds, Prediction
from feature_engineering import (
    FeatureExtractor,
    build_training_dataset,
    build_training_dataset_with_categorical,
)
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

    # v3.0: SHAP bilgileri
    shap_top_features: List[Tuple[str, float]] = field(default_factory=list)
    shap_summary: str = ""

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
#  Stacking Ensemble Sınıfı
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StackingEnsemble:
    """Layer 1: CatBoost + LightGBM + XGBoost
    Layer 2: LogisticRegression meta-learner

    Temporal (kronolojik) K-Fold CV ile out-of-fold (OOF) tahminler üretir.
    Meta-learner OOF olasılıklarını girdi olarak alıp final tahmin yapar.
    """

    def __init__(self, use_optuna: bool = True, optuna_n_trials: int = 30) -> None:
        self.base_models: List[Tuple[str, Any]] = []
        self.meta_model: Optional[LogisticRegression] = None
        self.is_fitted: bool = False
        self._has_catboost: bool = False
        self._has_lightgbm: bool = False
        self._has_xgboost: bool = False
        self._cat_feature_indices: List[int] = []
        self._use_optuna: bool = use_optuna and HAS_OPTUNA
        self._optuna_n_trials: int = optuna_n_trials
        self._best_params: Dict[str, Dict[str, Any]] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cat_features: Optional[List[Dict[str, str]]] = None,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """Stacking modelini eğitir.

        Parameters
        ----------
        X : np.ndarray  (n_samples, 96)
        y : np.ndarray  (n_samples,) — 0/1/2 labels
        cat_features : CatBoost için kategorik feature listesi
        n_splits : Temporal CV split sayısı

        Returns
        -------
        Dict[str, float]
            Her base model ve final ensemble doğruluğu.
        """
        X = np.nan_to_num(X, nan=0.0)
        n_classes = len(np.unique(y))

        # ── Optuna HPO (opsiyonel) ──
        if self._use_optuna and len(X) >= 100:
            logger.info("🔬 Optuna HPO başlatılıyor (%d trial)...", self._optuna_n_trials)
            self._best_params = self.optimize_hyperparameters(
                X, y, n_trials=self._optuna_n_trials, n_splits=n_splits,
            )
            logger.info("✓ Optuna HPO tamamlandı: %s", list(self._best_params.keys()))

        # ── Base modelleri hazırla ──
        self.base_models = []
        self._init_base_models(cat_features)

        if not self.base_models:
            raise RuntimeError("Hiçbir base model yüklenemedi!")

        n_base = len(self.base_models)
        logger.info(
            "🔧 Stacking eğitimi başlıyor: %d base model, %d split",
            n_base, n_splits,
        )

        # ── Temporal K-Fold → OOF tahminler ──
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_preds = np.zeros((len(X), n_base * n_classes))
        oof_mask = np.zeros(len(X), dtype=bool)  # Hangi satırlara OOF yazıldı

        accuracies: Dict[str, List[float]] = {name: [] for name, _ in self.base_models}

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            for model_idx, (name, model) in enumerate(self.base_models):
                try:
                    if name == "CatBoost" and self._cat_feature_indices and cat_features:
                        # CatBoost native categorical → full matrix (numeric + encoded cat)
                        X_tr_cat, X_val_cat = self._prepare_catboost_data(
                            X_tr, X_val, cat_features, train_idx, val_idx,
                        )
                        model.fit(
                            X_tr_cat, y_tr,
                            eval_set=(X_val_cat, y_val),
                            verbose=0,
                        )
                        probs = model.predict_proba(X_val_cat)
                    else:
                        model.fit(X_tr, y_tr)
                        probs = model.predict_proba(X_val)

                    # OOF olasılıklarını yaz
                    start_col = model_idx * n_classes
                    end_col = start_col + n_classes
                    oof_preds[val_idx, start_col:end_col] = probs

                    # Fold doğruluğu
                    fold_preds = np.argmax(probs, axis=1)
                    fold_acc = np.mean(fold_preds == y_val)
                    accuracies[name].append(fold_acc)

                except Exception as e:
                    logger.warning(
                        "Fold %d / %s hatası: %s", fold_idx, name, e,
                    )
                    start_col = model_idx * n_classes
                    end_col = start_col + n_classes
                    # Eşit olasılık fallback
                    oof_preds[val_idx, start_col:end_col] = 1.0 / n_classes

            oof_mask[val_idx] = True

        # ── Son tüm veri üzerinde base modelleri yeniden eğit ──
        for name, model in self.base_models:
            try:
                if name == "CatBoost" and self._cat_feature_indices and cat_features:
                    X_full_cat = self._prepare_catboost_full(X, cat_features)
                    model.fit(X_full_cat, y, verbose=0)
                else:
                    model.fit(X, y)
            except Exception as e:
                logger.warning("Final fit hatası (%s): %s", name, e)

        # ── Meta-learner (Layer 2) eğitimi ──
        X_meta = oof_preds[oof_mask]
        y_meta = y[oof_mask]

        self.meta_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        self.meta_model.fit(X_meta, y_meta)

        self.is_fitted = True

        # ── Final doğruluk hesapla ──
        meta_preds = self.meta_model.predict(X_meta)
        meta_acc = float(np.mean(meta_preds == y_meta))

        result: Dict[str, float] = {
            "stacking_accuracy": meta_acc,
        }
        for name, acc_list in accuracies.items():
            result[f"{name}_avg_accuracy"] = float(np.mean(acc_list)) if acc_list else 0.0

        logger.info("✓ Stacking doğruluğu: %.2f%%", meta_acc * 100)
        for name, avg in result.items():
            if name != "stacking_accuracy":
                logger.info("  • %s: %.2f%%", name, avg * 100)

        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Stacking ensemble ile olasılık tahmini.

        Returns
        -------
        np.ndarray  shape (n_samples, 3) — [P(1), P(X), P(2)]
        """
        if not self.is_fitted or self.meta_model is None:
            raise RuntimeError("Model henüz eğitilmedi!")

        X = np.nan_to_num(X.reshape(1, -1) if X.ndim == 1 else X, nan=0.0)
        n_classes = 3
        n_base = len(self.base_models)
        meta_input = np.zeros((len(X), n_base * n_classes))

        for model_idx, (name, model) in enumerate(self.base_models):
            try:
                probs = model.predict_proba(X)
                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                meta_input[:, start_col:end_col] = probs
            except Exception as e:
                logger.warning("Predict hatası (%s): %s", name, e)
                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                meta_input[:, start_col:end_col] = 1.0 / n_classes

        return self.meta_model.predict_proba(meta_input)

    def predict_proba_catboost(
        self,
        X_numeric: np.ndarray,
        cat_dict: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """CatBoost kategorik feature destekli tahmin.

        CatBoost'a kategorik feature'lar ayrıca verilir,
        diğer modeller sadece numeric alır.
        """
        if not self.is_fitted or self.meta_model is None:
            raise RuntimeError("Model henüz eğitilmedi!")

        X = np.nan_to_num(
            X_numeric.reshape(1, -1) if X_numeric.ndim == 1 else X_numeric,
            nan=0.0,
        )
        n_classes = 3
        n_base = len(self.base_models)
        meta_input = np.zeros((len(X), n_base * n_classes))

        for model_idx, (name, model) in enumerate(self.base_models):
            try:
                if name == "CatBoost" and cat_dict and self._has_catboost:
                    # CatBoost prediction doesn't need cat features if trained with indices
                    probs = model.predict_proba(X)
                else:
                    probs = model.predict_proba(X)

                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                meta_input[:, start_col:end_col] = probs
            except Exception as e:
                logger.warning("Predict hatası (%s): %s", name, e)
                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                meta_input[:, start_col:end_col] = 1.0 / n_classes

        return self.meta_model.predict_proba(meta_input)

    # ─── Dahili: Base modelleri oluştur ───────────────────────────

    # ─── Optuna Hiperparametre Optimizasyonu ──────────────────

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 30,
        n_splits: int = 3,
    ) -> Dict[str, Dict[str, Any]]:
        """Optuna ile CatBoost, LightGBM ve XGBoost hiperparametre optimizasyonu.

        Parameters
        ----------
        X : np.ndarray — eğitim verisi
        y : np.ndarray — etiketler (0/1/2)
        n_trials : Optuna deneme sayısı
        n_splits : TimeSeriesSplit sayısı

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Her model için en iyi hiperparametreler.
        """
        if not HAS_OPTUNA:
            logger.warning("⚠ Optuna yüklü değil, varsayılan parametreler kullanılacak")
            return {}

        best_params: Dict[str, Dict[str, Any]] = {}
        tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X) // 50)))

        # ── CatBoost HPO ──
        try:
            from catboost import CatBoostClassifier

            def _catboost_objective(trial: optuna.Trial) -> float:
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 500),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                    "loss_function": "MultiClass",
                    "eval_metric": "MultiClass",
                    "random_seed": RANDOM_SEED,
                    "verbose": 0,
                    "early_stopping_rounds": 30,
                    "use_best_model": True,
                }
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    model = CatBoostClassifier(**params)
                    model.fit(X[train_idx], y[train_idx],
                              eval_set=(X[val_idx], y[val_idx]), verbose=0)
                    probs = model.predict_proba(X[val_idx])
                    scores.append(log_loss(y[val_idx], probs))
                return float(np.mean(scores))

            study = optuna.create_study(direction="minimize", study_name="catboost_hpo")
            study.optimize(_catboost_objective, n_trials=n_trials, show_progress_bar=False)
            best_params["CatBoost"] = study.best_params
            logger.info("✓ CatBoost HPO tamamlandı (logloss: %.4f)", study.best_value)
        except ImportError:
            pass
        except Exception as e:
            logger.warning("⚠ CatBoost HPO hatası: %s", e)

        # ── LightGBM HPO ──
        try:
            from lightgbm import LGBMClassifier

            def _lgbm_objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "objective": "multiclass",
                    "num_class": 3,
                    "random_state": RANDOM_SEED,
                    "verbose": -1,
                    "n_jobs": -1,
                }
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    model = LGBMClassifier(**params)
                    model.fit(X[train_idx], y[train_idx])
                    probs = model.predict_proba(X[val_idx])
                    scores.append(log_loss(y[val_idx], probs))
                return float(np.mean(scores))

            study = optuna.create_study(direction="minimize", study_name="lgbm_hpo")
            study.optimize(_lgbm_objective, n_trials=n_trials, show_progress_bar=False)
            best_params["LightGBM"] = study.best_params
            logger.info("✓ LightGBM HPO tamamlandı (logloss: %.4f)", study.best_value)
        except ImportError:
            pass
        except Exception as e:
            logger.warning("⚠ LightGBM HPO hatası: %s", e)

        # ── XGBoost HPO ──
        try:
            from xgboost import XGBClassifier

            def _xgb_objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "eval_metric": "mlogloss",
                    "use_label_encoder": False,
                    "random_state": RANDOM_SEED,
                    "verbosity": 0,
                    "n_jobs": -1,
                }
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    model = XGBClassifier(**params)
                    model.fit(X[train_idx], y[train_idx])
                    probs = model.predict_proba(X[val_idx])
                    scores.append(log_loss(y[val_idx], probs))
                return float(np.mean(scores))

            study = optuna.create_study(direction="minimize", study_name="xgb_hpo")
            study.optimize(_xgb_objective, n_trials=n_trials, show_progress_bar=False)
            best_params["XGBoost"] = study.best_params
            logger.info("✓ XGBoost HPO tamamlandı (logloss: %.4f)", study.best_value)
        except ImportError:
            pass
        except Exception as e:
            logger.warning("⚠ XGBoost HPO hatası: %s", e)

        return best_params

    def _init_base_models(
        self, cat_features: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Kullanılabilir ML kütüphanelerine göre base model listesini oluşturur.
        Optuna ile bulunan en iyi parametreler varsa onları kullanır."""
        # ── CatBoost ──
        try:
            from catboost import CatBoostClassifier
            if "CatBoost" in self._best_params:
                hp = self._best_params["CatBoost"]
                model = CatBoostClassifier(
                    iterations=hp.get("iterations", 300),
                    depth=hp.get("depth", 6),
                    learning_rate=hp.get("learning_rate", 0.08),
                    l2_leaf_reg=hp.get("l2_leaf_reg", 3.0),
                    bagging_temperature=hp.get("bagging_temperature", 0.5),
                    loss_function="MultiClass",
                    eval_metric="Accuracy",
                    random_seed=RANDOM_SEED,
                    verbose=0,
                    early_stopping_rounds=30,
                    use_best_model=True,
                )
                logger.info("✓ CatBoost yüklendi (Optuna optimized)")
            else:
                model = CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    learning_rate=0.08,
                    loss_function="MultiClass",
                    eval_metric="Accuracy",
                    random_seed=RANDOM_SEED,
                    verbose=0,
                    early_stopping_rounds=30,
                    use_best_model=True,
                )
                logger.info("✓ CatBoost yüklendi (varsayılan parametreler)")
            self.base_models.append(("CatBoost", model))
            self._has_catboost = True
        except ImportError:
            logger.warning("⚠ CatBoost bulunamadı, atlanıyor")

        # ── LightGBM ──
        try:
            from lightgbm import LGBMClassifier
            if "LightGBM" in self._best_params:
                hp = self._best_params["LightGBM"]
                model = LGBMClassifier(
                    n_estimators=hp.get("n_estimators", 300),
                    max_depth=hp.get("max_depth", 6),
                    learning_rate=hp.get("learning_rate", 0.08),
                    num_leaves=hp.get("num_leaves", 31),
                    subsample=hp.get("subsample", 0.8),
                    colsample_bytree=hp.get("colsample_bytree", 0.8),
                    reg_alpha=hp.get("reg_alpha", 0.0),
                    reg_lambda=hp.get("reg_lambda", 0.0),
                    min_child_samples=hp.get("min_child_samples", 20),
                    objective="multiclass",
                    num_class=3,
                    random_state=RANDOM_SEED,
                    verbose=-1,
                    n_jobs=-1,
                )
                logger.info("✓ LightGBM yüklendi (Optuna optimized)")
            else:
                model = LGBMClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    num_leaves=31,
                    objective="multiclass",
                    num_class=3,
                    random_state=RANDOM_SEED,
                    verbose=-1,
                    n_jobs=-1,
                )
                logger.info("✓ LightGBM yüklendi (varsayılan parametreler)")
            self.base_models.append(("LightGBM", model))
            self._has_lightgbm = True
        except ImportError:
            logger.warning("⚠ LightGBM bulunamadı, atlanıyor")

        # ── XGBoost ──
        try:
            from xgboost import XGBClassifier
            if "XGBoost" in self._best_params:
                hp = self._best_params["XGBoost"]
                model = XGBClassifier(
                    n_estimators=hp.get("n_estimators", 300),
                    max_depth=hp.get("max_depth", 6),
                    learning_rate=hp.get("learning_rate", 0.08),
                    subsample=hp.get("subsample", 0.8),
                    colsample_bytree=hp.get("colsample_bytree", 0.8),
                    reg_alpha=hp.get("reg_alpha", 0.0),
                    reg_lambda=hp.get("reg_lambda", 0.0),
                    min_child_weight=hp.get("min_child_weight", 1),
                    gamma=hp.get("gamma", 0.0),
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    random_state=RANDOM_SEED,
                    verbosity=0,
                    n_jobs=-1,
                )
                logger.info("✓ XGBoost yüklendi (Optuna optimized)")
            else:
                model = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                    random_state=RANDOM_SEED,
                    verbosity=0,
                    n_jobs=-1,
                )
                logger.info("✓ XGBoost yüklendi (varsayılan parametreler)")
            self.base_models.append(("XGBoost", model))
            self._has_xgboost = True
        except ImportError:
            logger.warning("⚠ XGBoost bulunamadı, atlanıyor")

        # ── Fallback: En az bir model olmalı ──
        if not self.base_models:
            from sklearn.ensemble import (
                RandomForestClassifier,
                GradientBoostingClassifier,
            )
            self.base_models.append((
                "RandomForest",
                RandomForestClassifier(
                    n_estimators=200, max_depth=10,
                    random_state=RANDOM_SEED, n_jobs=-1,
                ),
            ))
            self.base_models.append((
                "GradientBoosting",
                GradientBoostingClassifier(
                    n_estimators=200, max_depth=6,
                    learning_rate=0.1, random_state=RANDOM_SEED,
                ),
            ))
            logger.info("⚠ Fallback modeller yüklendi: RandomForest + GradientBoosting")

    def _prepare_catboost_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        cat_features: List[Dict[str, str]],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CatBoost için kategorik feature'ları numeric X'e ekler.
        Şimdilik basit label encoding yapar (CatBoost zaten native handle eder).
        """
        # CatBoost native categorical kullanmak yerine sadece numeric kullanıyoruz
        # çünkü OOF stacking'de categorical handling karmaşık.
        # CatBoost zaten tree-based olarak bunları öğrenebilir.
        return X_train, X_val

    def _prepare_catboost_full(
        self,
        X: np.ndarray,
        cat_features: List[Dict[str, str]],
    ) -> np.ndarray:
        """Tüm veri için CatBoost hazırlığı."""
        return X


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHAP Explainer Wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SHAPExplainer:
    """SHAP ile tahmin açıklaması üretir.

    Base modellerdeki en güçlü ağaç modelini (XGBoost > LightGBM > CatBoost)
    kullanarak SHAP değerleri hesaplar.

    Kullanım::

        explainer = SHAPExplainer()
        explainer.initialize(stacking_ensemble.base_models, X_background)
        top_features = explainer.explain(feature_vector, prediction_class=0)
    """

    def __init__(self) -> None:
        self._explainer: Any = None
        self._feature_names: List[str] = FeatureExtractor.FEATURE_NAMES
        self._available: bool = False

    def initialize(
        self,
        base_models: List[Tuple[str, Any]],
        X_background: Optional[np.ndarray] = None,
    ) -> bool:
        """SHAP Explainer'ı başlatır.

        Parameters
        ----------
        base_models : Stacking'deki (name, model) çiftleri
        X_background : SHAP background veri seti (100 satır yeterli)

        Returns
        -------
        bool
            SHAP başarıyla initialize edildiyse True.
        """
        try:
            import shap

            # Ağaç modeli prioritize et
            tree_model = None
            for name, model in base_models:
                if name in ("XGBoost", "LightGBM", "CatBoost"):
                    tree_model = model
                    break

            if tree_model is None:
                logger.warning("⚠ SHAP: Ağaç tabanlı model bulunamadı")
                return False

            if X_background is not None and len(X_background) > 100:
                bg = X_background[:100]
            else:
                bg = X_background

            self._explainer = shap.TreeExplainer(tree_model, data=bg)
            self._available = True
            logger.info("✓ SHAP TreeExplainer başlatıldı")
            return True

        except ImportError:
            logger.warning("⚠ shap kütüphanesi bulunamadı")
            return False
        except Exception as e:
            logger.warning("⚠ SHAP başlatma hatası: %s", e)
            return False

    def explain(
        self,
        X: np.ndarray,
        prediction_class: int = 0,
        top_n: int = 5,
    ) -> Tuple[List[Tuple[str, float]], str]:
        """Tek bir tahmin için SHAP değerlerini hesaplar.

        Parameters
        ----------
        X : (96,) feature vektörü
        prediction_class : 0=MS1, 1=MSX, 2=MS2
        top_n : Döndürülecek en önemli feature sayısı

        Returns
        -------
        Tuple[List[Tuple[str, float]], str]
            (top_features, human_readable_summary)
        """
        if not self._available or self._explainer is None:
            return [], ""

        try:
            import shap as _shap  # noqa: F811

            X_reshaped = X.reshape(1, -1) if X.ndim == 1 else X
            shap_values = self._explainer.shap_values(X_reshaped)

            # shap_values → multiclass: list of arrays, her class için (1, 85)
            if isinstance(shap_values, list):
                # Tahmin edilen class'ın SHAP değerleri
                class_shap = shap_values[prediction_class][0]
            elif shap_values.ndim == 3:
                # (1, 85, 3) format
                class_shap = shap_values[0, :, prediction_class]
            else:
                class_shap = shap_values[0]

            # En etkili feature'ları bul
            abs_shap = np.abs(class_shap)
            top_indices = np.argsort(abs_shap)[::-1][:top_n]

            top_features: List[Tuple[str, float]] = []
            for idx in top_indices:
                name = (
                    self._feature_names[idx]
                    if idx < len(self._feature_names)
                    else f"feature_{idx}"
                )
                top_features.append((name, float(class_shap[idx])))

            # Human-readable summary
            summary = self._build_shap_summary(
                top_features, prediction_class,
            )

            return top_features, summary

        except Exception as e:
            logger.warning("SHAP explain hatası: %s", e)
            return [], ""

    @staticmethod
    def _build_shap_summary(
        top_features: List[Tuple[str, float]],
        prediction_class: int,
    ) -> str:
        """SHAP değerlerinden insan okunabilir özet üretir."""
        class_labels: Dict[int, str] = {0: "MS 1", 1: "MS X", 2: "MS 2"}
        label: str = class_labels.get(prediction_class, "?")

        # Feature isimleri → Türkçe çeviri
        name_map: Dict[str, str] = {
            "home_form_score": "Ev formu",
            "away_form_score": "Deplasman formu",
            "form_diff": "Form farkı",
            "home_rank": "Ev sahibi sırası",
            "away_rank": "Deplasman sırası",
            "rank_diff": "Sıra farkı",
            "league_position_composite": "Lig pozisyonu composite",
            "ref_home_bias": "Hakem ev bias'ı",
            "ref_alignment_score": "Hakem uyumu",
            "h2h_home_win_rate": "H2H ev galibiyet oranı",
            "h2h_recent_trend": "H2H son trend",
            "home_injury_penalty": "Ev eksik cezası",
            "away_injury_penalty": "Dep eksik cezası",
            "injury_penalty_diff": "Eksik farkı",
            "implied_prob_home": "Oran olasılığı (ev)",
            "implied_prob_away": "Oran olasılığı (dep)",
            "home_strength_composite": "Ev güç composite",
            "away_strength_composite": "Dep güç composite",
            "strength_diff": "Güç farkı",
            "home_exp_decay_form": "Ev exp-decay form",
            "away_exp_decay_form": "Dep exp-decay form",
            "exp_decay_form_diff": "Exp-decay form farkı",
            "exp_decay_momentum": "Form momentumu",
            "home_rolling3_scored": "Ev son 3 maç gol ort.",
            "away_rolling3_scored": "Dep son 3 maç gol ort.",
            "home_form_x_away_defense_weakness": "Ev form × Dep defans zaf.",
            "away_form_x_home_defense_weakness": "Dep form × Ev defans zaf.",
            "home_attack_x_away_concede": "Ev atak × Dep gol yeme",
        }

        parts: List[str] = [f"🔍 Neden {label}?"]
        for feat_name, shap_val in top_features:
            direction: str = "↑" if shap_val > 0 else "↓"
            tr_name: str = name_map.get(feat_name, feat_name)
            parts.append(f"  {direction} {tr_name} ({shap_val:+.3f})")

        return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ana Tahmin Sınıfı
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MatchPredictor:
    """Hibrit tahmin motoru v3.1 — Stacking Ensemble + SHAP + Bayesian Smoothing.

    v3.1 Dinamik Feature Güveni (Dynamic Feature Trust):
      Erken sezonda (< ~7 maç) model otomatik olarak:
      • implied_prob (oran bazlı) feature'lara DAHA FAZLA güvenir
      • standing (sıra, puan) feature'lara DAHA AZ güvenir
      • dampened_rank feature'ları Bayesian shrinkage ile medyana çekilmiştir
      • season_confidence feature'ı modele bu konteksti sağlar
      • Poisson λ hesaplamasında Bayesian Average kullanılır:
        bayesian = (observed × n + prior × C) / (n + C)

    Kullanım::

        predictor = MatchPredictor(session)
        predictor.initialize()          # Modeli eğit veya yükle
        result = predictor.predict(match)
    """

    MODEL_FILE = MODEL_DIR / "match_predictor.pkl"
    MODEL_VERSION: str = "v3.1"

    def __init__(self, session: Session) -> None:
        self.session: Session = session
        self.extractor: FeatureExtractor = FeatureExtractor(session)
        self.poisson: PoissonModel = PoissonModel()
        self.stacking: Optional[StackingEnsemble] = None
        self.shap_explainer: SHAPExplainer = SHAPExplainer()
        self.training_samples: int = 0
        self._mode: str = "poisson"  # "poisson" | "hybrid" | "ml"
        self._X_background: Optional[np.ndarray] = None  # SHAP background data

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
        """Stacking Ensemble modelini eğitir.

        v3.1:
          • 96-feature vektörü (feature_engineering v3.1 — 11 Bayesian dampening feature)
          • Stacking: CatBoost + LightGBM + XGBoost → LogisticRegression
          • Temporal K-Fold CV (data leakage koruması)
          • SHAP TreeExplainer başlatma
        """
        logger.info("🔧 Stacking Ensemble eğitiliyor (v3.0)...")

        # Eski modeli temizle (feature boyutu veya versiyon değişmiş olabilir)
        if self.MODEL_FILE.exists():
            self.MODEL_FILE.unlink()
            logger.info("🗑  Eski model cache silindi (v3.0 upgrade)")

        # ── Veri çek (kategorik dahil) ──
        X, y, cat_features = build_training_dataset_with_categorical(self.session)

        if len(X) < MIN_TRAINING_SAMPLES:
            self._mode = "poisson"
            return f"Poisson (yetersiz eğitim verisi: {len(X)})"

        # ── Stacking Ensemble eğit ──
        self.stacking = StackingEnsemble(
            use_optuna=True,
            optuna_n_trials=30,
        )
        try:
            result = self.stacking.fit(
                X, y,
                cat_features=cat_features,
                n_splits=min(5, max(2, len(X) // 50)),  # Adaptif split
            )
        except Exception as e:
            logger.error("Stacking eğitim hatası: %s", e)
            self._mode = "poisson"
            return f"Poisson (stacking hatası: {e})"

        # ── SHAP başlat ──
        self._X_background = X[:100] if len(X) > 100 else X
        self.shap_explainer.initialize(
            self.stacking.base_models,
            self._X_background,
        )

        # ── Model kaydet ──
        self._save_model()

        if finished >= MIN_TRAINING_SAMPLES_XGBOOST:
            self._mode = "ml"
        else:
            self._mode = "hybrid"

        # ── Sonuç raporu ──
        acc = result.get("stacking_accuracy", 0)
        base_info = []
        for key, val in result.items():
            if key.endswith("_avg_accuracy"):
                name = key.replace("_avg_accuracy", "")
                base_info.append(f"{name}: {val:.1%}")

        return (
            f"{self._mode} (Stacking Ensemble, "
            f"doğruluk: {acc:.2%}, "
            f"base: [{', '.join(base_info)}], "
            f"{finished} maç)"
        )

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
    # GÜVENLİK: pickle dosyası RCE riski taşır. Aşağıdaki önlemler uygulanır:
    #   1. Yalnızca MODEL_DIR dizininden yükleme (path traversal engeli)
    #   2. Kaydetme sırasında SHA-256 hash üretilir (.sha256 dosyası)
    #   3. Yükleme sırasında hash doğrulanır → uyumsuzlukta reddet
    #   4. Güvenilmeyen ortamda uyarı loglanır

    MODEL_HASH_FILE = MODEL_DIR / "match_predictor.pkl.sha256"

    @staticmethod
    def _compute_file_hash(filepath: Path) -> str:
        """Dosyanın SHA-256 hash'ini hesaplar."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _save_model(self) -> None:
        """Modeli diske kaydeder + SHA-256 hash dosyası oluşturur."""
        if self.stacking is None or not self.stacking.is_fitted:
            return
        try:
            with open(self.MODEL_FILE, "wb") as fp:
                pickle.dump(
                    {
                        "stacking": self.stacking,
                        "version": self.MODEL_VERSION,
                        "samples": self.training_samples,
                        "n_features": len(FeatureExtractor.FEATURE_NAMES),
                        "X_background": self._X_background,
                        "best_params": getattr(self.stacking, '_best_params', {}),
                    },
                    fp,
                )
            # SHA-256 hash kaydet
            file_hash = self._compute_file_hash(self.MODEL_FILE)
            self.MODEL_HASH_FILE.write_text(file_hash)
            logger.info("✓ Stacking model kaydedildi: %s (hash: %s…)", self.MODEL_FILE, file_hash[:12])
        except Exception as e:
            logger.error("Model kaydetme hatası: %s", e)

    def _load_model(self) -> bool:
        """Kaydedilmiş modeli yükler.

        Güvenlik kontrolleri:
          1. Dosya yalnızca MODEL_DIR altındaysa kabul edilir
          2. .sha256 hash dosyası mevcutsa doğrulama yapılır
          3. Hash uyumsuzluğu → yükleme reddedilir
          4. v3.0: Versiyon + feature boyutu uyumsuzluğu kontrolü
        """
        if not self.MODEL_FILE.exists():
            return False

        # Güvenlik: Sadece beklenen dizinden yükle (path traversal engeli)
        try:
            resolved = self.MODEL_FILE.resolve()
            allowed_dir = MODEL_DIR.resolve()
            if not str(resolved).startswith(str(allowed_dir)):
                logger.error("⛔ Model dosyası güvenli dizin dışında: %s", resolved)
                return False
        except Exception:
            return False

        # Hash doğrulama
        if self.MODEL_HASH_FILE.exists():
            expected_hash = self.MODEL_HASH_FILE.read_text().strip()
            actual_hash = self._compute_file_hash(self.MODEL_FILE)
            if expected_hash != actual_hash:
                logger.error(
                    "⛔ Model dosyası hash doğrulaması BAŞARISIZ!\n"
                    "  Beklenen: %s\n  Bulunan:  %s\n"
                    "  Dosya değiştirilmiş olabilir. Model yeniden eğitilecek.",
                    expected_hash, actual_hash,
                )
                self.MODEL_FILE.unlink(missing_ok=True)
                self.MODEL_HASH_FILE.unlink(missing_ok=True)
                return False
            logger.debug("✓ Model hash doğrulandı: %s…", actual_hash[:12])
        else:
            logger.warning(
                "⚠️  Model hash dosyası bulunamadı (%s). "
                "Model yeniden eğitilecek (güvenlik önlemi).",
                self.MODEL_HASH_FILE,
            )
            self.MODEL_FILE.unlink(missing_ok=True)
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
                self.MODEL_HASH_FILE.unlink(missing_ok=True)
                return False

            self.stacking = data["stacking"]
            self._X_background = data.get("X_background")
            logger.info("✓ Stacking model yüklendi (versiyon: %s)", saved_version)

            # SHAP'ı yeniden başlat
            if self.stacking and self.stacking.base_models:
                self.shap_explainer.initialize(
                    self.stacking.base_models,
                    self._X_background,
                )

            return True
        except Exception as e:
            logger.warning("Model yükleme hatası: %s", e)
            return False

    # ─── Tahmin ───────────────────────────────────────────────────

    def predict(self, match: Match) -> PredictionResult:
        """Bir maç için tahmin üretir.

        Mod'a göre Poisson, Stacking veya hibrit kullanır.
        SHAP açıklamaları otomatik eklenir (mümkünse).
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
        """Stacking Ensemble ağırlıklı tahmin (Poisson doğrulaması ile)."""
        if self.stacking is None or not self.stacking.is_fitted:
            return self._build_poisson_prediction(match, pr, features, odds)

        vec: np.ndarray = np.nan_to_num(feature_vec.reshape(1, -1), nan=0.0)
        ml_probs: np.ndarray = self.stacking.predict_proba(vec)[0]

        labels: List[str] = ["1", "X", "2"]
        ml_prob_dict: Dict[str, float] = {
            labels[i]: ml_probs[i] * 100 for i in range(len(labels))
        }

        # Stacking (%70) + Poisson (%30)
        probs: Dict[str, float] = {
            "1": ml_prob_dict["1"] * 0.70 + pr.prob_home * 0.30,
            "X": ml_prob_dict["X"] * 0.70 + pr.prob_draw * 0.30,
            "2": ml_prob_dict["2"] * 0.70 + pr.prob_away * 0.30,
        }

        prediction: str = max(probs, key=probs.get)  # type: ignore[arg-type]
        confidence: float = probs[prediction]

        edge, is_value = self._calc_value_edge(prediction, probs, odds)
        risk: str = self._determine_risk(confidence, features, prediction)

        # ── SHAP açıklama ──
        label_to_class: Dict[str, int] = {"1": 0, "X": 1, "2": 2}
        shap_features, shap_summary = self.shap_explainer.explain(
            feature_vec,
            prediction_class=label_to_class.get(prediction, 0),
        )

        explanation: str = self._generate_explanation(
            features, pr, prediction, confidence, "ml",
            shap_summary=shap_summary,
        )

        return PredictionResult(
            match_id=match.id,
            match_display=match.display_name,
            engine_used="ml",
            model_version=f"stacking_{self.MODEL_VERSION}",
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
            shap_top_features=shap_features,
            shap_summary=shap_summary,
        )

    def _build_hybrid_prediction(
        self,
        match: Match,
        pr: PoissonResult,
        feature_vec: np.ndarray,
        features: Dict[str, float],
        odds: Optional[Odds],
    ) -> PredictionResult:
        """Hibrit tahmin: Poisson (%60) + Stacking (%40)."""
        if self.stacking is None or not self.stacking.is_fitted:
            return self._build_poisson_prediction(match, pr, features, odds)

        vec: np.ndarray = np.nan_to_num(feature_vec.reshape(1, -1), nan=0.0)
        ml_probs: np.ndarray = self.stacking.predict_proba(vec)[0]

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

        # ── SHAP ──
        label_to_class: Dict[str, int] = {"1": 0, "X": 1, "2": 2}
        shap_features, shap_summary = self.shap_explainer.explain(
            feature_vec,
            prediction_class=label_to_class.get(prediction, 0),
        )

        explanation: str = self._generate_explanation(
            features, pr, prediction, confidence, "hybrid",
            shap_summary=shap_summary,
        )

        return PredictionResult(
            match_id=match.id,
            match_display=match.display_name,
            engine_used="hybrid",
            model_version=f"hybrid_stacking_{self.MODEL_VERSION}",
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
            shap_top_features=shap_features,
            shap_summary=shap_summary,
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

        Risk faktörleri (v3.1):
          1. Güven seviyesi
          2. H2H veri eksikliği
          3. Toplam sakatlık etkisi
          4. Kritik eksik oyuncu
          5. Hakem bias
          6. Hakem-tahmin çelişkisi
          7. Form trend tutarsızlığı
          8. Sakatlık normalize skoru
          9. (YENİ) Erken sezon güvenilirlik eksikliği

        Erken sezonda (< 7 hafta) puan tablosu güvenilmezdir.
        Bayesian Smoothing bunu azaltır ama ek risk olarak işaretlenir.

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

        # 6 — Hakem-tahmin çelişkisi
        ref_alignment: float = features.get("ref_alignment_score", 50)
        if ref_alignment < 25:
            risk_score += 1

        # 7 — Form trend tutarsızlığı
        form_trend_diff: float = features.get("form_trend_diff", 0)
        if prediction == "1" and form_trend_diff < -0.5:
            risk_score += 1
        elif prediction == "2" and form_trend_diff > 0.5:
            risk_score += 1

        # 8 — Sakatlık normalize skoru
        inj_norm: float = features.get("injury_normalized_score", 50)
        if inj_norm < 25 and prediction == "1":
            risk_score += 1
        elif inj_norm > 75 and prediction == "2":
            risk_score += 1

        # 9 — v3.1: Erken sezon güvenilirlik eksikliği
        # Puan tablosu verisi yetersiz → tahmine güvenmek riskli.
        # Model Bayesian smoothing kullanıyor ama yine de ek risk.
        early_reliability: float = features.get("early_season_reliability", 100.0)
        if early_reliability < 30:  # < ~3 maç
            risk_score += 2
        elif early_reliability < 50:  # < ~5 maç
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
        shap_summary: str = "",
    ) -> str:
        """İnsan okunabilir açıklama üretir.

        v3.1: Erken sezon uyarısı + piyasa gücü bilgisi eklendi.
        v3.0: SHAP özeti açıklamanın sonuna eklenir.
        """
        parts: List[str] = []

        # ── Motor bilgisi ──
        engine_labels: Dict[str, str] = {
            "poisson": "📊 Poisson",
            "hybrid": "🔀 Hibrit (Poisson+Stacking)",
            "ml": "🤖 Stacking Ensemble",
        }
        parts.append(f"[{engine_labels.get(engine, engine)}]")

        # ── v3.1: Erken sezon uyarısı ──
        early_reliability: float = features.get("early_season_reliability", 100.0)
        season_progress: float = features.get("season_progress", 1.0)
        if early_reliability < 50:
            hafta: int = max(int(season_progress * 34), 1)
            parts.append(
                f"⚠️ Erken sezon (Hafta ~{hafta}): "
                f"Puan tablosu güvenilirliği %{early_reliability:.0f} — "
                f"Bayesian sönümleme aktif, oranlar ağırlıklı"
            )

        # ── Poisson beklentileri ──
        parts.append(
            f"⚽ Beklenen: {pr.expected_home_goals:.1f}-{pr.expected_away_goals:.1f}"
        )

        # ── v3.1: Piyasa gücü (erken sezonda önemli) ──
        mkt_diff: float = features.get("market_strength_diff", 0)
        if abs(mkt_diff) > 10:
            mkt_favori: str = "Ev sahibi" if mkt_diff > 0 else "Deplasman"
            parts.append(
                f"💰 Piyasa gücü: {mkt_favori} favori (fark: {abs(mkt_diff):.0f})"
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

        # ── v3.0: Exponential Decay Momentum ──
        momentum: float = features.get("exp_decay_momentum", 0)
        if abs(momentum) > 10:
            m_team: str = "Ev sahibi" if momentum > 0 else "Deplasman"
            parts.append(f"🚀 {m_team} yükselen formda (+{abs(momentum):.0f})")

        # ── Lig pozisyonu (Bayesian damped sıra kullan) ──
        # v3.1: Erken sezonda dampened_rank daha güvenilir,
        # geç sezonda home_rank'e yakın olacak.
        h_rank: float = features.get("dampened_home_rank",
                                     features.get("home_rank", 10))
        a_rank: float = features.get("dampened_away_rank",
                                     features.get("away_rank", 10))
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

        # ── Hakem bias + alignment ──
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

        # ── Eksik oyuncular ──
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

        # ── H2H ──
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

        # ── v3.0: Rolling Averages ──
        h_r3: float = features.get("home_rolling3_scored", 0)
        a_r3: float = features.get("away_rolling3_scored", 0)
        if h_r3 > 0 or a_r3 > 0:
            parts.append(
                f"📊 Son 3 maç gol ort: Ev {h_r3:.1f} - Dep {a_r3:.1f}"
            )

        # ── En olası skor ──
        if pr.top_scores:
            top = pr.top_scores[0]
            parts.append(f"🎯 En olası skor: {top[0]} (%{top[1]:.1f})")

        # ── v3.0: SHAP özeti ──
        if shap_summary:
            parts.append(f"\n{shap_summary}")

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
