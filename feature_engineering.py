"""
🔧 Feature Engineering Modülü v3.0
Ham veritabanı verilerini ML modeline girecek sayısal özelliklere dönüştürür.

v3.0 Değişiklikleri (v2.1 üzerinden):
  ✓ Rolling Averages: Son 3, 5 ve 10 maçlık hareketli ortalamalar
    (gol atma, gol yeme)
  ✓ Exponential Decay: Eski maçların ağırlığını zamanla azaltan form puanı
    (decay_factor = 0.85 ile üstel ağırlıklandırma)
  ✓ Interaction Features: Ev Sahibi Form × Deplasman Defans Zafiyeti,
    Hakem Bias × Ev Sahibi Form, Form Farkı × H2H Trend gibi çapraz
    özellikler
  ✓ Tüm v2.1 feature'ları korundu (geriye uyumlu + yeni 24 feature eklendi)
  ✓ Takım/hakem isimleri kategorik feature olarak döndürülebilir
    (CatBoost entegrasyonu için)

Üretilen Feature'lar (96 toplam):
  • Form (6):     Son maç form puanı, trend, fark
  • Lig (8):      Sıra, puan, fark, galibiyet oranları, composite skor
  • Hakem (8):    MS yüzdeleri, üst/kg oranları, bias skoru, alignment
  • H2H (8):      Toplam maç, kazanma oranları, üst oranı, son trend, oran tutma
  • Sakat (9):    Ceza puanı, eksik sayısı, kritik eksik, normalize skor
  • Oran (5):     İma edilen olasılıklar, bookmaker margin
  • Son Maç (6):  Ortalama gol atma/yeme, galibiyet oranı
  • Türetilmiş (7): Form-adjusted, composite güç, hakem-tahmin uyumu
  ─── YENİ (v3.0) ───
  • Rolling Avg (12): 3/5/10 maçlık gol atma/yeme ortalamaları
  • Exponential Decay (4): Üstel ağırlıklı form puanları ve momentum
  • Interaction Features (8): Çapraz özellikler (form×defans, hakem×ev vb.)
  • Kategorik (4): Takım ve hakem isimleri (CatBoost için, ayrı metot)
  ─── YENİ (v3.1 — Sezon Başı Sönümleme) ───
  • Sezon İlerleme (4): season_progress, confidence, played sayıları
  • Bayesian Sönümleme (3): Medyana çekilen sıra ve fark
  • Piyasa Gücü (4): Oran bazlı güç, fark, erken sezon güvenilirliği
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from models import (
    Match, Odds, TeamStanding, RecentMatch,
    H2HMatch, RefereeStats, Injury,
)
from config import (
    FORM_WINDOW,
    TYPICAL_SEASON_LENGTH,
    BAYESIAN_PRIOR_MATCHES,
    EARLY_SEASON_THRESHOLD,
    now_istanbul,
    TZ_ISTANBUL,
)

logger = logging.getLogger(__name__)


# ─── Türkçe Ay Haritası ──────────────────────────────────────────
_TR_MONTHS: Dict[str, int] = {
    "oca": 1,  "şub": 2,  "mar": 3,  "nis": 4,  "may": 5,  "haz": 6,
    "tem": 7,  "ağu": 8,  "eyl": 9,  "eki": 10, "kas": 11, "ara": 12,
    # ASCII fallback (scraper tutarsızlıkları için)
    "sub": 2,  "agu": 8,
}

# Türkçe relative-date anahtar kelimeleri → gün offseti
_TR_RELATIVE: Dict[str, int] = {
    "bugün": 0, "bugun": 0,
    "yarın": 1, "yarin": 1,
    "dün": -1,  "dun": -1,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Yardımcı Fonksiyonlar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _parse_turkish_date(
    date_str: Optional[str],
    ref_year: Optional[int] = None,
) -> Optional[datetime]:
    """Türkçe tarih stringlerini ``datetime`` nesnesine çevirir.

    Desteklenen formatlar
    ---------------------
    * ``"31 Oca"``       → 31 Ocak (yıl = *ref_year* veya mevcut yıl)
    * ``"5 Ara"``        → 5 Aralık
    * ``"02.08.2025"``   → 2 Ağustos 2025  (dd.mm.yyyy)
    * ``"Bugün"``        → bugünün tarihi
    * ``"Yarın"``        → yarının tarihi
    """
    if not date_str:
        return None
    s: str = str(date_str).strip()
    if ref_year is None:
        ref_year = now_istanbul().year

    # ── Relative dates ("Bugün", "Yarın") ──
    key = s.lower()
    if key in _TR_RELATIVE:
        base = now_istanbul().replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        return base + timedelta(days=_TR_RELATIVE[key])

    # ── Format 1: "dd.mm.yyyy" ──
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None

    # ── Format 2: "31 Oca" — Türkçe kısa ay ──
    m = re.match(r"(\d{1,2})\s+(\w+)", s)
    if m:
        day: int = int(m.group(1))
        month_str: str = m.group(2).lower()[:3]
        month_num: Optional[int] = _TR_MONTHS.get(month_str)
        if month_num is not None:
            try:
                return datetime(ref_year, month_num, day)
            except ValueError:
                return None

    return None


def _resolve_match_datetime(match: Match) -> datetime:
    """Bir maçın kronolojik referans noktasını belirler.

    Öncelik:
      1. match_date + match_time → gerçek maç zamanı
      2. match_date yalnız       → maç günü 00:00
      3. created_at              → DB insertion zamanı
      4. datetime.utcnow()       → son çare
    """
    dt = _parse_turkish_date(match.match_date)
    if dt is not None:
        if match.match_time:
            time_m = re.match(r"(\d{1,2}):(\d{2})", str(match.match_time).strip())
            if time_m:
                dt = dt.replace(hour=int(time_m.group(1)), minute=int(time_m.group(2)))
        return dt
    if match.created_at is not None:
        return match.created_at
    return now_istanbul()


def _form_to_points(form_str: Optional[str]) -> float:
    """Form stringini 0-100 arası puana çevirir.
    G=3, B=1, M=0. Normalize: (alınan / max) × 100.
    """
    if not form_str:
        return 50.0
    points: int = sum(
        3 if c == "G" else (1 if c == "B" else 0)
        for c in str(form_str).upper()
        if c in "GBM"
    )
    count: int = sum(1 for c in str(form_str).upper() if c in "GBM")
    if count == 0:
        return 50.0
    return (points / (count * 3)) * 100


def _form_trend(form_str: Optional[str]) -> float:
    """Son maçlardaki trendi hesaplar (−1 … +1).
    En son maç en yüksek üstel ağırlığı alır.
    """
    if not form_str or len(form_str) < 2:
        return 0.0
    chars: list[str] = [c for c in str(form_str).upper() if c in "GBM"]
    if len(chars) < 2:
        return 0.0
    weights = np.array([2**i for i in range(len(chars))], dtype=np.float64)
    values = np.array(
        [3.0 if c == "G" else (1.0 if c == "B" else 0.0) for c in chars],
        dtype=np.float64,
    )
    weighted_avg: float = float(np.average(values, weights=weights))
    return (weighted_avg - 1.5) / 1.5


def _form_exponential_decay(form_str: Optional[str], decay: float = 0.85) -> float:
    """Exponential Decay ile form puanı hesaplar (v3.0).

    En son oynanan maç en yüksek ağırlığı alır.
    Her geçmiş maç ``decay`` faktörü ile azalan ağırlık alır.

    Formül:
        score = Σ (result_i × decay^i) / Σ (3 × decay^i)
        result: G=3, B=1, M=0

    decay = 0.85 → 5 maç öncesi = %44 ağırlık, 10 maç öncesi = %20

    Returns
    -------
    float
        0-100 arası normalize edilmiş form puanı.
    """
    if not form_str:
        return 50.0
    chars: list[str] = [c for c in str(form_str).upper() if c in "GBM"]
    if not chars:
        return 50.0

    # chars[0] en eski, chars[-1] en yeni → reversed → en yeni = index 0
    chars_reversed = list(reversed(chars))
    weighted_sum: float = 0.0
    max_sum: float = 0.0
    for i, c in enumerate(chars_reversed):
        w = decay ** i
        val = 3.0 if c == "G" else (1.0 if c == "B" else 0.0)
        weighted_sum += val * w
        max_sum += 3.0 * w

    if max_sum == 0:
        return 50.0
    return (weighted_sum / max_sum) * 100.0


def _parse_score(score_str: Optional[str]) -> Tuple[int, int]:
    """Skor stringini (ev, deplasman) çiftine çevirir.
    Geçersiz veya boş skor → (-1, -1).
    """
    if not score_str:
        return (-1, -1)
    try:
        parts = str(score_str).replace(" ", "").split("-")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return (-1, -1)


def _implied_probability(odd: float) -> float:
    """Bahis oranından ima edilen olasılığı (%) hesaplar.
    P = (1 / odd) × 100.
    """
    if not odd or odd <= 1.0:
        return 0.0
    return (1.0 / odd) * 100.0


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Değeri [lo, hi] aralığına sıkıştırır."""
    return max(lo, min(val, hi))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ana Feature Extraction Sınıfı
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FeatureExtractor:
    """Bir maç için tüm sayısal özellikleri çıkaran sınıf.

    Kullanım::

        extractor = FeatureExtractor(session)
        features = extractor.extract(match)            # Dict[str, float]
        vector   = extractor.extract_vector(match)      # np.ndarray (85,)
        cat_feat = extractor.extract_categorical(match)  # Dict[str, str]
    """

    # ── 85 Numeric Feature (ML modeli için sabit sıra) ───────────
    FEATURE_NAMES: list[str] = [
        # ── Form (6) ──
        "home_form_score", "away_form_score", "form_diff",
        "home_form_trend", "away_form_trend", "form_trend_diff",

        # ── Lig pozisyonu (8) ──
        "home_rank", "away_rank", "rank_diff",
        "home_points", "away_points", "points_diff",
        "home_win_rate", "away_win_rate",

        # ── Lig composite + detay (4) ──
        "league_position_composite",
        "home_goal_diff", "away_goal_diff", "goal_diff_diff",

        # ── Hakem (8) ──
        "ref_home_pct", "ref_draw_pct", "ref_away_pct", "ref_over_pct",
        "ref_home_bias",
        "ref_over_tendency",
        "ref_kg_var_pct",
        "ref_alignment_score",

        # ── H2H (8) ──
        "h2h_total", "h2h_home_win_rate", "h2h_draw_rate",
        "h2h_away_win_rate", "h2h_over_rate",
        "h2h_recent_trend",
        "h2h_avg_goals",
        "h2h_odds_accuracy",

        # ── Sakat/Cezalı (9) ──
        "home_injury_penalty", "away_injury_penalty",
        "injury_penalty_diff", "total_injury_importance",
        "home_injury_count", "away_injury_count",
        "home_critical_injury_count", "away_critical_injury_count",
        "injury_normalized_score",

        # ── Oran bazlı (5) ──
        "implied_prob_home", "implied_prob_draw", "implied_prob_away",
        "implied_prob_over", "bookmaker_margin",

        # ── Son maç detayları (6) ──
        "home_recent_goals_scored", "home_recent_goals_conceded",
        "away_recent_goals_scored", "away_recent_goals_conceded",
        "home_recent_win_pct", "away_recent_win_pct",

        # ── Türetilmiş / Cross-Feature (7) ──
        "form_adjusted_home_score",
        "form_adjusted_away_score",
        "home_strength_composite",
        "away_strength_composite",
        "strength_diff",
        "referee_tahmin_uyumu",
        "h2h_tahmin_uyumu",

        # ═══════════════════════════════════════════════════════════
        #  v3.0 YENİ FEATURE'LAR (24 adet)
        # ═══════════════════════════════════════════════════════════

        # ── Rolling Averages: Son 3 maç (4) ──
        "home_rolling3_scored", "home_rolling3_conceded",
        "away_rolling3_scored", "away_rolling3_conceded",

        # ── Rolling Averages: Son 5 maç (4) ──
        "home_rolling5_scored", "home_rolling5_conceded",
        "away_rolling5_scored", "away_rolling5_conceded",

        # ── Rolling Averages: Son 10 maç (4) ──
        "home_rolling10_scored", "home_rolling10_conceded",
        "away_rolling10_scored", "away_rolling10_conceded",

        # ── Exponential Decay Form (4) ──
        "home_exp_decay_form", "away_exp_decay_form",
        "exp_decay_form_diff",
        "exp_decay_momentum",  # Son 3 vs son 10 maç farkı (yükselen form)

        # ── Interaction Features (8) ──
        "home_form_x_away_defense_weakness",    # Ev formu × Dep defans zafiyeti
        "away_form_x_home_defense_weakness",    # Dep formu × Ev defans zafiyeti
        "ref_bias_x_home_form",                 # Hakem bias × Ev formu
        "form_diff_x_h2h_trend",                # Form farkı × H2H son trend
        "injury_diff_x_form_diff",              # Sakatlık farkı × Form farkı
        "home_attack_x_away_concede",           # Ev atak × Dep gol yeme
        "rank_diff_x_form_diff",                # Sıra farkı × Form farkı
        "implied_prob_x_ref_alignment",         # Oran olasılığı × Hakem uyumu

        # ═══════════════════════════════════════════════════════════
        #  v3.1 SEZON BAŞI SÖNÜMLEME FEATURE'LARI (11 adet)
        # ═══════════════════════════════════════════════════════════

        # ── Sezon İlerleme (4) ──
        "season_progress",                      # 0.0-1.0 sezon ilerleme oranı
        "season_confidence",                    # Sigmoid tabanlı güven (0-100)
        "home_played",                          # Ev sahibi oynanan maç sayısı
        "away_played",                          # Deplasman oynanan maç sayısı

        # ── Bayesian Sönümlenmiş Sıralama (3) ──
        "dampened_home_rank",                   # Medyana çekilen sıra
        "dampened_away_rank",
        "dampened_rank_diff",

        # ── Piyasa Gücü / Erken Sezon Güvenilirliği (4) ──
        "relative_market_strength_home",        # Oran bazlı ev sahibi gücü (0-100)
        "relative_market_strength_away",        # Oran bazlı deplasman gücü (0-100)
        "market_strength_diff",                 # Piyasa gücü farkı
        "early_season_reliability",             # Puan tablosu güvenilirliği (0-100)
    ]

    # ── CatBoost kategorik feature isimleri ──
    CATEGORICAL_FEATURE_NAMES: list[str] = [
        "home_team_name",
        "away_team_name",
        "referee_name",
        "league_name",
    ]

    def __init__(self, session: Session) -> None:
        self.session: Session = session

    # ─── Public API ───────────────────────────────────────────────

    def extract(self, match: Match) -> Dict[str, float]:
        """Bir maç için tüm 85 sayısal özelliği çıkarır."""
        f: Dict[str, float] = {}
        ref_dt: datetime = _resolve_match_datetime(match)

        self._extract_standing_features(match, f)
        self._extract_referee_features(match, f)
        self._extract_h2h_features(match, f, ref_dt)
        self._extract_injury_features(match, f)
        self._extract_odds_features(match, f)
        self._extract_recent_match_features(match, f, ref_dt)
        self._extract_derived_features(f)

        # ── v3.0 Yeni Feature Grupları ──
        self._extract_rolling_averages(match, f, ref_dt)
        self._extract_exponential_decay_features(match, f)
        self._extract_interaction_features(f)

        # ── v3.1 Sezon Başı Sönümleme ──
        self._extract_season_dampening_features(match, f)

        return f

    def extract_vector(self, match: Match) -> np.ndarray:
        """Feature dict'ini sabit sıralı numpy vektörüne çevirir (96 boyut)."""
        feat: Dict[str, float] = self.extract(match)
        return np.array(
            [feat.get(name, 0.0) for name in self.FEATURE_NAMES],
            dtype=np.float64,
        )

    def extract_categorical(self, match: Match) -> Dict[str, str]:
        """CatBoost için kategorik feature'ları döndürür.

        Returns
        -------
        Dict[str, str]
            Takım isimleri, hakem adı ve lig ismi.
        """
        cat: Dict[str, str] = {}
        cat["home_team_name"] = match.home_team.name if match.home_team else "Bilinmiyor"
        cat["away_team_name"] = match.away_team.name if match.away_team else "Bilinmiyor"
        cat["referee_name"] = match.referee.name if match.referee else "Bilinmiyor"
        cat["league_name"] = match.league.name if match.league else "Bilinmiyor"
        return cat

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  1) FORM & LİG POZİSYONU
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_standing_features(
        self, match: Match, f: Dict[str, float],
    ) -> None:
        """Puan tablosu, form ve lig-pozisyonu composite skoru."""
        home_st: Optional[TeamStanding] = self._get_standing(match.id, "Ev Sahibi")
        away_st: Optional[TeamStanding] = self._get_standing(match.id, "Deplasman")

        # ── Form puanları ──
        h_form: float = _form_to_points(home_st.form if home_st else None)
        a_form: float = _form_to_points(away_st.form if away_st else None)
        f["home_form_score"] = h_form
        f["away_form_score"] = a_form
        f["form_diff"] = h_form - a_form

        h_trend: float = _form_trend(home_st.form if home_st else None)
        a_trend: float = _form_trend(away_st.form if away_st else None)
        f["home_form_trend"] = h_trend
        f["away_form_trend"] = a_trend
        f["form_trend_diff"] = h_trend - a_trend

        # ── Sıra & Puan ──
        h_rank: int = home_st.rank if home_st and home_st.rank else 10
        a_rank: int = away_st.rank if away_st and away_st.rank else 10
        f["home_rank"] = float(h_rank)
        f["away_rank"] = float(a_rank)
        f["rank_diff"] = float(a_rank - h_rank)

        h_pts: int = home_st.points if home_st and home_st.points else 0
        a_pts: int = away_st.points if away_st and away_st.points else 0
        f["home_points"] = float(h_pts)
        f["away_points"] = float(a_pts)
        f["points_diff"] = float(h_pts - a_pts)

        # ── Gol Averajı ──
        h_gd: int = home_st.goal_diff if home_st and home_st.goal_diff else 0
        a_gd: int = away_st.goal_diff if away_st and away_st.goal_diff else 0
        f["home_goal_diff"] = float(h_gd)
        f["away_goal_diff"] = float(a_gd)
        f["goal_diff_diff"] = float(h_gd - a_gd)

        # ── Galibiyet Oranları ──
        h_played: int = home_st.played if home_st and home_st.played else 1
        h_won: int = home_st.won if home_st and home_st.won else 0
        a_played: int = away_st.played if away_st and away_st.played else 1
        a_won: int = away_st.won if away_st and away_st.won else 0
        f["home_win_rate"] = (h_won / max(h_played, 1)) * 100
        f["away_win_rate"] = (a_won / max(a_played, 1)) * 100

        # ── Lig Pozisyonu Composite ──
        sira_diff: int = a_rank - h_rank
        puan_diff: int = h_pts - a_pts
        av_diff: int = h_gd - a_gd
        composite: float = 50.0 + (sira_diff * 2) + (puan_diff * 0.5) + (av_diff * 0.5)
        f["league_position_composite"] = _clamp(composite)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  2) HAKEM (Bias + KG + Üst + Alignment)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_referee_features(
        self, match: Match, f: Dict[str, float],
    ) -> None:
        """Hakem istatistikleri, bias skoru ve alignment feature."""
        ref_stats: Optional[RefereeStats] = (
            self.session.query(RefereeStats)
            .filter_by(match_id=match.id)
            .first()
        )

        ms1: float = ref_stats.ms1_pct if ref_stats and ref_stats.ms1_pct else 33.3
        msx: float = ref_stats.msx_pct if ref_stats and ref_stats.msx_pct else 33.3
        ms2: float = ref_stats.ms2_pct if ref_stats and ref_stats.ms2_pct else 33.3
        ust: float = ref_stats.ust_pct if ref_stats and ref_stats.ust_pct else 50.0
        kg_var: float = ref_stats.kg_var_pct if ref_stats and ref_stats.kg_var_pct else 50.0

        f["ref_home_pct"] = ms1
        f["ref_draw_pct"] = msx
        f["ref_away_pct"] = ms2
        f["ref_over_pct"] = ust
        f["ref_home_bias"] = ms1 - ms2
        f["ref_over_tendency"] = ust
        f["ref_kg_var_pct"] = kg_var

        # ── Alignment: Oran-bazlı favoriye hakem uyumu ──
        odds: Optional[Odds] = (
            self.session.query(Odds)
            .filter_by(match_id=match.id)
            .first()
        )
        if odds and odds.ms_1 and odds.ms_x and odds.ms_2:
            ip_h: float = _implied_probability(odds.ms_1)
            ip_d: float = _implied_probability(odds.ms_x)
            ip_a: float = _implied_probability(odds.ms_2)
            best = max(ip_h, ip_d, ip_a)
            if best == ip_h:
                f["ref_alignment_score"] = ms1
            elif best == ip_a:
                f["ref_alignment_score"] = ms2
            else:
                f["ref_alignment_score"] = msx
        else:
            f["ref_alignment_score"] = 50.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  3) H2H — Fail-Safe Tarih Filtreli
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_h2h_features(
        self,
        match: Match,
        f: Dict[str, float],
        ref_dt: datetime,
    ) -> None:
        """H2H (Head-to-Head) özellikleri — fail-safe tarih filtreli."""
        h2h_all: List[H2HMatch] = (
            self.session.query(H2HMatch)
            .filter_by(match_id=match.id)
            .all()
        )

        # ── Fail-Safe tarih filtresi ──
        h2h_list: List[H2HMatch] = []
        excluded_count: int = 0
        for h in h2h_all:
            h_dt: Optional[datetime] = _parse_turkish_date(h.date)
            if h_dt is None:
                excluded_count += 1
                continue
            if h_dt < ref_dt:
                h2h_list.append(h)

        if excluded_count > 0:
            logger.debug(
                "H2H Fail-Safe: %d kayıt dışlandı (match_id=%d)",
                excluded_count, match.id,
            )

        total: int = len(h2h_list)
        f["h2h_total"] = float(total)

        if total == 0:
            f["h2h_home_win_rate"] = 33.3
            f["h2h_draw_rate"] = 33.3
            f["h2h_away_win_rate"] = 33.3
            f["h2h_over_rate"] = 50.0
            f["h2h_recent_trend"] = 0.0
            f["h2h_avg_goals"] = 2.5
            f["h2h_odds_accuracy"] = 50.0
            return

        home_wins: int = 0
        draws: int = 0
        away_wins: int = 0
        overs: int = 0
        total_goals: int = 0
        valid_scores: int = 0
        odds_correct: int = 0
        odds_total: int = 0

        for h in h2h_list:
            hg, ag = _parse_score(h.score)
            if hg < 0:
                continue
            valid_scores += 1
            total_goals += hg + ag

            if hg > ag:
                home_wins += 1
            elif hg == ag:
                draws += 1
            else:
                away_wins += 1
            if hg + ag > 2:
                overs += 1

            for won_flag in (h.odd_1_won, h.odd_x_won, h.odd_2_won):
                if won_flag is not None:
                    odds_total += 1
                    if won_flag:
                        odds_correct += 1

        f["h2h_home_win_rate"] = (home_wins / total) * 100 if total else 33.3
        f["h2h_draw_rate"] = (draws / total) * 100 if total else 33.3
        f["h2h_away_win_rate"] = (away_wins / total) * 100 if total else 33.3
        f["h2h_over_rate"] = (overs / total) * 100 if total else 50.0
        f["h2h_avg_goals"] = total_goals / valid_scores if valid_scores else 2.5
        f["h2h_odds_accuracy"] = (
            (odds_correct / odds_total * 100) if odds_total else 50.0
        )

        # ── Son 3 H2H maçın trendi ──
        recent_h2h: List[H2HMatch] = h2h_list[:3]
        trend_val: float = 0.0
        trend_count: int = 0
        for h in recent_h2h:
            hg, ag = _parse_score(h.score)
            if hg < 0:
                continue
            trend_count += 1
            if hg > ag:
                trend_val += 1.0
            elif hg < ag:
                trend_val -= 1.0
        f["h2h_recent_trend"] = (trend_val / trend_count) if trend_count else 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  4) SAKAT/CEZALI (Gelişmiş Önem Puanı)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_injury_features(
        self, match: Match, f: Dict[str, float],
    ) -> None:
        """Sakat/cezalı oyuncu feature'ları."""
        injuries: List[Injury] = (
            self.session.query(Injury)
            .filter_by(match_id=match.id)
            .all()
        )

        home_penalty: float = 0.0
        away_penalty: float = 0.0
        home_count: int = 0
        away_count: int = 0
        home_critical: int = 0
        away_critical: int = 0

        for inj in injuries:
            imp: float = self._calculate_importance(inj)
            if inj.team_id == match.home_team_id:
                home_penalty += imp
                home_count += 1
                if imp >= 7.0:
                    home_critical += 1
            elif inj.team_id == match.away_team_id:
                away_penalty += imp
                away_count += 1
                if imp >= 7.0:
                    away_critical += 1

        f["home_injury_penalty"] = home_penalty
        f["away_injury_penalty"] = away_penalty
        f["injury_penalty_diff"] = away_penalty - home_penalty
        f["total_injury_importance"] = home_penalty + away_penalty
        f["home_injury_count"] = float(home_count)
        f["away_injury_count"] = float(away_count)
        f["home_critical_injury_count"] = float(home_critical)
        f["away_critical_injury_count"] = float(away_critical)

        ceza_farki: float = away_penalty - home_penalty
        f["injury_normalized_score"] = _clamp(50.0 + ceza_farki * 2)

    @staticmethod
    def _calculate_importance(inj: Injury) -> float:
        """Oyuncunun takım için önem puanını hesaplar.
        importance = (Baz_Puan + Skorer_Katkısı) × Pozisyon_Çarpanı
        Tipik aralık: 1.0 – 22.5.
        """
        score: float = 0.0
        starts: int = inj.starts or 0
        if starts >= 15:
            score += 10.0
        elif starts >= 10:
            score += 7.0
        elif starts >= 5:
            score += 4.0
        else:
            score += 1.0

        contrib: int = (inj.goals or 0) + (inj.assists or 0)
        if contrib >= 10:
            score += 5.0
        elif contrib >= 5:
            score += 3.0
        elif contrib >= 2:
            score += 1.0

        pos: str = (inj.position or "").lower()
        if pos in ("forvet", "santrafor"):
            score *= 1.2
        elif pos in ("orta saha", "ortasaha"):
            score *= 1.1
        elif pos == "kaleci":
            score *= 1.5

        return round(score, 2)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  5) ORAN BAZLI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_odds_features(
        self, match: Match, f: Dict[str, float],
    ) -> None:
        """Bahis oranı özellikleri + bookmaker marjı."""
        odds: Optional[Odds] = (
            self.session.query(Odds)
            .filter_by(match_id=match.id)
            .first()
        )
        if not odds:
            f["implied_prob_home"] = 33.3
            f["implied_prob_draw"] = 33.3
            f["implied_prob_away"] = 33.3
            f["implied_prob_over"] = 50.0
            f["bookmaker_margin"] = 0.0
            return

        ip_home: float = _implied_probability(odds.ms_1 or 0)
        ip_draw: float = _implied_probability(odds.ms_x or 0)
        ip_away: float = _implied_probability(odds.ms_2 or 0)
        ip_over: float = _implied_probability(odds.ust_2_5 or 0)

        f["implied_prob_home"] = ip_home
        f["implied_prob_draw"] = ip_draw
        f["implied_prob_away"] = ip_away
        f["implied_prob_over"] = ip_over

        total_prob: float = ip_home + ip_draw + ip_away
        f["bookmaker_margin"] = max(total_prob - 100, 0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  6) SON MAÇ DETAYLARI (Fail-Safe Tarih Filtreli)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_recent_match_features(
        self,
        match: Match,
        f: Dict[str, float],
        ref_dt: datetime,
    ) -> None:
        """Son maç detay özellikleri — fail-safe tarih filtreli."""
        home_all: List[RecentMatch] = (
            self.session.query(RecentMatch)
            .filter_by(match_id=match.id, team_type="Ev Sahibi")
            .all()
        )
        away_all: List[RecentMatch] = (
            self.session.query(RecentMatch)
            .filter_by(match_id=match.id, team_type="Deplasman")
            .all()
        )

        home_recent: List[RecentMatch] = self._filter_recent_by_date(
            home_all, ref_dt,
        )[:FORM_WINDOW]
        away_recent: List[RecentMatch] = self._filter_recent_by_date(
            away_all, ref_dt,
        )[:FORM_WINDOW]

        h_scored, h_conceded, h_wins = self._calc_recent_stats(home_recent)
        f["home_recent_goals_scored"] = h_scored
        f["home_recent_goals_conceded"] = h_conceded
        f["home_recent_win_pct"] = h_wins

        a_scored, a_conceded, a_wins = self._calc_recent_stats(away_recent)
        f["away_recent_goals_scored"] = a_scored
        f["away_recent_goals_conceded"] = a_conceded
        f["away_recent_win_pct"] = a_wins

    @staticmethod
    def _filter_recent_by_date(
        matches: List[RecentMatch],
        target_dt: datetime,
    ) -> List[RecentMatch]:
        """Parse edilemeyen tarihler DIŞLANIR (data leakage fix)."""
        filtered: List[RecentMatch] = []
        for m in matches:
            m_dt: Optional[datetime] = _parse_turkish_date(m.date)
            if m_dt is None:
                continue
            if m_dt < target_dt:
                filtered.append(m)
        return filtered

    @staticmethod
    def _calc_recent_stats(
        matches: List[RecentMatch],
    ) -> Tuple[float, float, float]:
        """Son maçlardan ortalama gol ve galibiyet oranı hesaplar.
        Returns: (avg_scored, avg_conceded, win_pct)
        """
        if not matches:
            return 0.0, 0.0, 50.0

        total_scored: float = 0.0
        total_conceded: float = 0.0
        wins: int = 0
        valid: int = 0

        for m in matches:
            hg, ag = _parse_score(m.score)
            if hg < 0:
                continue
            valid += 1
            if m.result == "Galibiyet":
                wins += 1

            if hg == ag:
                total_scored += hg
                total_conceded += ag
            elif (m.result in ("Galibiyet",) and hg > ag) or \
                 (m.result in ("Mağlubiyet", "Maglubiyet") and hg < ag) or \
                 (m.result is None):
                total_scored += hg
                total_conceded += ag
            else:
                total_scored += ag
                total_conceded += hg

        if valid == 0:
            return 0.0, 0.0, 50.0

        return (
            total_scored / valid,
            total_conceded / valid,
            (wins / valid) * 100,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  7) TÜRETİLMİŞ / CROSS-FEATURE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_derived_features(self, f: Dict[str, float]) -> None:
        """Diğer feature'lardan türetilen composite özellikler."""
        # ── Form-adjusted skor ──
        h_adj: float = max(
            0.0,
            f.get("home_form_score", 50) - f.get("home_injury_penalty", 0) * 2,
        )
        a_adj: float = max(
            0.0,
            f.get("away_form_score", 50) - f.get("away_injury_penalty", 0) * 2,
        )
        f["form_adjusted_home_score"] = h_adj
        f["form_adjusted_away_score"] = a_adj

        # ── Genel güç composite ──
        h_goals_scored: float = f.get("home_recent_goals_scored", 0)
        a_goals_scored: float = f.get("away_recent_goals_scored", 0)
        h_win_rate: float = f.get("home_win_rate", 50)
        a_win_rate: float = f.get("away_win_rate", 50)
        league_comp: float = f.get("league_position_composite", 50)

        h_strength: float = (
            h_adj * 0.35
            + league_comp * 0.30
            + min(h_goals_scored * 25, 100) * 0.15
            + h_win_rate * 0.20
        )
        a_strength: float = (
            a_adj * 0.35
            + (100 - league_comp) * 0.30
            + min(a_goals_scored * 25, 100) * 0.15
            + a_win_rate * 0.20
        )

        f["home_strength_composite"] = round(h_strength, 2)
        f["away_strength_composite"] = round(a_strength, 2)
        f["strength_diff"] = round(h_strength - a_strength, 2)

        # ── Tahmin-uyum Feature'ları ──
        ip_h: float = f.get("implied_prob_home", 33.3)
        ip_d: float = f.get("implied_prob_draw", 33.3)
        ip_a: float = f.get("implied_prob_away", 33.3)
        best_ip: float = max(ip_h, ip_d, ip_a)

        if best_ip == ip_h:
            f["referee_tahmin_uyumu"] = f.get("ref_home_pct", 33.3)
            f["h2h_tahmin_uyumu"] = f.get("h2h_home_win_rate", 33.3)
        elif best_ip == ip_a:
            f["referee_tahmin_uyumu"] = f.get("ref_away_pct", 33.3)
            f["h2h_tahmin_uyumu"] = f.get("h2h_away_win_rate", 33.3)
        else:
            f["referee_tahmin_uyumu"] = f.get("ref_draw_pct", 33.3)
            f["h2h_tahmin_uyumu"] = f.get("h2h_draw_rate", 33.3)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  8) v3.0 — ROLLING AVERAGES (Son 3, 5, 10 Maç)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_rolling_averages(
        self,
        match: Match,
        f: Dict[str, float],
        ref_dt: datetime,
    ) -> None:
        """Son N maçlık hareketli gol ortalamaları (3, 5, 10 pencere)."""
        home_all: List[RecentMatch] = (
            self.session.query(RecentMatch)
            .filter_by(match_id=match.id, team_type="Ev Sahibi")
            .all()
        )
        away_all: List[RecentMatch] = (
            self.session.query(RecentMatch)
            .filter_by(match_id=match.id, team_type="Deplasman")
            .all()
        )

        home_filtered: List[RecentMatch] = self._filter_recent_by_date(home_all, ref_dt)
        away_filtered: List[RecentMatch] = self._filter_recent_by_date(away_all, ref_dt)

        for window in (3, 5, 10):
            h_scored, h_conceded = self._rolling_goals(home_filtered[:window])
            a_scored, a_conceded = self._rolling_goals(away_filtered[:window])

            f[f"home_rolling{window}_scored"] = h_scored
            f[f"home_rolling{window}_conceded"] = h_conceded
            f[f"away_rolling{window}_scored"] = a_scored
            f[f"away_rolling{window}_conceded"] = a_conceded

    @staticmethod
    def _rolling_goals(
        matches: List[RecentMatch],
    ) -> Tuple[float, float]:
        """N maçlık pencerede ortalama gol atma/yeme.
        Returns: (avg_scored, avg_conceded)
        """
        if not matches:
            return 0.0, 0.0

        scored: float = 0.0
        conceded: float = 0.0
        valid: int = 0

        for m in matches:
            hg, ag = _parse_score(m.score)
            if hg < 0:
                continue
            valid += 1

            if m.result == "Galibiyet":
                if hg > ag:
                    scored += hg; conceded += ag
                else:
                    scored += ag; conceded += hg
            elif m.result in ("Mağlubiyet", "Maglubiyet"):
                if hg < ag:
                    scored += hg; conceded += ag
                else:
                    scored += ag; conceded += hg
            else:
                scored += hg; conceded += ag

        if valid == 0:
            return 0.0, 0.0
        return scored / valid, conceded / valid

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  9) v3.0 — EXPONENTIAL DECAY FORM
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_exponential_decay_features(
        self,
        match: Match,
        f: Dict[str, float],
    ) -> None:
        """Exponential Decay ile form puanları.

        Klasik form puanına kıyasla eski maçları geometrik olarak azaltır.
        decay = 0.85 → 5 maç öncesi %44, 10 maç öncesi %20 ağırlık.

        Momentum: Son 3 maçlık exp-decay ile genel form arasındaki fark.
        Pozitif momentum → yükselen form.
        """
        home_st: Optional[TeamStanding] = self._get_standing(match.id, "Ev Sahibi")
        away_st: Optional[TeamStanding] = self._get_standing(match.id, "Deplasman")

        h_exp: float = _form_exponential_decay(home_st.form if home_st else None)
        a_exp: float = _form_exponential_decay(away_st.form if away_st else None)

        f["home_exp_decay_form"] = h_exp
        f["away_exp_decay_form"] = a_exp
        f["exp_decay_form_diff"] = h_exp - a_exp

        # ── Momentum: Son 3 maç gol ortalaması vs son 10 ortalaması ──
        h_form_3 = f.get("home_rolling3_scored", 0) * 25  # Gol bazlı proxy 0-100
        a_form_3 = f.get("away_rolling3_scored", 0) * 25
        h_form_10 = f.get("home_rolling10_scored", 0) * 25
        a_form_10 = f.get("away_rolling10_scored", 0) * 25

        h_momentum = h_form_3 - h_form_10 if h_form_10 > 0 else 0.0
        a_momentum = a_form_3 - a_form_10 if a_form_10 > 0 else 0.0
        f["exp_decay_momentum"] = h_momentum - a_momentum

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  10) v3.0 — INTERACTION FEATURES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_interaction_features(self, f: Dict[str, float]) -> None:
        """Çapraz (interaction) özellikler.

        İki farklı boyutu çarparak yeni sinyal üretir.
        Ağaç tabanlı modeller bunu örtük yapabilir ama açık
        interaction feature'lar daha düşük ağaç derinliğinde yakalama sağlar.

        1. home_form_x_away_defense_weakness: Ev formu × Dep gol yeme oranı
        2. away_form_x_home_defense_weakness: Dep formu × Ev gol yeme oranı
        3. ref_bias_x_home_form: Hakem ev bias'ı × Ev formu (normalize /100)
        4. form_diff_x_h2h_trend: Form farkı × H2H son trend
        5. injury_diff_x_form_diff: Sakatlık farkı × Form farkı
        6. home_attack_x_away_concede: Ev gol atma × Dep gol yeme
        7. rank_diff_x_form_diff: Sıra farkı × Form farkı
        8. implied_prob_x_ref_alignment: Bookmaker favori olasılığı × Hakem uyumu
        """
        # 1. Ev Formu × Deplasman Defans Zafiyeti
        home_form: float = f.get("home_form_score", 50)
        away_concede: float = f.get("away_recent_goals_conceded", 1.0)
        f["home_form_x_away_defense_weakness"] = (home_form / 100.0) * away_concede

        # 2. Deplasman Formu × Ev Defans Zafiyeti
        away_form: float = f.get("away_form_score", 50)
        home_concede: float = f.get("home_recent_goals_conceded", 1.0)
        f["away_form_x_home_defense_weakness"] = (away_form / 100.0) * home_concede

        # 3. Hakem Bias × Ev Formu
        ref_bias: float = f.get("ref_home_bias", 0)
        f["ref_bias_x_home_form"] = (ref_bias / 100.0) * home_form

        # 4. Form Farkı × H2H Trend
        form_diff: float = f.get("form_diff", 0)
        h2h_trend: float = f.get("h2h_recent_trend", 0)
        f["form_diff_x_h2h_trend"] = (form_diff / 100.0) * h2h_trend

        # 5. Sakatlık Farkı × Form Farkı
        injury_diff: float = f.get("injury_penalty_diff", 0)
        f["injury_diff_x_form_diff"] = (injury_diff / 20.0) * (form_diff / 100.0)

        # 6. Ev Sahibi Atak × Deplasman Gol Yeme
        home_scored: float = f.get("home_recent_goals_scored", 1.0)
        f["home_attack_x_away_concede"] = home_scored * away_concede

        # 7. Sıra Farkı × Form Farkı
        rank_diff: float = f.get("rank_diff", 0)
        f["rank_diff_x_form_diff"] = (rank_diff / 20.0) * (form_diff / 100.0)

        # 8. Oran Olasılığı × Hakem Uyumu
        ip_max: float = max(
            f.get("implied_prob_home", 33.3),
            f.get("implied_prob_draw", 33.3),
            f.get("implied_prob_away", 33.3),
        )
        ref_align: float = f.get("ref_alignment_score", 50)
        f["implied_prob_x_ref_alignment"] = (ip_max / 100.0) * (ref_align / 100.0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  11) v3.1 — SEZON BAŞI SÖNÜMLEME (Bayesian Smoothing)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _extract_season_dampening_features(
        self,
        match: Match,
        f: Dict[str, float],
    ) -> None:
        """Sezon başı sönümleme (Bayesian Smoothing) feature'ları.

        **Problem:** Sezonun ilk haftalarında (1–7. hafta) puan tablosu
        yanıltıcıdır. Tarihsel olarak zayıf bir takım ilk maçını 5-0
        kazanırsa, ham verilerle 1. sırada görünür ve model onu büyük
        favori olarak tahmin eder.

        **Çözüm — Bayesian Shrinkage:**
        * ``dampened_rank = (raw_rank × played + median_rank × C) / (played + C)``
        * ``C = 5`` (prior) → Her takım "ortalama" kabul edilir ve yeterli
          veri toplandıkça gerçek gücüne yakınsar.
        * ``season_progress`` ve ``season_confidence`` feature'ları modele
          "şu an sezonun neresindeyiz" bilgisi verir.
        * ``relative_market_strength`` bahis oranlarından türetilir ve
          erken sezonda puan tablosundan çok daha güvenilirdir.

        Üretilen Feature'lar (11):
          season_progress, season_confidence, home_played, away_played,
          dampened_home_rank, dampened_away_rank, dampened_rank_diff,
          relative_market_strength_home, relative_market_strength_away,
          market_strength_diff, early_season_reliability
        """
        home_st: Optional[TeamStanding] = self._get_standing(
            match.id, "Ev Sahibi",
        )
        away_st: Optional[TeamStanding] = self._get_standing(
            match.id, "Deplasman",
        )

        C: float = float(BAYESIAN_PRIOR_MATCHES)  # Prior sabiti
        MEDIAN_RANK: float = 10.0  # 20 takımlık lig medyanı

        # ── Oynanan maç sayıları ──
        h_played: int = home_st.played if home_st and home_st.played else 0
        a_played: int = away_st.played if away_st and away_st.played else 0
        f["home_played"] = float(h_played)
        f["away_played"] = float(a_played)

        avg_played: float = (h_played + a_played) / 2.0

        # ── Season Progress (0.0 – 1.0) ──
        season_progress: float = min(
            avg_played / max(TYPICAL_SEASON_LENGTH, 1), 1.0,
        )
        f["season_progress"] = season_progress

        # ── Season Confidence: Sigmoid fonksiyonu ──
        # k=15, midpoint=EARLY_SEASON_THRESHOLD (~0.20)
        # 0 maç → ~%5 güven, 7 maç → ~%50 güven, 17 maç → ~%95 güven
        k: float = 15.0
        midpoint: float = EARLY_SEASON_THRESHOLD
        exp_val: float = -k * (season_progress - midpoint)
        exp_val = max(min(exp_val, 500.0), -500.0)  # Overflow koruması
        season_confidence: float = 100.0 / (1.0 + math.exp(exp_val))
        f["season_confidence"] = season_confidence

        # ── Dampened Rank (Bayesian Shrinkage → Medyana Çekme) ──
        # Formül: dampened = (raw × played + median × C) / (played + C)
        # Hafta 1: (1 × 1 + 10 × 5) / 6 = 8.5 (1. sıra → ~8.5 damped)
        # Hafta 20: (1 × 20 + 10 × 5) / 25 = 2.8 (1. sıra → ~2.8 damped)
        h_rank_raw: float = float(
            home_st.rank if home_st and home_st.rank else MEDIAN_RANK,
        )
        a_rank_raw: float = float(
            away_st.rank if away_st and away_st.rank else MEDIAN_RANK,
        )

        dampened_h_rank: float = (
            (h_rank_raw * h_played + MEDIAN_RANK * C) / (h_played + C)
        )
        dampened_a_rank: float = (
            (a_rank_raw * a_played + MEDIAN_RANK * C) / (a_played + C)
        )

        f["dampened_home_rank"] = dampened_h_rank
        f["dampened_away_rank"] = dampened_a_rank
        f["dampened_rank_diff"] = dampened_a_rank - dampened_h_rank

        # ── Relative Market Strength (Oran Bazlı Güç Proxy) ──
        # Erken sezonda puan tablosu güvenilir değilken, bahis piyasası
        # takımların gerçek gücünü daha iyi yansıtır.
        # Normalize implied probability → 0-100 arası güç skoru.
        ip_home: float = f.get("implied_prob_home", 33.3)
        ip_away: float = f.get("implied_prob_away", 33.3)
        ip_draw: float = f.get("implied_prob_draw", 33.3)

        ip_total: float = ip_home + ip_draw + ip_away
        if ip_total > 0:
            rms_home: float = (ip_home / ip_total) * 100.0
            rms_away: float = (ip_away / ip_total) * 100.0
        else:
            rms_home = 33.3
            rms_away = 33.3

        f["relative_market_strength_home"] = rms_home
        f["relative_market_strength_away"] = rms_away
        f["market_strength_diff"] = rms_home - rms_away

        # ── Early Season Reliability (Composite Güvenilirlik) ──
        # Bayesian güvenilirlik: n / (n + C)
        # 0 maç → %0, 5 maç → %50, 10 maç → %67, 20 maç → %80
        min_played: int = min(h_played, a_played)
        reliability: float = (min_played / (min_played + C)) * 100.0
        f["early_season_reliability"] = reliability

        logger.debug(
            "Sezon sönümleme: progress=%.2f, confidence=%.1f%%, "
            "reliability=%.1f%%, h_rank %.1f→%.1f, a_rank %.1f→%.1f",
            season_progress, season_confidence, reliability,
            h_rank_raw, dampened_h_rank, a_rank_raw, dampened_a_rank,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Veritabanı Sorguları
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _get_standing(
        self, match_id: int, team_type: str,
    ) -> Optional[TeamStanding]:
        """Belirtilen maç ve takım tipi için puan tablosu kaydı döndürür."""
        return (
            self.session.query(TeamStanding)
            .filter_by(match_id=match_id, team_type=team_type)
            .first()
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Eğitim İçin Toplu Feature Çıkarma (Kronolojik Sıralı)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_training_dataset(
    session: Session,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sonuçlanmış maçlardan eğitim veri seti oluşturur.

    Maçlar kronolojik sıralanır.
    Returns: (X, y) — X: feature matrisi (n, 85), y: etiketler (0/1/2).
    """
    extractor = FeatureExtractor(session)

    finished_matches: List[Match] = (
        session.query(Match)
        .filter(Match.is_finished == True)  # noqa: E712
        .order_by(Match.created_at.asc())
        .all()
    )

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    label_map: Dict[str, int] = {"1": 0, "X": 1, "2": 2}

    for match in finished_matches:
        result: Optional[str] = match.result
        if result is None or result not in label_map:
            continue
        try:
            vec: np.ndarray = extractor.extract_vector(match)
            X_list.append(vec)
            y_list.append(label_map[result])
        except Exception as e:
            logger.warning(
                "Feature çıkarma hatası (Maç %s): %s", match.nesine_code, e,
            )
            continue

    if not X_list:
        return (
            np.array([]).reshape(0, len(FeatureExtractor.FEATURE_NAMES)),
            np.array([], dtype=np.int64),
        )

    return np.array(X_list), np.array(y_list, dtype=np.int64)


def build_training_dataset_with_categorical(
    session: Session,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    """Eğitim veri seti + kategorik feature'lar (CatBoost için).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]
        (X_numeric, y, categorical_features_list)
    """
    extractor = FeatureExtractor(session)

    finished_matches: List[Match] = (
        session.query(Match)
        .filter(Match.is_finished == True)  # noqa: E712
        .order_by(Match.created_at.asc())
        .all()
    )

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    cat_list: List[Dict[str, str]] = []
    label_map: Dict[str, int] = {"1": 0, "X": 1, "2": 2}

    for match in finished_matches:
        result: Optional[str] = match.result
        if result is None or result not in label_map:
            continue
        try:
            vec: np.ndarray = extractor.extract_vector(match)
            cat: Dict[str, str] = extractor.extract_categorical(match)
            X_list.append(vec)
            y_list.append(label_map[result])
            cat_list.append(cat)
        except Exception as e:
            logger.warning(
                "Feature çıkarma hatası (Maç %s): %s", match.nesine_code, e,
            )
            continue

    if not X_list:
        return (
            np.array([]).reshape(0, len(FeatureExtractor.FEATURE_NAMES)),
            np.array([], dtype=np.int64),
            [],
        )

    return np.array(X_list), np.array(y_list, dtype=np.int64), cat_list
