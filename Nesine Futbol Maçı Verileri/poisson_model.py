"""
📊 Poisson Dağılımı Modülü
Takımların gol beklentileri üzerinden skor tahmini ve olasılık hesaplama.

Temel Kavramlar:
  • Attack Strength  = Takımın gol atma oranı / lig ortalaması
  • Defense Strength = Takımın gol yeme oranı / lig ortalaması
  • λ (lambda)       = Attack × Defense × Lig Ort. Gol

Çıktılar:
  • Olası skor matrisi (0-0 → 5-5)
  • MS 1/X/2 olasılıkları
  • Alt/Üst 2.5 olasılıkları
  • KG Var/Yok olasılıkları
  • En olası 3 skor
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.stats import poisson

from config import (
    POISSON_MAX_GOALS,
    LEAGUE_AVG_GOALS,
    BAYESIAN_PRIOR_MATCHES,
    TYPICAL_SEASON_LENGTH,
    FORM_WINDOW,
)

logger = logging.getLogger(__name__)


@dataclass
class PoissonResult:
    """Poisson analiz sonucunu taşıyan veri sınıfı."""
    home_lambda: float = 0.0
    away_lambda: float = 0.0

    prob_home: float = 0.0
    prob_draw: float = 0.0
    prob_away: float = 0.0

    prob_over_25: float = 0.0
    prob_under_25: float = 0.0

    prob_btts_yes: float = 0.0    # KG Var
    prob_btts_no: float = 0.0     # KG Yok

    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0
    expected_total_goals: float = 0.0

    # Skor matrisi (max_goals+1 × max_goals+1)
    score_matrix: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    # En olası skorlar [(skor_str, olasılık), ...]
    top_scores: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def prediction(self) -> str:
        """En olası sonuç: 1, X veya 2."""
        probs = {'1': self.prob_home, 'X': self.prob_draw, '2': self.prob_away}
        return max(probs, key=probs.get)

    def summary(self) -> str:
        return (
            f"λ_ev={self.home_lambda:.2f}, λ_dep={self.away_lambda:.2f} | "
            f"1={self.prob_home:.1f}% X={self.prob_draw:.1f}% 2={self.prob_away:.1f}% | "
            f"Ü2.5={self.prob_over_25:.1f}% A2.5={self.prob_under_25:.1f}%"
        )


class PoissonModel:
    """
    Poisson dağılımı ile skor tahmini motoru.

    Kullanım:
        model = PoissonModel()
        result = model.predict(
            home_attack=1.3, home_defense=0.9,
            away_attack=1.1, away_defense=1.2,
            league_avg=2.65
        )
    """

    def __init__(self, max_goals: int = POISSON_MAX_GOALS):
        self.max_goals = max_goals

    def predict(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
        league_avg: float = LEAGUE_AVG_GOALS
    ) -> PoissonResult:
        """
        Poisson dağılımı ile maç tahmini.

        Args:
            home_attack:  Ev sahibi atak gücü (> 1 = lig ortalamasının üstü)
            home_defense: Ev sahibi savunma zayıflığı (> 1 = lig ortalamasının üstü)
            away_attack:  Deplasman atak gücü
            away_defense: Deplasman savunma zayıflığı
            league_avg:   Lig maç başına gol ortalaması

        Returns:
            PoissonResult: Tüm olasılıklar ve tahminler
        """
        result = PoissonResult()

        # λ hesaplama
        # Ev sahibi beklenen gol = Ev atak × Dep savunma zayıflığı × (Lig ort / 2) × Ev avantajı
        home_lambda = home_attack * away_defense * (league_avg / 2) * 1.05  # %5 ev avantajı
        away_lambda = away_attack * home_defense * (league_avg / 2) * 0.95

        # Minimum/maksimum sınırlama
        home_lambda = max(0.2, min(home_lambda, 5.0))
        away_lambda = max(0.2, min(away_lambda, 5.0))

        result.home_lambda = home_lambda
        result.away_lambda = away_lambda
        result.expected_home_goals = home_lambda
        result.expected_away_goals = away_lambda
        result.expected_total_goals = home_lambda + away_lambda

        # Skor matrisi hesaplama
        n = self.max_goals + 1
        score_matrix = np.zeros((n, n))

        home_probs = [poisson.pmf(i, home_lambda) for i in range(n)]
        away_probs = [poisson.pmf(j, away_lambda) for j in range(n)]

        for i in range(n):
            for j in range(n):
                score_matrix[i][j] = home_probs[i] * away_probs[j]

        result.score_matrix = score_matrix

        # MS 1/X/2 olasılıkları
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        over_25 = 0.0
        btts_yes = 0.0

        for i in range(n):
            for j in range(n):
                p = score_matrix[i][j]
                if i > j:
                    home_win += p
                elif i == j:
                    draw += p
                else:
                    away_win += p

                if i + j > 2:
                    over_25 += p
                if i > 0 and j > 0:
                    btts_yes += p

        result.prob_home = home_win * 100
        result.prob_draw = draw * 100
        result.prob_away = away_win * 100
        result.prob_over_25 = over_25 * 100
        result.prob_under_25 = (1 - over_25) * 100
        result.prob_btts_yes = btts_yes * 100
        result.prob_btts_no = (1 - btts_yes) * 100

        # En olası skorlar
        scores = []
        for i in range(n):
            for j in range(n):
                scores.append((f"{i}-{j}", score_matrix[i][j] * 100))
        scores.sort(key=lambda x: x[1], reverse=True)
        result.top_scores = scores[:5]

        return result

    def predict_from_features(self, features: Dict[str, float]) -> PoissonResult:
        """
        Feature dict'inden Poisson tahmini üretir.

        v3.1 — Bayesian Smoothing:
        Ham gol istatistiklerini dogrudan kullanmak yerine Bayesian Average
        uygular. Böylece erken sezonda (az maç verisi) tahminler lig
        ortalamasına çekilir, veri arttıkça gerçek takım gücüne yakınsar.

        Formül:
            bayesian_avg = (observed_avg × n + prior × C) / (n + C)
            n   = oynanmış/gözlenmiş maç sayısı
            C   = Bayesian prior sabiti (varsayılan: 5)
            prior = lig ortalaması (takım başına gol)

        Örnek (Hafta 1, takım 5 gol attı):
            raw_avg = 5.0
            bayesian = (5.0 × 1 + 1.325 × 5) / (1 + 5) = 1.94  → Sönümlenmiş!
        Örnek (Hafta 20, takım maç başı 2.5 gol):
            bayesian = (2.5 × 20 + 1.325 × 5) / (20 + 5) = 2.27  → Az sönüm
        """
        avg = LEAGUE_AVG_GOALS / 2  # Maç başına takım başına ortalama gol
        C = float(BAYESIAN_PRIOR_MATCHES)  # Bayesian prior sabiti

        # Son maç gol istatistikleri (ortalama/maç)
        h_scored = features.get('home_recent_goals_scored', 0.0)
        h_conceded = features.get('home_recent_goals_conceded', 0.0)
        a_scored = features.get('away_recent_goals_scored', 0.0)
        a_conceded = features.get('away_recent_goals_conceded', 0.0)

        # Sezon ilerleme bilgisi (v3.1 feature'larından)
        h_played = max(features.get('home_played', 0.0), 0.0)
        a_played = max(features.get('away_played', 0.0), 0.0)
        season_confidence = features.get('season_confidence', 50.0)

        has_recent = (h_scored + h_conceded + a_scored + a_conceded) > 0.01

        if has_recent:
            # ── Bayesian Average: (observed × n + prior × C) / (n + C) ──
            # n = min(played, FORM_WINDOW) çünkü recent stats
            # en fazla FORM_WINDOW maçtan gelir
            n_home = min(h_played, FORM_WINDOW) if h_played > 0 else FORM_WINDOW
            n_away = min(a_played, FORM_WINDOW) if a_played > 0 else FORM_WINDOW

            h_scored_bayes = (h_scored * n_home + avg * C) / (n_home + C)
            h_conceded_bayes = (h_conceded * n_home + avg * C) / (n_home + C)
            a_scored_bayes = (a_scored * n_away + avg * C) / (n_away + C)
            a_conceded_bayes = (a_conceded * n_away + avg * C) / (n_away + C)

            home_attack = max(h_scored_bayes / max(avg, 0.5), 0.3)
            home_defense = max(h_conceded_bayes / max(avg, 0.5), 0.3)
            away_attack = max(a_scored_bayes / max(avg, 0.5), 0.3)
            away_defense = max(a_conceded_bayes / max(avg, 0.5), 0.3)

            logger.debug(
                "Bayesian λ: h_scored %.2f→%.2f, a_scored %.2f→%.2f "
                "(n_home=%.0f, n_away=%.0f, C=%.0f)",
                h_scored, h_scored_bayes, a_scored, a_scored_bayes,
                n_home, n_away, C,
            )
        else:
            # ── Recent veri yok → Standings'den Bayesian tahmin ──
            h_pts = features.get('home_points', 0)
            a_pts = features.get('away_points', 0)
            h_rank = features.get('home_rank', 10)
            a_rank = features.get('away_rank', 10)
            h_gd = features.get('home_goal_diff', 0)
            a_gd = features.get('away_goal_diff', 0)

            # Erken sezonda dampened_rank tercih et (varsa)
            dampened_h_rank = features.get('dampened_home_rank', h_rank)
            dampened_a_rank = features.get('dampened_away_rank', a_rank)

            # Gol farkından Bayesian strength tahmini
            if h_gd != 0 or a_gd != 0:
                eff_h_played = max(h_played, 1)
                eff_a_played = max(a_played, 1)

                # Bayesian: (raw_per_match × played + prior × C) / (played + C)
                h_gd_per_match = h_gd / eff_h_played
                a_gd_per_match = a_gd / eff_a_played

                h_gd_bayes = (h_gd_per_match * eff_h_played + 0.0 * C) / (eff_h_played + C)
                a_gd_bayes = (a_gd_per_match * eff_a_played + 0.0 * C) / (eff_a_played + C)

                home_attack = max(1.0 + h_gd_bayes * 0.5, 0.4)
                home_defense = max(1.0 - h_gd_bayes * 0.3, 0.4)
                away_attack = max(1.0 + a_gd_bayes * 0.5, 0.4)
                away_defense = max(1.0 - a_gd_bayes * 0.3, 0.4)
            else:
                # Sıralama farkından — Bayesian damped sıra kullan
                home_attack = max(1.5 - (dampened_h_rank - 1) * 0.06, 0.5)
                home_defense = max(0.7 + (dampened_h_rank - 1) * 0.04, 0.5)
                away_attack = max(1.5 - (dampened_a_rank - 1) * 0.06, 0.5)
                away_defense = max(0.7 + (dampened_a_rank - 1) * 0.04, 0.5)

            # Puan farkından düzeltme (Bayesian-weighted)
            if h_pts > 0 and a_pts > 0:
                pts_ratio = h_pts / max(a_pts, 1)
                # Erken sezonda puan etkisini azalt
                confidence_factor = season_confidence / 100.0
                pts_adj = min(max((pts_ratio - 1.0) * 0.15 * confidence_factor, -0.2), 0.2)
                home_attack *= (1 + pts_adj)
                away_attack *= (1 - pts_adj)

        # ── Oran bazlı düzeltme (bahis piyasası bilgisi) ──
        # Erken sezonda oran bilgisi daha önemlidir
        ip_home = features.get('implied_prob_home', 33.3)
        ip_away = features.get('implied_prob_away', 33.3)
        if ip_home > 0 and ip_away > 0:
            odds_ratio = ip_home / max(ip_away, 1)
            # Erken sezonda oranların etkisi ARTTIRILIR (2x)
            # Geç sezonda normal ağırlık (1x)
            early_season_boost = 1.0 + (1.0 - season_confidence / 100.0)
            base_adj = min(max((odds_ratio - 1.0) * 0.08, -0.15), 0.15)
            odds_adj = base_adj * early_season_boost
            odds_adj = min(max(odds_adj, -0.25), 0.25)  # Güvenli sınır
            home_attack *= (1 + odds_adj)
            away_attack *= (1 - odds_adj)

        # Form etkisi (± %10 ayarlama, erken sezonda azaltılmış)
        form_diff = features.get('form_diff', 0.0)
        if abs(form_diff) > 5:
            confidence_factor = season_confidence / 100.0
            form_factor = min(max(form_diff / 100, -0.10), 0.10) * confidence_factor
            home_attack *= (1 + form_factor)
            away_attack *= (1 - form_factor)

        # Sakat oyuncu etkisi
        inj_diff = features.get('injury_penalty_diff', 0.0)
        if abs(inj_diff) > 1:
            inj_factor = min(max(inj_diff / 50, -0.15), 0.15)
            home_attack *= (1 + inj_factor)
            away_attack *= (1 - inj_factor)

        return self.predict(
            home_attack=home_attack,
            home_defense=home_defense,
            away_attack=away_attack,
            away_defense=away_defense
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Yardımcı Fonksiyon: Basit istatistikle strength hesaplama
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calculate_strengths_from_standings(
    home_goals_diff_str: Optional[str],
    away_goals_diff_str: Optional[str],
    home_played: int = 10,
    away_played: int = 10,
    league_avg: float = LEAGUE_AVG_GOALS
) -> Tuple[float, float, float, float]:
    """
    Puan tablosundaki A-Y (gol atma-yeme) verisinden Bayesian-smoothed
    strength hesaplar.

    v3.1 — Bayesian Average:
    Ham gol oranlarını dogrudan kullanmak yerine:
        bayesian = (observed × played + prior × C) / (played + C)
    Bu sayede erken sezonda lig ortalamasina cekilir.

    Args:
        home_goals_diff_str: "45-22" formatında
        away_goals_diff_str: "30-35" formatında
        home_played: Ev sahibi oynanan maç sayısı
        away_played: Deplasman oynanan maç sayısı

    Returns:
        (home_attack, home_defense, away_attack, away_defense)
    """
    avg_per_team = league_avg / 2
    C = float(BAYESIAN_PRIOR_MATCHES)

    def parse_goals(s):
        if not s:
            return avg_per_team, avg_per_team
        try:
            parts = str(s).replace(" ", "").split("-")
            scored = int(parts[0])
            conceded = int(parts[1]) if len(parts) > 1 else scored
            return scored, conceded
        except (ValueError, IndexError):
            return avg_per_team, avg_per_team

    h_scored, h_conceded = parse_goals(home_goals_diff_str)
    a_scored, a_conceded = parse_goals(away_goals_diff_str)

    # Bayesian Average: (observed_per_game × played + prior × C) / (played + C)
    h_scored_pg = h_scored / max(home_played, 1)
    h_conceded_pg = h_conceded / max(home_played, 1)
    a_scored_pg = a_scored / max(away_played, 1)
    a_conceded_pg = a_conceded / max(away_played, 1)

    h_scored_bayes = (h_scored_pg * home_played + avg_per_team * C) / (home_played + C)
    h_conceded_bayes = (h_conceded_pg * home_played + avg_per_team * C) / (home_played + C)
    a_scored_bayes = (a_scored_pg * away_played + avg_per_team * C) / (away_played + C)
    a_conceded_bayes = (a_conceded_pg * away_played + avg_per_team * C) / (away_played + C)

    h_att = h_scored_bayes / max(avg_per_team, 0.5)
    h_def = h_conceded_bayes / max(avg_per_team, 0.5)
    a_att = a_scored_bayes / max(avg_per_team, 0.5)
    a_def = a_conceded_bayes / max(avg_per_team, 0.5)

    return (
        max(h_att, 0.3),
        max(h_def, 0.3),
        max(a_att, 0.3),
        max(a_def, 0.3)
    )
