"""
🕷️ Nesine.com Scraper → SQLAlchemy Veritabanı
Mevcut scraper mantığını koruyarak verileri doğrudan DB'ye yazan adaptör.

Çalışma akışı:
  1. Selenium + BeautifulSoup ile veri çek (mevcut NesineScraper)
  2. Her veriyi SQLAlchemy modeline dönüştür
  3. Upsert mantığıyla veritabanına yaz

Bu dosya mevcut nesine_scraper_optimized.py'yi import eder,
verilerini DB'ye aktarır. Scraper kodu değişmez.
"""

import logging
import sys
from typing import Optional

from sqlalchemy.orm import Session

from database import get_session, init_db, get_or_create, upsert
from models import (
    League, Team, Referee, Match, Odds,
    TeamStanding, RecentMatch, H2HMatch,
    RefereeStats as RefereeStatsModel,
    RefereeMatch as RefereeMatchModel,
    Injury,
)

# Mevcut scraper'ı import et
from nesine_scraper_optimized import (
    NesineScraper,
    MatchData, TeamStanding as TSDataclass,
    LastMatch, CompetitionHistory,
    RefereeMatch as RMDataclass, RefereeStats as RSDataclass,
    InjuryData,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Yardımcı Dönüşüm Fonksiyonları
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_float(val) -> Optional[float]:
    """String'i güvenli float'a çevirir."""
    if val is None:
        return None
    try:
        return float(str(val).replace(',', '.'))
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    """String'i güvenli int'e çevirir."""
    if val is None:
        return None
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None


def _pct_to_float(val) -> Optional[float]:
    """Yüzde stringini float'a çevirir."""
    if val is None:
        return None
    try:
        return float(str(val).replace('%', '').replace(',', '.').strip())
    except (ValueError, TypeError):
        return None


def _is_won(val) -> Optional[bool]:
    """'Evet'/'Hayır' → bool."""
    if val is None:
        return None
    return str(val).strip().lower() == 'evet'


def _parse_goals_diff(ay_str: Optional[str]) -> tuple:
    """'45-22' → (goals_diff_str, goal_diff_int)"""
    if not ay_str:
        return None, None
    try:
        parts = str(ay_str).replace(" ", "").split("-")
        if len(parts) == 2:
            scored = int(parts[0])
            conceded = int(parts[1])
            return ay_str, scored - conceded
    except (ValueError, IndexError):
        pass
    return ay_str, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ana Veritabanı Yazıcı Sınıfı
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ScraperDBWriter:
    """
    NesineScraper verilerini SQLAlchemy veritabanına yazan sınıf.

    Kullanım:
        scraper = NesineScraper(match_count=20)
        scraper.run()

        writer = ScraperDBWriter()
        writer.write_all(scraper)
    """

    def __init__(self):
        init_db()

    def write_all(self, scraper: NesineScraper) -> dict:
        """
        Scraper'daki tüm verileri veritabanına yazar.

        Returns:
            Yazılan kayıt sayıları
        """
        stats = {
            'matches': 0,
            'standings': 0,
            'recent_matches': 0,
            'h2h_matches': 0,
            'referee_stats': 0,
            'referee_matches': 0,
            'injuries': 0,
        }

        with get_session() as session:
            # 1. Maçlar + Oranlar
            match_map = {}  # nesine_code → match_id
            for md in scraper.matches:
                match_id = self._write_match(session, md)
                if match_id:
                    match_map[md.Maç_Kodu] = match_id
                    stats['matches'] += 1

            # 2. Puan Tablosu
            for st in scraper.standings:
                if self._write_standing(session, st, match_map):
                    stats['standings'] += 1

            # 3. Son Maçlar
            for lm in scraper.last_matches:
                if self._write_recent_match(session, lm, match_map):
                    stats['recent_matches'] += 1

            # 4. Rekabet Geçmişi (H2H)
            for ch in scraper.competition_history:
                if self._write_h2h(session, ch, match_map):
                    stats['h2h_matches'] += 1

            # 5. Hakem İstatistikleri
            for rs in scraper.referee_stats:
                if self._write_referee_stats(session, rs, match_map):
                    stats['referee_stats'] += 1

            # 6. Hakem Maçları
            for rm in scraper.referee_matches:
                if self._write_referee_match(session, rm, match_map):
                    stats['referee_matches'] += 1

            # 7. Sakat / Cezalı
            for inj in scraper.injury_data:
                if self._write_injury(session, inj, match_map):
                    stats['injuries'] += 1

        logger.info("✓ Tüm veriler veritabanına yazıldı")
        for key, count in stats.items():
            logger.info(f"  {key}: {count} kayıt")

        return stats

    # ─── Alt Yazıcılar ───────────────────────────────────────────

    def _write_match(self, session: Session, md: MatchData) -> Optional[int]:
        """Tek bir maçı veritabanına yazar (upsert)."""
        try:
            if not md.Maç_Kodu or not md.Maç:
                return None

            teams = md.Maç.split(' - ')
            if len(teams) != 2:
                return None

            home_name = teams[0].strip()
            away_name = teams[1].strip()

            # Lig
            league, _ = get_or_create(session, League, name=md.Lig or "Bilinmeyen")

            # Takımlar
            home_team, _ = get_or_create(session, Team, name=home_name)
            away_team, _ = get_or_create(session, Team, name=away_name)

            # Maç (upsert)
            match, created = upsert(
                session, Match,
                filter_kwargs={'nesine_code': str(md.Maç_Kodu)},
                update_kwargs={
                    'league_id': league.id,
                    'home_team_id': home_team.id,
                    'away_team_id': away_team.id,
                    'match_date': md.Tarih,
                    'match_time': md.Saat,
                    'mbs': md.MBS,
                    'stats_link': md.İstatistik_Link,
                    'market_count': _safe_int(md.Market_Sayısı),
                }
            )

            # Oranlar (upsert)
            upsert(
                session, Odds,
                filter_kwargs={'match_id': match.id},
                update_kwargs={
                    'ms_1': _safe_float(md.MS_1),
                    'ms_x': _safe_float(md.MS_X),
                    'ms_2': _safe_float(md.MS_2),
                    'alt_2_5': _safe_float(md.Alt_2_5),
                    'ust_2_5': _safe_float(md.Üst_2_5),
                    'hnd': md.HND,
                    'hnd_1': _safe_float(md.HND_1),
                    'hnd_x': _safe_float(md.HND_X),
                    'hnd_2': _safe_float(md.HND_2),
                    'cs_1x': _safe_float(md.ÇS_1X),
                    'cs_12': _safe_float(md.ÇS_12),
                    'cs_x2': _safe_float(md.ÇS_X2),
                    'kg_var': _safe_float(md.KG_Var),
                    'kg_yok': _safe_float(md.KG_Yok),
                }
            )

            action = "oluşturuldu" if created else "güncellendi"
            logger.debug(f"  Maç {action}: {md.Maç} ({md.Maç_Kodu})")
            return match.id

        except Exception as e:
            logger.error(f"Maç yazma hatası ({md.Maç_Kodu}): {e}")
            return None

    def _write_standing(self, session: Session, st: TSDataclass,
                        match_map: dict) -> bool:
        """Puan tablosu kaydı yazar."""
        try:
            match_id = match_map.get(st.Maç_Kodu)
            if not match_id or not st.Takım:
                return False

            team, _ = get_or_create(session, Team, name=st.Takım)
            goals_diff_str, goal_diff = _parse_goals_diff(st.A_Y)

            upsert(
                session, TeamStanding,
                filter_kwargs={'match_id': match_id, 'team_type': st.Takım_Tipi},
                update_kwargs={
                    'team_id': team.id,
                    'rank': _safe_int(st.Sıra),
                    'played': _safe_int(st.O),
                    'won': _safe_int(st.G),
                    'drawn': _safe_int(st.B),
                    'lost': _safe_int(st.M),
                    'goals_diff_str': goals_diff_str,
                    'goal_diff': _safe_int(st.AV) or goal_diff,
                    'points': _safe_int(st.P),
                    'form': st.Form,
                }
            )
            return True

        except Exception as e:
            logger.error(f"Puan tablosu yazma hatası: {e}")
            return False

    def _write_recent_match(self, session: Session, lm: LastMatch,
                            match_map: dict) -> bool:
        """Son maç kaydı yazar."""
        try:
            match_id = match_map.get(lm.Maç_Kodu)
            if not match_id:
                return False

            team, _ = get_or_create(session, Team, name=lm.Takım or "Bilinmeyen")

            rm = RecentMatch(
                match_id=match_id,
                team_id=team.id,
                team_type=lm.Takım_Tipi,
                league=lm.Lig,
                date=lm.Tarih,
                home_team_name=lm.Ev_Sahibi,
                away_team_name=lm.Deplasman,
                score=lm.MS,
                ht_score=lm.İY,
                result=lm.Sonuç,
            )
            session.add(rm)
            return True

        except Exception as e:
            logger.error(f"Son maç yazma hatası: {e}")
            return False

    def _write_h2h(self, session: Session, ch: CompetitionHistory,
                   match_map: dict) -> bool:
        """H2H maç kaydı yazar."""
        try:
            match_id = match_map.get(ch.Maç_Kodu)
            if not match_id:
                return False

            h2h = H2HMatch(
                match_id=match_id,
                league=ch.Lig,
                date=ch.Tarih,
                home_team_name=ch.Ev_Sahibi,
                away_team_name=ch.Deplasman,
                score=ch.MS,
                ht_score=ch.İY,
                odd_1=_safe_float(ch.Oran_1),
                odd_1_won=_is_won(ch.Oran_1_Geldi),
                odd_x=_safe_float(ch.Oran_X),
                odd_x_won=_is_won(ch.Oran_X_Geldi),
                odd_2=_safe_float(ch.Oran_2),
                odd_2_won=_is_won(ch.Oran_2_Geldi),
                odd_alt=_safe_float(ch.Oran_Alt),
                odd_alt_won=_is_won(ch.Oran_Alt_Geldi),
                odd_ust=_safe_float(ch.Oran_Üst),
                odd_ust_won=_is_won(ch.Oran_Üst_Geldi),
            )
            session.add(h2h)
            return True

        except Exception as e:
            logger.error(f"H2H yazma hatası: {e}")
            return False

    def _write_referee_stats(self, session: Session, rs: RSDataclass,
                             match_map: dict) -> bool:
        """Hakem istatistik kaydı yazar."""
        try:
            match_id = match_map.get(rs.Maç_Kodu)
            if not match_id or not rs.Hakem_Adı:
                return False

            referee, _ = get_or_create(session, Referee, name=rs.Hakem_Adı)

            upsert(
                session, RefereeStatsModel,
                filter_kwargs={'match_id': match_id, 'referee_id': referee.id},
                update_kwargs={
                    'ms1_count': _safe_int(rs.MS1_Sayı),
                    'ms1_pct': _pct_to_float(rs.MS1_Yüzde),
                    'msx_count': _safe_int(rs.MSX_Sayı),
                    'msx_pct': _pct_to_float(rs.MSX_Yüzde),
                    'ms2_count': _safe_int(rs.MS2_Sayı),
                    'ms2_pct': _pct_to_float(rs.MS2_Yüzde),
                    'alt_count': _safe_int(rs.Alt_2_5_Sayı),
                    'alt_pct': _pct_to_float(rs.Alt_2_5_Yüzde),
                    'ust_count': _safe_int(rs.Üst_2_5_Sayı),
                    'ust_pct': _pct_to_float(rs.Üst_2_5_Yüzde),
                    'kg_var_count': _safe_int(rs.KG_Var_Sayı),
                    'kg_var_pct': _pct_to_float(rs.KG_Var_Yüzde),
                    'kg_yok_count': _safe_int(rs.KG_Yok_Sayı),
                    'kg_yok_pct': _pct_to_float(rs.KG_Yok_Yüzde),
                }
            )
            return True

        except Exception as e:
            logger.error(f"Hakem istatistik yazma hatası: {e}")
            return False

    def _write_referee_match(self, session: Session, rm: RMDataclass,
                             match_map: dict) -> bool:
        """Hakem geçmiş maç kaydı yazar."""
        try:
            match_id = match_map.get(rm.Maç_Kodu)
            if not match_id or not rm.Hakem_Adı:
                return False

            referee, _ = get_or_create(session, Referee, name=rm.Hakem_Adı)

            ref_match = RefereeMatchModel(
                match_id=match_id,
                referee_id=referee.id,
                league=rm.Lig,
                date=rm.Tarih,
                home_team_name=rm.Ev_Sahibi,
                away_team_name=rm.Deplasman,
                score=rm.MS,
                ht_score=rm.İY,
                odd_1=_safe_float(rm.Oran_1),
                odd_1_won=_is_won(rm.Oran_1_Geldi),
                odd_x=_safe_float(rm.Oran_X),
                odd_x_won=_is_won(rm.Oran_X_Geldi),
                odd_2=_safe_float(rm.Oran_2),
                odd_2_won=_is_won(rm.Oran_2_Geldi),
                odd_alt=_safe_float(rm.Oran_Alt),
                odd_alt_won=_is_won(rm.Oran_Alt_Geldi),
                odd_ust=_safe_float(rm.Oran_Üst),
                odd_ust_won=_is_won(rm.Oran_Üst_Geldi),
            )
            session.add(ref_match)
            return True

        except Exception as e:
            logger.error(f"Hakem maç yazma hatası: {e}")
            return False

    def _write_injury(self, session: Session, inj: InjuryData,
                      match_map: dict) -> bool:
        """Sakat/cezalı oyuncu kaydı yazar."""
        try:
            match_id = match_map.get(inj.Maç_Kodu)
            if not match_id or not inj.Oyuncu:
                return False

            team, _ = get_or_create(session, Team, name=inj.Takım or "Bilinmeyen")

            injury = Injury(
                match_id=match_id,
                team_id=team.id,
                number=_safe_int(inj.Numara),
                player_name=inj.Oyuncu,
                age=_safe_int(inj.Yaş),
                position=inj.Pozisyon,
                matches_played=_safe_int(inj.Maç_Sayısı) or 0,
                starts=_safe_int(inj.İlk_11) or 0,
                goals=_safe_int(inj.Gol) or 0,
                assists=_safe_int(inj.Asist) or 0,
                status=inj.Durum,
                description=inj.Açıklama,
            )
            session.add(injury)
            return True

        except Exception as e:
            logger.error(f"Sakat/cezalı yazma hatası: {e}")
            return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Bağımsız çalıştırma
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_scraper_to_db(match_count: int = 20) -> dict:
    """
    Scraper'ı çalıştırır ve verileri veritabanına yazar.

    Args:
        match_count: Çekilecek maç sayısı

    Returns:
        Yazılan kayıt istatistikleri
    """
    logger.info(f"🕷️ Scraper başlatılıyor ({match_count} maç)...")

    scraper = NesineScraper(match_count=match_count)
    scraper.run()

    logger.info("📥 Veriler veritabanına aktarılıyor...")
    writer = ScraperDBWriter()
    stats = writer.write_all(scraper)

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_scraper_to_db(count)
