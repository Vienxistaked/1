"""
🗄️ SQLAlchemy ORM Modelleri
Normalize edilmiş ilişkisel veritabanı şeması.

Tablolar:
  leagues       – Lig bilgileri
  teams         – Takım bilgileri
  referees      – Hakem bilgileri
  matches       – Maç verileri (ana tablo)
  odds          – Bahis oranları (maça bağlı)
  team_standings – Puan tablosu (maça bağlı)
  recent_matches – Takımların son maçları
  h2h_matches   – Kafa kafaya geçmiş maçlar
  referee_stats – Hakem istatistikleri
  referee_matches – Hakemin yönettiği geçmiş maçlar
  injuries      – Sakat/cezalı oyuncular
  predictions   – Model tahminleri (geçmiş kayıt)
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, Boolean,
    ForeignKey, UniqueConstraint, Index, create_engine
)
from sqlalchemy.orm import declarative_base, relationship

# JSONB desteği: PostgreSQL'de JSONB, SQLite'da JSON kullanır
try:
    from sqlalchemy.dialects.postgresql import JSONB
except ImportError:
    JSONB = None  # type: ignore[assignment, misc]

from sqlalchemy.types import JSON


def _get_json_type():
    """Veritabanı motoruna göre JSONB veya JSON tipini döndürür."""
    from config import DB_ENGINE
    if DB_ENGINE == "postgresql" and JSONB is not None:
        return JSONB
    return JSON


Base = declarative_base()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LİGLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class League(Base):
    __tablename__ = "leagues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False, index=True)
    country = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    # İlişkiler
    matches = relationship("Match", back_populates="league")

    def __repr__(self):
        return f"<League(name='{self.name}')>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAKIMLAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # İlişkiler
    home_matches = relationship("Match", back_populates="home_team",
                                foreign_keys="Match.home_team_id")
    away_matches = relationship("Match", back_populates="away_team",
                                foreign_keys="Match.away_team_id")
    standings = relationship("TeamStanding", back_populates="team")
    recent_matches = relationship("RecentMatch", back_populates="team")
    injuries = relationship("Injury", back_populates="team")

    def __repr__(self):
        return f"<Team(name='{self.name}')>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HAKEMLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Referee(Base):
    __tablename__ = "referees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # İlişkiler
    matches = relationship("Match", back_populates="referee")
    stats = relationship("RefereeStats", back_populates="referee")
    referee_matches = relationship("RefereeMatch", back_populates="referee")

    def __repr__(self):
        return f"<Referee(name='{self.name}')>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAÇLAR (Ana Tablo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nesine_code = Column(String(20), unique=True, nullable=False, index=True)

    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    referee_id = Column(Integer, ForeignKey("referees.id"), nullable=True)

    match_date = Column(String(50))  # "Bugün", "Yarın" vb.
    match_time = Column(String(10))
    mbs = Column(String(5))
    stats_link = Column(String(500))
    market_count = Column(Integer)

    # Sonuç alanları (maç bittikten sonra doldurulur)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    ht_home_score = Column(Integer, nullable=True)
    ht_away_score = Column(Integer, nullable=True)
    is_finished = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # İlişkiler
    league = relationship("League", back_populates="matches")
    home_team = relationship("Team", back_populates="home_matches",
                             foreign_keys=[home_team_id])
    away_team = relationship("Team", back_populates="away_matches",
                             foreign_keys=[away_team_id])
    referee = relationship("Referee", back_populates="matches")
    odds = relationship("Odds", back_populates="match", uselist=False,
                        cascade="all, delete-orphan")
    home_standing = relationship(
        "TeamStanding", back_populates="match",
        primaryjoin="and_(Match.id==TeamStanding.match_id, "
                    "TeamStanding.team_type=='Ev Sahibi')",
        foreign_keys="TeamStanding.match_id",
        viewonly=True, uselist=False
    )
    away_standing = relationship(
        "TeamStanding", back_populates="match",
        primaryjoin="and_(Match.id==TeamStanding.match_id, "
                    "TeamStanding.team_type=='Deplasman')",
        foreign_keys="TeamStanding.match_id",
        viewonly=True, uselist=False
    )
    standings = relationship("TeamStanding", back_populates="match_rel",
                             foreign_keys="TeamStanding.match_id",
                             cascade="all, delete-orphan")
    h2h_matches = relationship("H2HMatch", back_populates="match",
                               cascade="all, delete-orphan")
    injuries = relationship("Injury", back_populates="match",
                            cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="match",
                               cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Match(code='{self.nesine_code}')>"

    @property
    def display_name(self) -> str:
        home = self.home_team.name if self.home_team else "?"
        away = self.away_team.name if self.away_team else "?"
        return f"{home} - {away}"

    @property
    def result(self) -> str | None:
        """Maç sonucunu döndürür: '1', 'X', '2' veya None"""
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "1"
        elif self.home_score < self.away_score:
            return "2"
        return "X"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ORANLAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Odds(Base):
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), unique=True, nullable=False)

    ms_1 = Column(Float)
    ms_x = Column(Float)
    ms_2 = Column(Float)

    alt_2_5 = Column(Float)
    ust_2_5 = Column(Float)

    hnd = Column(String(10))
    hnd_1 = Column(Float)
    hnd_x = Column(Float)
    hnd_2 = Column(Float)

    cs_1x = Column(Float)
    cs_12 = Column(Float)
    cs_x2 = Column(Float)

    kg_var = Column(Float)
    kg_yok = Column(Float)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    match = relationship("Match", back_populates="odds")

    def __repr__(self):
        return f"<Odds(match_id={self.match_id}, 1={self.ms_1}, X={self.ms_x}, 2={self.ms_2})>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PUAN TABLOSU (takım, maç başına)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TeamStanding(Base):
    __tablename__ = "team_standings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    team_type = Column(String(20))  # "Ev Sahibi" | "Deplasman"

    rank = Column(Integer)
    played = Column(Integer)
    won = Column(Integer)
    drawn = Column(Integer)
    lost = Column(Integer)
    goals_diff_str = Column(String(20))  # "45-22" gibi
    goal_diff = Column(Integer)          # Averaj
    points = Column(Integer)
    form = Column(String(20))            # "GGMBM"

    __table_args__ = (
        UniqueConstraint("match_id", "team_type", name="uq_standing_match_type"),
    )

    match_rel = relationship("Match", back_populates="standings",
                             foreign_keys=[match_id])
    match = relationship("Match", viewonly=True, foreign_keys=[match_id])
    team = relationship("Team", back_populates="standings")

    def __repr__(self):
        return f"<Standing(team_type='{self.team_type}', rank={self.rank})>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SON MAÇLAR (takım bazlı)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RecentMatch(Base):
    __tablename__ = "recent_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    team_type = Column(String(20))  # "Ev Sahibi" | "Deplasman"

    league = Column(String(200))
    date = Column(String(50))
    home_team_name = Column(String(200))
    away_team_name = Column(String(200))
    score = Column(String(20))
    ht_score = Column(String(20))
    result = Column(String(20))  # "Galibiyet", "Mağlubiyet", "Beraberlik"

    match_rel = relationship("Match", foreign_keys=[match_id])
    team = relationship("Team", back_populates="recent_matches")

    def __repr__(self):
        return f"<RecentMatch(team='{self.team_type}', result='{self.result}')>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  REKABET GEÇMİŞİ (H2H)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class H2HMatch(Base):
    __tablename__ = "h2h_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)

    league = Column(String(200))
    date = Column(String(50))
    home_team_name = Column(String(200))
    away_team_name = Column(String(200))
    score = Column(String(20))
    ht_score = Column(String(20))

    odd_1 = Column(Float)
    odd_1_won = Column(Boolean)
    odd_x = Column(Float)
    odd_x_won = Column(Boolean)
    odd_2 = Column(Float)
    odd_2_won = Column(Boolean)
    odd_alt = Column(Float)
    odd_alt_won = Column(Boolean)
    odd_ust = Column(Float)
    odd_ust_won = Column(Boolean)

    match = relationship("Match", back_populates="h2h_matches")

    def __repr__(self):
        return f"<H2H({self.home_team_name} vs {self.away_team_name})>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HAKEM İSTATİSTİKLERİ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RefereeStats(Base):
    __tablename__ = "referee_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    referee_id = Column(Integer, ForeignKey("referees.id"), nullable=False)

    ms1_count = Column(Integer)
    ms1_pct = Column(Float)
    msx_count = Column(Integer)
    msx_pct = Column(Float)
    ms2_count = Column(Integer)
    ms2_pct = Column(Float)

    alt_count = Column(Integer)
    alt_pct = Column(Float)
    ust_count = Column(Integer)
    ust_pct = Column(Float)

    kg_var_count = Column(Integer)
    kg_var_pct = Column(Float)
    kg_yok_count = Column(Integer)
    kg_yok_pct = Column(Float)

    __table_args__ = (
        UniqueConstraint("match_id", "referee_id", name="uq_ref_stats_match"),
    )

    referee = relationship("Referee", back_populates="stats")

    def __repr__(self):
        return f"<RefereeStats(referee_id={self.referee_id})>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HAKEM GEÇMİŞ MAÇLARI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RefereeMatch(Base):
    __tablename__ = "referee_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    referee_id = Column(Integer, ForeignKey("referees.id"), nullable=False)

    league = Column(String(200))
    date = Column(String(50))
    home_team_name = Column(String(200))
    away_team_name = Column(String(200))
    score = Column(String(20))
    ht_score = Column(String(20))

    odd_1 = Column(Float)
    odd_1_won = Column(Boolean)
    odd_x = Column(Float)
    odd_x_won = Column(Boolean)
    odd_2 = Column(Float)
    odd_2_won = Column(Boolean)
    odd_alt = Column(Float)
    odd_alt_won = Column(Boolean)
    odd_ust = Column(Float)
    odd_ust_won = Column(Boolean)

    referee = relationship("Referee", back_populates="referee_matches")

    def __repr__(self):
        return f"<RefereeMatch(referee_id={self.referee_id})>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAKAT / CEZALI OYUNCULAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Injury(Base):
    __tablename__ = "injuries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    number = Column(Integer)
    player_name = Column(String(200), nullable=False)
    age = Column(Integer)
    position = Column(String(50))

    matches_played = Column(Integer, default=0)
    starts = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)

    status = Column(String(50))    # "Sakatlık" | "Cezalı"
    description = Column(Text)

    match = relationship("Match", back_populates="injuries")
    team = relationship("Team", back_populates="injuries")

    @property
    def importance_score(self) -> float:
        """Oyuncunun takım için önem puanı (0-20 arası)."""
        score = 0.0
        if self.starts and self.starts >= 15:
            score += 10
        elif self.starts and self.starts >= 10:
            score += 7
        elif self.starts and self.starts >= 5:
            score += 4
        else:
            score += 1

        contrib = (self.goals or 0) + (self.assists or 0)
        if contrib >= 10:
            score += 5
        elif contrib >= 5:
            score += 3
        elif contrib >= 2:
            score += 1

        pos = (self.position or "").lower()
        if pos in ("forvet", "santrafor"):
            score *= 1.2
        elif pos in ("orta saha", "ortasaha"):
            score *= 1.1
        elif pos == "kaleci":
            score *= 1.5

        return round(score, 2)

    def __repr__(self):
        return f"<Injury(player='{self.player_name}', status='{self.status}')>"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAHMİNLER (model çıktıları, geçmiş takibi)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)

    model_version = Column(String(50))   # "poisson_v1", "xgboost_v3", …
    engine_used = Column(String(20))     # "poisson", "ml", "fallback"

    # Olasılıklar
    prob_home = Column(Float)
    prob_draw = Column(Float)
    prob_away = Column(Float)
    prob_over_25 = Column(Float)
    prob_under_25 = Column(Float)

    # Poisson beklentileri
    expected_home_goals = Column(Float)
    expected_away_goals = Column(Float)

    # Tahmin ve Value
    prediction = Column(String(5))       # "1", "X", "2"
    confidence = Column(Float)
    value_edge = Column(Float)
    is_value_bet = Column(Boolean, default=False)
    risk_level = Column(String(30))

    explanation = Column(Text)

    # v3.0: SHAP açıklama özeti (PostgreSQL'de JSONB, SQLite'da JSON)
    shap_summary = Column(_get_json_type(), nullable=True)

    # v4.0: En olası skorlar (JSONB) — [("1-0", 12.5), ...] formatinda
    top_scores = Column(_get_json_type(), nullable=True)

    # Doğrulama (maç sonrası)
    actual_result = Column(String(5), nullable=True)
    is_correct = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction(match_id={self.match_id}, pred='{self.prediction}', conf={self.confidence})>"
