"""
🗄️ Veritabanı Yöneticisi
SQLAlchemy engine, session fabrikası ve upsert yardımcıları.
PostgreSQL (production) ve SQLite (geliştirme) desteği.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session

from config import DATABASE_URL, DB_ENGINE
from models import Base

logger = logging.getLogger(__name__)

# ─── Engine & Session ─────────────────────────────────────────────
_engine_kwargs: dict = {
    "echo": False,
    "pool_pre_ping": True,
}

if DB_ENGINE == "sqlite":
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL connection pool ayarları
    _engine_kwargs["pool_size"] = 10
    _engine_kwargs["max_overflow"] = 20
    _engine_kwargs["pool_recycle"] = 3600

engine = create_engine(DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Tüm tabloları oluşturur (varsa dokunmaz) + migration."""
    Base.metadata.create_all(bind=engine)
    _run_migrations()
    logger.info("✓ Veritabanı tabloları hazır (%s)", DB_ENGINE)


def _run_migrations() -> None:
    """Mevcut tablolara eksik kolonları ekler (ALTER TABLE).
    
    PostgreSQL ve SQLite için uyumlu migration mantığı.
    """
    if DB_ENGINE == "sqlite":
        _run_sqlite_migrations()
    else:
        _run_pg_migrations()


def _run_sqlite_migrations() -> None:
    """SQLite için migration."""
    import sqlite3
    from config import DB_PATH

    migrations = [
        ("predictions", "shap_summary", "TEXT"),
        ("predictions", "top_scores", "TEXT"),
    ]

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    for table, column, col_type in migrations:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            logger.info("Migration: %s.%s eklendi", table, column)
        except sqlite3.OperationalError:
            pass  # Kolon zaten mevcut
    conn.commit()
    conn.close()


def _run_pg_migrations() -> None:
    """PostgreSQL için migration."""
    migrations = [
        ("predictions", "shap_summary", "JSONB"),
        ("predictions", "top_scores", "JSONB"),
    ]

    with engine.connect() as conn:
        for table, column, col_type in migrations:
            try:
                # Kolon var mı kontrol et
                result = conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = :table AND column_name = :column"
                ), {"table": table, "column": column})
                if result.fetchone() is None:
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                    ))
                    conn.commit()
                    logger.info("Migration: %s.%s eklendi (%s)", table, column, col_type)
            except Exception as e:
                logger.debug("Migration atlandı (%s.%s): %s", table, column, e)


def drop_db() -> None:
    """Tüm tabloları siler (dikkat!)."""
    Base.metadata.drop_all(bind=engine)
    logger.info("⚠ Tüm tablolar silindi")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context-manager ile güvenli session kullanımı.

    Kullanım:
        with get_session() as session:
            session.add(obj)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─── Upsert Yardımcıları ─────────────────────────────────────────
def get_or_create(session: Session, model, defaults: Optional[dict] = None, **kwargs):
    """
    Verilen filtre ile kayıt arar; yoksa oluşturur.
    
    Returns:
        (instance, created: bool)
    """
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False

    params = {**kwargs, **(defaults or {})}
    instance = model(**params)
    session.add(instance)
    session.flush()  # ID ataması için
    return instance, True


def upsert(session: Session, model, filter_kwargs: dict, update_kwargs: dict):
    """
    Kayıt varsa günceller, yoksa oluşturur (Update on Conflict).
    
    Args:
        session: SQLAlchemy session
        model: ORM model sınıfı
        filter_kwargs: Eşleşme filtresi (örn: {'nesine_code': '123'})
        update_kwargs: Güncellenecek alanlar

    Returns:
        (instance, created: bool)
    """
    instance = session.query(model).filter_by(**filter_kwargs).first()

    if instance:
        for key, value in update_kwargs.items():
            setattr(instance, key, value)
        session.flush()
        return instance, False
    else:
        params = {**filter_kwargs, **update_kwargs}
        instance = model(**params)
        session.add(instance)
        session.flush()
        return instance, True


def db_stats(session: Session) -> dict:
    """Veritabanındaki tablo başına kayıt sayısını döndürür."""
    inspector = inspect(engine)
    stats = {}
    for table_name in inspector.get_table_names():
        count = session.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        ).scalar()
        stats[table_name] = count
    return stats


# ─── Active Learning Sorguları ────────────────────────────────────

def get_pending_predictions(session: Session) -> list:
    """
    Tarihi geçmiş ama sonucu (actual_result) girilmemiş tahminleri döndürür.

    Kullanım Amacı:
        Active Learning döngüsünde, kullanıcının henüz doğrulamadığı
        geçmiş maçları bulmak ve CLI aracılığıyla sonuç girmesini sağlamak.

    Mantık:
        - predictions tablosunda actual_result IS NULL olan kayıtları bul
        - İlişkili maçın tarihi bugünden önce olmalı VEYA is_finished == True
        - Match ve Team ilişkilerini eagerly yükle (display_name için)

    Returns:
        list[Prediction]: Doğrulanmayı bekleyen tahmin kayıtları
    """
    from datetime import datetime
    from models import Match, Prediction

    # actual_result henüz girilmemiş tüm tahminleri çek
    pending = (
        session.query(Prediction)
        .join(Match, Prediction.match_id == Match.id)
        .filter(Prediction.actual_result.is_(None))
        .order_by(Prediction.created_at.asc())
        .all()
    )

    return pending


def update_match_result(
    session: Session,
    match_id: int,
    home_score: int,
    away_score: int,
) -> str:
    """
    Bir maçın skorunu ve ilişkili tahminlerin doğrulama alanlarını günceller.

    Args:
        session: SQLAlchemy session
        match_id: Maç ID'si
        home_score: Ev sahibi gol sayısı
        away_score: Deplasman gol sayısı

    Returns:
        str: Belirlenen sonuç ("1", "X", "2")
    """
    from models import Match, Prediction

    # 1) Maç kaydını güncelle
    match = session.get(Match, match_id)
    if not match:
        raise ValueError(f"Match ID {match_id} bulunamadı!")

    match.home_score = home_score
    match.away_score = away_score
    match.is_finished = True

    # Sonucu belirle
    if home_score > away_score:
        result = "1"
    elif home_score < away_score:
        result = "2"
    else:
        result = "X"

    # 2) Bu maça ait tüm tahminleri doğrula
    predictions = (
        session.query(Prediction)
        .filter(Prediction.match_id == match_id)
        .all()
    )

    for pred in predictions:
        pred.actual_result = result
        pred.is_correct = (pred.prediction == result)

    session.flush()
    return result
