"""
🚀 Nesine Futbol Tahmin Sistemi — Active Learning Orchestrator

Human-in-the-Loop Aktif Öğrenme döngüsü ile çalışan ana pipeline.

Akış:
  1. Pending Review  → Doğrulanmamış geçmiş tahminleri bul
  2. Interactive CLI  → Kullanıcıdan maç sonuçlarını al
  3. Online Retrain   → Model yeni verilerle yeniden eğitilir
  4. Scrape & Predict → Yeni maçlar çekilir, tahmin yapılır, rapor üretilir

Kullanım:
  python main.py                  # Tam Active Learning pipeline
  python main.py --analyze        # Sadece analiz (mevcut DB verisiyle)
  python main.py --scrape 30      # Sadece scrape (30 maç)
  python main.py --retrain        # ML modelini yeniden eğit
  python main.py --validate       # Geçmiş tahminleri doğrula
  python main.py --stats          # Veritabanı istatistikleri
  python main.py --review         # Sadece pending review (sonuç girişi)
"""

import argparse
import csv
import logging
import os
import re
import sys
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from config import (
    LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT,
    MAX_REPORT_MATCHES, VALUE_BET_MIN_CONFIDENCE,
    BASE_DIR,
)
from database import (
    init_db, get_session, db_stats,
    get_pending_predictions, update_match_result,
)
from models import Match, Odds, Prediction
from predictor import MatchPredictor, PredictionResult
from scraper_db import run_scraper_to_db
from feature_engineering import _resolve_match_datetime

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Loglama Ayarları
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                BASE_DIR / "logs" / f"nesine_{datetime.now():%Y%m%d}.log",
                encoding='utf-8'
            ),
        ]
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Kullanıcı Girişi Doğrulama Fonksiyonları
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Skor formatı regex: "2-1", "0-0", "3 - 2" vb.
SCORE_REGEX = re.compile(r"^\s*(\d{1,2})\s*[-–]\s*(\d{1,2})\s*$")

# Direkt sonuç formatı: "1", "X", "x", "2"
RESULT_REGEX = re.compile(r"^\s*([1Xx2])\s*$")


def parse_score_input(raw: str) -> Optional[Tuple[int, int]]:
    """
    Kullanıcının girdiği skor stringini parse eder.

    Kabul edilen formatlar:
      - "2-1"  → (2, 1)
      - "0 - 3" → (0, 3)

    Returns:
        (home_score, away_score) tuple veya None (geçersiz format)
    """
    match = SCORE_REGEX.match(raw)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def parse_result_input(raw: str) -> Optional[Tuple[int, int, str]]:
    """
    Kullanıcının girdiği direkt sonucu ("1", "X", "2") parse eder.

    NOT: Direkt sonuç girildiğinde varsayılan skorlar atanır:
      - "1" → (1, 0) — ev sahibi kazandı
      - "X" → (0, 0) — berabere
      - "2" → (0, 1) — deplasman kazandı

    Returns:
        (home_score, away_score, result) tuple veya None
    """
    match = RESULT_REGEX.match(raw)
    if match:
        result = match.group(1).upper()
        # Varsayılan skorlar (tam skor bilinmediğinde)
        if result == "1":
            return 1, 0, "1"
        elif result == "X":
            return 0, 0, "X"
        elif result == "2":
            return 0, 1, "2"
    return None


def get_match_result_from_user(match_display: str) -> Optional[Tuple[int, int]]:
    """
    Kullanıcıdan bir maç sonucunu etkileşimli olarak alır.

    Sağlam giriş doğrulama döngüsü:
      - Skor formatı (2-1)
      - Direkt sonuç (1/X/2)
      - Pas geç (p/pas/skip)     → None döner (bu maç atlanır)
      - Geçersiz giriş → tekrar sorar

    Args:
        match_display: Maç adı (ör: "Galatasaray - Fenerbahçe")

    Returns:
        (home_score, away_score) veya None (pas geçildi)
    """
    print(f"\n  📌 {match_display}")
    print(f"     Format: Skor (ör: 2-1) veya Sonuç (1/X/2) veya Pas geç (p)")

    while True:
        raw = input("     ➤ Sonuç: ").strip()

        # Pas geç kontrolü
        if raw.lower() in ("p", "pas", "skip", "geç", "gec", ""):
            print("     ⏭️  Pas geçildi")
            return None

        # Skor formatı dene (ör: 2-1)
        score = parse_score_input(raw)
        if score is not None:
            home, away = score
            if home > away:
                result_str = "1"
            elif home < away:
                result_str = "2"
            else:
                result_str = "X"
            print(f"     ✅ Skor: {home}-{away} → MS{result_str}")
            return score

        # Direkt sonuç dene (1/X/2)
        result = parse_result_input(raw)
        if result is not None:
            home, away, res = result
            print(f"     ✅ Sonuç: MS{res} (varsayılan skor: {home}-{away})")
            return home, away

        # Geçersiz giriş
        print("     ❌ Geçersiz format! Lütfen şu formatlardan birini kullanın:")
        print("        Skor: 2-1, 0-0, 3-2")
        print("        Sonuç: 1, X, 2")
        print("        Pas geç: p")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ADIM 1: Pending Review — Bekleyen Tahmin Kontrolü
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_pending_review(session: Session) -> int:
    """
    Veritabanında doğrulanmamış geçmiş tahminleri bulur ve
    kullanıcıdan etkileşimli olarak sonuç girmesini ister.

    Akış:
      1. get_pending_predictions() ile bekleyen tahminleri sorgula
      2. Her maç için kullanıcıya sor
      3. Girilen skoru hem matches hem predictions tablolarına yaz
      4. Güncellenen maç sayısını döndür

    Returns:
        int: Kullanıcı tarafından sonucu girilen maç sayısı
    """
    print("=" * 80)
    print("📋 ADIM 1: BEKLEYEN TAHMİN KONTROLÜ (Pending Review)")
    print("=" * 80)
    print()

    pending = get_pending_predictions(session)

    if not pending:
        print("✅ Doğrulanmayı bekleyen tahmin yok — devam ediliyor.")
        print()
        return 0

    # Benzersiz maçları çıkar (bir maçta birden fazla tahmin olabilir)
    seen_match_ids = set()
    unique_pending = []
    for pred in pending:
        if pred.match_id not in seen_match_ids:
            seen_match_ids.add(pred.match_id)
            unique_pending.append(pred)

    print(f"⚠️  {len(unique_pending)} maçın sonucu henüz girilmemiş.")
    print(f"   Her maç için skor veya sonuç girmeniz isteniyor.")
    print(f"   Bilmediğiniz maçları 'p' ile pas geçebilirsiniz.")
    print("-" * 80)

    updated_count = 0

    for idx, pred in enumerate(unique_pending, 1):
        # Maç bilgilerini al
        match = session.get(Match, pred.match_id)
        if not match:
            logger.warning("Match ID %d bulunamadı, atlanıyor", pred.match_id)
            continue

        # Maç zaten skor girilmişse atla (başka bir yerden güncellenmiş olabilir)
        if match.home_score is not None and match.away_score is not None:
            # Maçın skoru var ama tahminler doğrulanmamış → otomatik doğrula
            result = match.result
            if result:
                for p in session.query(Prediction).filter_by(match_id=match.id).all():
                    if p.actual_result is None:
                        p.actual_result = result
                        p.is_correct = (p.prediction == result)
                updated_count += 1
                print(f"\n  🔄 [{idx}/{len(unique_pending)}] {match.display_name} — "
                      f"Skor zaten mevcut: {match.home_score}-{match.away_score} "
                      f"(MS{result}) → otomatik doğrulandı")
                continue

        # Tahmin bilgisini göster
        display = match.display_name
        pred_info = f"Tahmin: MS{pred.prediction} (%{pred.confidence:.1f})"
        date_info = f"{match.match_date or '?'} {match.match_time or ''}"

        print(f"\n  [{idx}/{len(unique_pending)}] 📅 {date_info} | {pred_info}")

        # Kullanıcıdan sonuç al
        score = get_match_result_from_user(display)

        if score is None:
            # Pas geçildi
            continue

        home_score, away_score = score

        try:
            # Maç ve tahmin kayıtlarını güncelle
            actual_result = update_match_result(
                session, match.id, home_score, away_score
            )
            updated_count += 1

            # Tahmin doğru muydu kontrol?
            was_correct = (pred.prediction == actual_result)
            emoji = "✅" if was_correct else "❌"
            print(f"     {emoji} Tahmin MS{pred.prediction} → Gerçek MS{actual_result} "
                  f"({'DOĞRU' if was_correct else 'YANLIŞ'})")

        except Exception as e:
            logger.error("Maç güncelleme hatası (ID: %d): %s", match.id, e)
            print(f"     ⚠️  Güncelleme hatası: {e}")

    # Tüm değişiklikleri commit et
    session.commit()

    print()
    print("-" * 80)
    print(f"📊 Sonuç: {updated_count}/{len(unique_pending)} maç sonucu güncellendi.")
    print()

    return updated_count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ADIM 2: Online Retraining — Model Yeniden Eğitimi
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_retrain(session: Session) -> bool:
    """
    Son verilerle ML modelini yeniden eğitir.

    Bu adım, kullanıcının girdiği yeni maç sonuçlarını öğrenmek için
    MatchPredictor.retrain() metodunu çağırır. Model yeni ağırlıklarıyla
    diske kaydedilir (.pkl).

    Returns:
        bool: Yeniden eğitim başarılı mı
    """
    print("=" * 80)
    print("🔧 ADIM 2: MODEL YENİDEN EĞİTİLİYOR (Online Retraining)")
    print("=" * 80)
    print()

    try:
        predictor = MatchPredictor(session)
        result = predictor.retrain()
        print(f"✅ Yeniden eğitim tamamlandı: {result}")
        print()
        return True
    except Exception as e:
        logger.error("Yeniden eğitim hatası: %s", e, exc_info=True)
        print(f"⚠️  Yeniden eğitim başarısız: {e}")
        print("   Model mevcut ağırlıklarla devam edecek.")
        print()
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ADIM 3: Scrape & Predict — Yeni Maçları Çek ve Tahmin Yap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_scrape_and_predict(session: Session):
    """
    Yeni maçları scrape edip tahmin yapar ve raporu sunar.

    Alt adımlar:
      3a. Kullanıcıdan kaç maç çekmek istediğini sor
      3b. Scraper çalıştır
      3c. Analiz yap (model ile tahmin üret)
      3d. Raporu terminale yazdır
    """
    print("=" * 80)
    print("🕷️  ADIM 3: YENİ MAÇLAR — SCRAPE & PREDICT")
    print("=" * 80)
    print()

    # 3a. Kullanıcıdan maç sayısı al
    try:
        raw = input("📋 Kaç adet maç çekmek istiyorsunuz? (varsayılan: 20, 0=atla): ").strip()
        if raw == "0":
            print("⏭️  Scrape adımı atlandı.")
            print()
            return
        match_count = int(raw) if raw else 20
        if match_count < 0:
            match_count = 20
    except ValueError:
        match_count = 20

    # 3b. Scraper çalıştır
    cmd_scrape(match_count)

    # 3c. Analiz yap
    results = cmd_analyze(session)

    # 3d. Rapor
    if results:
        print_report(results, session)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Mevcut Komut Fonksiyonları (Korundu)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_scrape(match_count: int):
    """Scraper çalıştır → verileri DB'ye yaz."""
    print("=" * 80)
    print("🕷️  NESINE.COM SCRAPER")
    print("=" * 80)

    stats = run_scraper_to_db(match_count)

    print()
    print("📊 Yazılan Kayıtlar:")
    for key, count in stats.items():
        print(f"   {key}: {count}")
    print()


def cmd_analyze(session: Session) -> List[PredictionResult]:
    """Tüm bekleyen maçları analiz et."""
    print("=" * 80)
    print("🤖 MAÇLAR ANALİZ EDİLİYOR")
    print("=" * 80)
    print()

    # Tahmin motorunu başlat
    predictor = MatchPredictor(session)
    mode_info = predictor.initialize()
    print(f"📊 Tahmin Motoru: {mode_info}")
    print(f"📊 Eğitim Verisi: {predictor.training_samples} sonuçlanmış maç")
    print("-" * 80)

    # Henüz bitmemiş maçları çek
    pending_matches = (
        session.query(Match)
        .filter(Match.is_finished == False)  # noqa: E712
        .all()
    )

    if not pending_matches:
        print("⚠️  Analiz edilecek bekleyen maç yok!")
        return []

    # ── Başlamış / canlı maçları filtrele ──
    now = datetime.now()
    upcoming_matches: list[Match] = []
    skipped = 0
    for match in pending_matches:
        match_dt = _resolve_match_datetime(match)
        if match_dt <= now:
            skipped += 1
            logger.debug(
                "Atlandı (başlamış): %s vs %s — %s %s",
                match.home_team, match.away_team,
                match.match_date, match.match_time,
            )
        else:
            upcoming_matches.append(match)

    if skipped:
        print(f"⏭️  {skipped} başlamış/canlı maç atlandı")
        logger.info("%d başlamış/canlı maç filtrelendi", skipped)

    if not upcoming_matches:
        print("⚠️  Tüm maçlar başlamış — tahmin edilecek maç kalmadı!")
        return []

    print(f"📋 {len(upcoming_matches)} maç analiz ediliyor "
          f"(toplam {len(pending_matches)}, {skipped} atlandı)...")
    print()

    # Toplu tahmin
    results = predictor.predict_batch(upcoming_matches)

    # Tahminleri veritabanına kaydet
    for result in results:
        _save_prediction(session, result)

    session.commit()
    print(f"✓ {len(results)} tahmin veritabanına kaydedildi")
    print()

    return results


def cmd_retrain(session: Session):
    """ML modelini yeniden eğit (standalone komut)."""
    print("=" * 80)
    print("🔧 MODEL YENİDEN EĞİTİLİYOR")
    print("=" * 80)
    print()

    predictor = MatchPredictor(session)
    result = predictor.retrain()
    print(f"✓ {result}")
    print()


def cmd_validate(session: Session):
    """Geçmiş tahminleri doğrula."""
    print("=" * 80)
    print("✅ TAHMİN DOĞRULAMA")
    print("=" * 80)
    print()

    # Sonuçlanmış ama henüz doğrulanmamış tahminler
    preds = (
        session.query(Prediction)
        .join(Match)
        .filter(
            Match.is_finished == True,      # noqa
            Prediction.actual_result.is_(None)
        )
        .all()
    )

    updated = 0
    for pred in preds:
        match = pred.match
        if match and match.result:
            pred.actual_result = match.result
            pred.is_correct = (pred.prediction == match.result)
            updated += 1

    session.commit()
    print(f"✓ {updated} tahmin doğrulandı")

    # Genel istatistikler
    predictor = MatchPredictor(session)
    stats = predictor.validate_past_predictions()

    if stats['total'] > 0:
        print(f"\n📊 Genel Doğruluk: {stats['accuracy']:.1f}% "
              f"({stats['correct']}/{stats['total']})")

        if 'by_engine' in stats:
            print("\n   Motor Bazlı:")
            for eng, data in stats['by_engine'].items():
                print(f"   • {eng}: {data['accuracy']:.1f}% "
                      f"({data['correct']}/{data['total']})")
    else:
        print("⚠️  Henüz doğrulanmış tahmin yok.")
    print()


def cmd_stats(session: Session):
    """Veritabanı istatistiklerini göster."""
    print("=" * 80)
    print("📊 VERİTABANI İSTATİSTİKLERİ")
    print("=" * 80)
    print()

    stats = db_stats(session)
    for table, count in stats.items():
        print(f"   {table}: {count} kayıt")
    print()


def cmd_review(session: Session):
    """Sadece pending review çalıştır (standalone komut)."""
    updated = step_pending_review(session)
    if updated > 0:
        # Doğruluk istatistiklerini göster
        predictor = MatchPredictor(session)
        stats = predictor.validate_past_predictions()
        if stats['total'] > 0:
            print(f"📊 Güncel Doğruluk: {stats['accuracy']:.1f}% "
                  f"({stats['correct']}/{stats['total']})")
            if 'by_engine' in stats:
                for eng, data in stats['by_engine'].items():
                    print(f"   • {eng}: {data['accuracy']:.1f}% "
                          f"({data['correct']}/{data['total']})")
        print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Rapor Fonksiyonları
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_report(results: List[PredictionResult], session: Session):
    """Tahmin raporunu terminale yazdırır."""
    if not results:
        return

    # Güvene göre sırala
    results.sort(key=lambda r: r.confidence, reverse=True)

    # ─── Value Bet Önerileri ──────────────────────────────────────
    value_bets = [r for r in results
                  if r.is_value_bet and r.confidence >= VALUE_BET_MIN_CONFIDENCE]

    print("=" * 80)
    print("📊 TAHMİN RAPORU")
    print("=" * 80)

    if value_bets:
        print()
        print("🎯 VALUE BET ÖNERİLERİ")
        print("-" * 80)

        for i, r in enumerate(value_bets[:10], 1):
            match = session.get(Match, r.match_id)
            odds = session.query(Odds).filter_by(match_id=r.match_id).first()

            print(f"\n{i}. {r.match_display}")
            if match:
                print(f"   📅 {match.match_date} {match.match_time} | "
                      f"🏆 {match.league.name if match.league else '?'}")
            if odds:
                print(f"   💰 Oranlar: 1={odds.ms_1:.2f} | X={odds.ms_x:.2f} | "
                      f"2={odds.ms_2:.2f}")

            print(f"   📈 Tahmin: MS{r.prediction} | Güven: {r.confidence:.1f}% | "
                  f"Edge: +{r.value_edge:.1f}%")
            print(f"   ⚽ Beklenen Skor: {r.expected_home_goals:.1f}-"
                  f"{r.expected_away_goals:.1f}")
            print(f"   📊 1={r.prob_home:.1f}% | X={r.prob_draw:.1f}% | "
                  f"2={r.prob_away:.1f}% | Ü2.5={r.prob_over_25:.1f}%")
            print(f"   {r.risk_level}")
            print(f"   💡 {r.explanation}")

            if r.top_scores:
                scores_str = ", ".join(f"{s[0]}(%{s[1]:.1f})" for s in r.top_scores[:3])
                print(f"   🎯 Olası Skorlar: {scores_str}")
    else:
        print("\n⚠️  Güçlü value bet bulunamadı.")

    # ─── Tüm Maçlar Özeti ────────────────────────────────────────
    print()
    print("=" * 80)
    print("📋 TÜM MAÇLAR ÖZETİ")
    print("=" * 80)
    print()
    print(f"{'Maç':<40} {'Tah.':<6} {'Güven':<8} {'1%':<7} {'X%':<7} "
          f"{'2%':<7} {'Edge':<8} {'Risk':<15}")
    print("-" * 100)

    for r in results[:MAX_REPORT_MATCHES]:
        vb = " 💰" if r.is_value_bet else ""
        print(f"{r.match_display[:39]:<40} MS{r.prediction:<4} "
              f"{r.confidence:>5.1f}%  "
              f"{r.prob_home:>5.1f}  {r.prob_draw:>5.1f}  {r.prob_away:>5.1f}  "
              f"{r.value_edge:>+6.1f}%  {r.risk_level}{vb}")

    # ─── Alt/Üst Analizi ─────────────────────────────────────────
    print()
    print("=" * 80)
    print("⚽ ALT / ÜST ANALİZİ (Poisson Beklentisi)")
    print("=" * 80)
    print()

    over_candidates = sorted(
        [r for r in results if r.prob_over_25 >= 55],
        key=lambda r: r.prob_over_25, reverse=True
    )

    if over_candidates:
        print("🔼 ÜST 2.5 Gol Önerileri:")
        print("-" * 60)
        for r in over_candidates[:7]:
            odds = session.query(Odds).filter_by(match_id=r.match_id).first()
            oran = (odds.ust_2_5 or 0.0) if odds else 0.0
            print(f"  • {r.match_display}")
            print(f"    Ü2.5: %{r.prob_over_25:.1f} | Oran: {oran:.2f} | "
                  f"Beklenen: {r.expected_home_goals:.1f}-{r.expected_away_goals:.1f}")

    under_candidates = sorted(
        [r for r in results if r.prob_under_25 >= 55],
        key=lambda r: r.prob_under_25, reverse=True
    )

    if under_candidates:
        print()
        print("🔽 ALT 2.5 Gol Önerileri:")
        print("-" * 60)
        for r in under_candidates[:7]:
            odds = session.query(Odds).filter_by(match_id=r.match_id).first()
            oran = (odds.alt_2_5 or 0.0) if odds else 0.0
            print(f"  • {r.match_display}")
            print(f"    A2.5: %{r.prob_under_25:.1f} | Oran: {oran:.2f} | "
                  f"Beklenen: {r.expected_home_goals:.1f}-{r.expected_away_goals:.1f}")

    # ─── İstatistiksel Özet ───────────────────────────────────────
    print()
    print("=" * 80)
    print("📊 İSTATİSTİKSEL ÖZET")
    print("=" * 80)

    if results:
        avg_conf = sum(r.confidence for r in results) / len(results)
        high = len([r for r in results if r.confidence >= 55])
        med = len([r for r in results if 45 <= r.confidence < 55])
        low = len([r for r in results if r.confidence < 45])

        engine_used = results[0].engine_used if results else "?"

        print(f"""
  Tahmin Motoru: {engine_used}
  Toplam Analiz: {len(results)} maç
  Ort. Güven: {avg_conf:.1f}%

  Güven Dağılımı:
    🟢 Yüksek (55%+): {high} maç
    🟡 Orta (45-55%): {med} maç
    🔴 Düşük (<45%):  {low} maç

  Value Bet: {len(value_bets)} maç
        """)

    # ── CSV'ye kaydet ──
    save_results_to_csv(results)


def _save_prediction(session: Session, result: PredictionResult):
    """Tahmini veritabanına kaydeder."""
    pred = Prediction(
        match_id=result.match_id,
        **result.to_prediction_model()
    )
    session.add(pred)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CSV Dışa Aktarım
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_results_to_csv(results: List[PredictionResult], filename: str = "Tahmin_Raporu.csv"):
    """
    Tahmin sonuçlarını detaylı bir şekilde CSV dosyasına kaydeder.
    UTF-8-SIG kodlaması ve ';' ayıracı ile Excel uyumludur.
    """
    if not results:
        return

    fieldnames = [
        "Tarih", "Mac", "Lig", "Tahmin", "Guven_Yuzdesi",
        "Beklenen_Ev_Gol", "Beklenen_Dep_Gol",
        "1_Olasilik", "X_Olasilik", "2_Olasilik",
        "Alt_2.5_Olasilik", "Ust_2.5_Olasilik",
        "Value_Bet", "Value_Edge", "Risk_Seviyesi",
        "Kullanilan_Motor", "Aciklama"
    ]

    file_path = os.path.join(str(BASE_DIR), filename)

    try:
        with open(file_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()

            for r in results:
                row = {
                    "Tarih": datetime.now().strftime("%Y-%m-%d"),
                    "Mac": r.match_display,
                    "Lig": "",
                    "Tahmin": f"MS {r.prediction}",
                    "Guven_Yuzdesi": f"{r.confidence:.2f}",
                    "Beklenen_Ev_Gol": f"{r.expected_home_goals:.2f}",
                    "Beklenen_Dep_Gol": f"{r.expected_away_goals:.2f}",
                    "1_Olasilik": f"{r.prob_home:.2f}",
                    "X_Olasilik": f"{r.prob_draw:.2f}",
                    "2_Olasilik": f"{r.prob_away:.2f}",
                    "Alt_2.5_Olasilik": f"{r.prob_under_25:.2f}",
                    "Ust_2.5_Olasilik": f"{r.prob_over_25:.2f}",
                    "Value_Bet": "EVET" if r.is_value_bet else "HAYIR",
                    "Value_Edge": f"{r.value_edge:.2f}",
                    "Risk_Seviyesi": r.risk_level,
                    "Kullanilan_Motor": r.engine_used,
                    "Aciklama": r.explanation
                }
                writer.writerow(row)

        print(f"\n✅ Tahminler CSV olarak kaydedildi: {file_path}")

    except Exception as e:
        logger.error(f"CSV kaydetme hatası: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ANA FONKSİYON — Active Learning Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """
    Active Learning (Human-in-the-Loop) Ana Döngüsü.

    Akış Diyagramı:
    ┌─────────────────────────────────────────────────┐
    │  1. Pending Review                              │
    │     Doğrulanmamış geçmiş tahminleri bul         │
    │     Kullanıcıdan maç sonuçlarını al             │
    │                  ↓                              │
    │  2. Online Retrain                              │
    │     Yeni verilerle modeli yeniden eğit           │
    │     Güncel ağırlıkları .pkl olarak kaydet        │
    │                  ↓                              │
    │  3. Scrape & Predict                            │
    │     Yeni maçları çek, tahmin yap, rapor sun      │
    └─────────────────────────────────────────────────┘

    Standalone komutlar (--flag ile):
      --scrape N   : Sadece N maç scrape et
      --analyze    : Sadece mevcut verilerle analiz
      --retrain    : Sadece model yeniden eğit
      --validate   : Geçmiş tahminleri doğrula
      --stats      : Veritabanı istatistikleri
      --review     : Sadece pending review (sonuç girişi)
    """

    parser = argparse.ArgumentParser(
        description="🎯 Nesine Futbol Tahmin Sistemi — Active Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py                  # Tam Active Learning pipeline
  python main.py --scrape 30      # 30 maç çek
  python main.py --analyze        # Sadece analiz
  python main.py --retrain        # Model yeniden eğit
  python main.py --validate       # Tahminleri doğrula
  python main.py --stats          # DB istatistikleri
  python main.py --review         # Sadece sonuç girişi
        """
    )
    parser.add_argument('--scrape', type=int, metavar='N',
                        help='Sadece N maç scrape et')
    parser.add_argument('--analyze', action='store_true',
                        help='Sadece analiz yap')
    parser.add_argument('--retrain', action='store_true',
                        help='ML modelini yeniden eğit')
    parser.add_argument('--validate', action='store_true',
                        help='Geçmiş tahminleri doğrula')
    parser.add_argument('--stats', action='store_true',
                        help='Veritabanı istatistiklerini göster')
    parser.add_argument('--review', action='store_true',
                        help='Sadece pending review (maç sonuçlarını gir)')

    args = parser.parse_args()

    setup_logging()
    init_db()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      🎯  NESİNE FUTBOL TAHMİN SİSTEMİ                     ║")
    print("║      Active Learning + Stacking Ensemble Pipeline          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    try:
        # ─── Standalone Komutlar ─────────────────────────────────
        if args.scrape:
            cmd_scrape(args.scrape)
            return

        if args.stats:
            with get_session() as session:
                cmd_stats(session)
            return

        if args.retrain:
            with get_session() as session:
                cmd_retrain(session)
            return

        if args.validate:
            with get_session() as session:
                cmd_validate(session)
            return

        if args.analyze:
            with get_session() as session:
                results = cmd_analyze(session)
                if results:
                    print_report(results, session)
            return

        if args.review:
            with get_session() as session:
                cmd_review(session)
            return

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  TAM ACTIVE LEARNING PIPELINE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        with get_session() as session:

            # ── ADIM 1: Pending Review ──────────────────────────
            # Doğrulanmamış geçmiş tahminleri kontrol et
            # Kullanıcıdan maç sonuçlarını al
            updated_count = step_pending_review(session)

            # ── ADIM 2: Online Retrain ──────────────────────────
            # Yeni sonuç girildiyse modeli yeniden eğit
            if updated_count > 0:
                print(f"🔄 {updated_count} yeni sonuç girildi — model güncelleniyor...")
                print()
                step_retrain(session)

                # Doğruluk istatistiklerini göster
                predictor = MatchPredictor(session)
                stats = predictor.validate_past_predictions()
                if stats['total'] > 0:
                    print(f"📊 Model Performansı: {stats['accuracy']:.1f}% doğruluk "
                          f"({stats['correct']}/{stats['total']} tahmin)")
                    if 'by_engine' in stats:
                        for eng, data in stats['by_engine'].items():
                            print(f"   • {eng}: {data['accuracy']:.1f}% "
                                  f"({data['correct']}/{data['total']})")
                    print()
            else:
                print("ℹ️  Yeni sonuç girilmedi — model mevcut ağırlıklarla devam ediyor.")
                print()

            # ── ADIM 3: Scrape & Predict ────────────────────────
            # Yeni maçları çek ve güncellenmiş model ile tahmin yap
            step_scrape_and_predict(session)

        print()
        print("=" * 80)
        print("🏁 Active Learning döngüsü tamamlandı!")
        print("   Sonraki çalıştırmada yeni tahminler doğrulanmayı bekleyecek.")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⚠️  İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        logger.error(f"Kritik hata: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
