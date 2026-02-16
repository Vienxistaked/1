# 🔍 Audit Raporu — Nesine Futbol Maçı Verileri

**Tarih:** 2025-07-17  
**Branch:** `audit-fixes`  
**Denetçi:** GitHub Copilot (Senior Python + ML + Security + DevOps)  
**Proje versiyonu:** v3.1 (Stacking Ensemble + Poisson Cold-Start)

---

## Özet

| Seviye | Bulgu Sayısı | Düzeltilen | Notlandırılan |
|--------|-------------|------------|---------------|
| **P0 — Kritik** | 7 | 7 | 0 |
| **P1 — Önemli** | 6 | 5 | 1 |
| **P2 — İyileştirme** | 5 | 4 | 1 |
| **Toplam** | **18** | **16** | **2** |

---

## Detaylı Bulgular

| # | Seviye | Bulgu | Etki | Kanıt | Çözüm | Durum |
|---|--------|-------|------|-------|-------|-------|
| F01 | **P0** | `.gitignore` eksik — `__pycache__/`, `*.db`, `logs/`, `*.pkl`, CSV'ler tracked | Credentials, binary model, cache dosyaları repo'ya sızıyor | Repo kökünde `.gitignore` yoktu | `.gitignore` oluşturuldu (57 satır) | ✅ Fixed |
| F02 | **P0** | `docker-compose.yml` — hardcoded `POSTGRES_PASSWORD: nesine_pass` | Üretim şifresi açık metin olarak repo'da | `docker-compose.yml:16` | `env_file: .env` + `.env.example` | ✅ Fixed |
| F03 | **P0** | Timezone — `datetime.now()` / `datetime.utcnow()` kullanımı | Tüm zamanlamaları UTC olarak kaydeder, Türkiye'de +3 saat kayma; kronolojik sıralama bozulur | `config.py`, `feature_engineering.py`, `main.py`, `models.py` | `now_istanbul()` yardımcı fonksiyonu + `TZ_ISTANBUL` sabiti | ✅ Fixed |
| F04 | **P0** | `_parse_turkish_date()` — naive vs aware datetime karışımı | `h_dt < ref_dt` karşılaştırması `TypeError` fırlatır; H2H feature hesaplaması çöker | `feature_engineering.py:566` | Tüm format dalları `tzinfo=TZ_ISTANBUL` ile tutarlı hale getirildi | ✅ Fixed |
| F05 | **P0** | Input validation — boş Enter review'u atlıyor | Kullanıcı yanlışlıkla Enter'a basınca maç doğrulanmadan geçiyor, contaminated training set | `main.py:step_pending_review()` | Boş string reddedilir, tekrar sorulur | ✅ Fixed |
| F06 | **P0** | `pickle.load()` — RCE (Remote Code Execution) riski | Kötü niyetli `.pkl` dosyası sunucuda rastgele kod çalıştırabilir | `predictor.py:_load_model()` | SHA-256 hash doğrulaması + path traversal koruması + hash eksikse yükleme reddi | ✅ Fixed |
| F07 | **P0** | DB session — tek session tüm pipeline boyunca | Uzun süren session lock, bağlantı zaman aşımı, yarı işlenmiş veri | `main.py:run_active_learning_pipeline()` | Her adım kendi `get_session()` context manager'ını kullanır | ✅ Fixed |
| F08 | **P1** | Feature sayısı tutarsızlığı (docstring 85, gerçek 96) | Yanlış expected boyut → model cache invalidation tetiklenmez | `predictor.py`, `feature_engineering.py` docstrings | Docstring'ler 96 yapıldı + `N_FEATURES = 96` config sabiti eklendi | ✅ Fixed |
| F09 | **P1** | Hardcoded `random_state=42` (12 yerde) | `RANDOM_SEED` env var ile geçersiz kılınamaz, tekrarlanabilirliğe zarar | `predictor.py` (12 satır) | Tümü `RANDOM_SEED` sabitine refactor edildi | ✅ Fixed |
| F10 | **P1** | Global random seed enforcement eksik | `numpy`, stdlib `random`, `PYTHONHASHSEED` seed'lenmemiş → non-deterministic runs | `config.py` | `random.seed()`, `np.random.seed()`, `PYTHONHASHSEED` config.py import sırasında ayarlanıyor | ✅ Fixed |
| F11 | **P1** | `requirements.txt` — versiyonlar pinlenmemiş | `pip install -r requirements.txt` farklı makinelerde farklı versiyonlar kurar | `requirements.txt` | Tüm 14 bağımlılık mevcut versiyona pinlendi (`==`) | ✅ Fixed |
| F12 | **P1** | f-string loglama (`logger.info(f"…")`) | String her çağrıda formatlanır (log seviyesi düşükse bile); Sentry/structured logging'e zarar | `scraper_db.py` (10 satır) | `logger.info("… %s", var)` lazy formatting'e geçirildi | ✅ Fixed |
| F13 | **P1** | `test_v2.py` — tamamen stale (`N_FEATURES=61`, `MODEL_VERSION=v2.1`) | %100 assertion failure, CI/CD'de hiçbir koruma yok | `test_v2.py` tüm dosya | Yeni `tests/test_smoke_v31.py` (28 test, v3.1 uyumlu) | ✅ Fixed |
| F14 | **P2** | `value_bet_analyzer.py` — 1057 satır dead code | CSV-based eski analiz sistemi, DB pipeline ile entegre değil; bakım yükü | `value_bet_analyzer.py` tüm dosya | 📝 Noted — gelecekte kaldırılmalı veya DB'ye entegre edilmeli | 📝 Noted |
| F15 | **P2** | `predictor_v2_backup.py` — yedek dosya repo'da | v2.1 kodu artık kullanılmıyor, karışıklığa sebep oluyor | `predictor_v2_backup.py` tüm dosya | 📝 Noted — silinmeli | 📝 Noted |
| F16 | **P2** | `catboost_info/` training artifacts tracked | Binary eğitim loglama dosyaları repo boyutunu şişirir | `catboost_info/` dizini | `.gitignore`'a eklendi, git-cached'den silindi | ✅ Fixed |
| F17 | **P2** | `.pytest_cache/` tracked | Test cache dosyaları repo'da kalıyor | `.pytest_cache/` dizini | `.gitignore`'a eklendi | ✅ Fixed |
| F18 | **P2** | Model hash dosyası (`.sha256`) yoktu | Pickle model dosyası integrity doğrulaması yapılamıyordu | `predictor.py` | `_save_model()` SHA-256 hash sidecar yazıyor, `_load_model()` doğruluyor | ✅ Fixed |

---

## Commit Geçmişi (`audit-fixes` branch)

| # | Commit | Mesaj |
|---|--------|-------|
| 1 | `f4fbe2c` | `fix: timezone — tüm datetime.now() → now_istanbul (Europe/Istanbul aware)` |
| 2 | `8120f74` | `fix: input validation — boş Enter review atlamasın` |
| 3 | `1374b4a` | `refactor: DB session stratejisi — adım başına ayrı session` |
| 4 | `5534dd5` | `refactor: fix feature count references (85 → 96)` |
| 5 | `571d6ae` | `security: pickle model dosyası için SHA-256 hash doğrulaması` |
| 6 | `339a5ad` | `chore: determinizm + requirements pinleme + f-string loglama düzeltmesi` |
| 7 | `50c1d0d` | `test: stale test_v2.py yerine v3.1 uyumlu test suite (28 test)` |
| 8 | `d8959bc` | `chore: catboost_info/ ve .pytest_cache/ gitignore'a eklendi` |

---

## Test Sonuçları

```
======================== 28 passed, 1 warning in 3.82s =========================
```

| Test Sınıfı | Test Sayısı | Durum |
|-------------|-------------|-------|
| `TestConfig` | 4 | ✅ |
| `TestTurkishDateParse` | 7 | ✅ |
| `TestFormCalculations` | 5 | ✅ |
| `TestFeatureExtractor` | 4 | ✅ |
| `TestBuildTrainingDataset` | 2 | ✅ |
| `TestPredictor` | 3 | ✅ |
| `TestPoissonModel` | 1 | ✅ |
| **Toplam** | **28** | ✅ |

> **Warning (beklenen):** LGBMClassifier feature names uyarısı — model `.fit()` sırasında feature isimlerinden farklı ndarray formatında veri alıyor. İşlevselliğe etkisi yok.

---

## Kırılma Değişiklikleri (Breaking Changes)

1. **Model cache invalidation:** Mevcut `.pkl` dosyaları hash olmadan yüklenemez. İlk çalıştırmada yeniden eğitim gerekir.
2. **`_parse_turkish_date`** artık tüm formatlarda timezone-aware datetime döndürür. Naive datetime bekleyen harici kod varsa kırılır.
3. **`requirements.txt`** sabit versiyona pinlendi — `pip install -r requirements.txt` sadece pinlenen versiyonları kurar.

---

## Kalan İyileştirmeler (Gelecek Sprint)

- [ ] `value_bet_analyzer.py` DB pipeline'a entegre et veya kaldır
- [ ] `predictor_v2_backup.py`, `feature_engineering_v2_backup.py` sil
- [ ] CI/CD pipeline (GitHub Actions) ekle — testleri otomatik çalıştır
- [ ] Model performans metrikleri dashboard'u (MLflow / Weights & Biases)
- [ ] `test_v2.py` silinmeli (artık stale, `tests/test_smoke_v31.py` kullanılıyor)
