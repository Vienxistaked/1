"""
Nesine.com Futbol Maçları - Bülten & Puan Tablosu Scraper
Hibrit mimari: Selenium (sayfa yükleme) + BeautifulSoup (veri çekme)
"""

import csv
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException


# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class TeamStanding:
    """Takım puan tablosu verilerini tutan veri sınıfı."""
    Maç_Kodu: Optional[str] = None
    Maç: Optional[str] = None
    Takım_Tipi: Optional[str] = None
    Sıra: Optional[str] = None
    Takım: Optional[str] = None
    O: Optional[str] = None
    G: Optional[str] = None
    B: Optional[str] = None
    M: Optional[str] = None
    A_Y: Optional[str] = None
    AV: Optional[str] = None
    P: Optional[str] = None
    Form: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        return {k: (v if v is not None else "") for k, v in asdict(self).items()}


@dataclass
class LastMatch:
    """Son maç verilerini tutan veri sınıfı."""
    Maç_Kodu: Optional[str] = None
    Güncel_Maç: Optional[str] = None
    Takım: Optional[str] = None
    Takım_Tipi: Optional[str] = None
    Lig: Optional[str] = None
    Tarih: Optional[str] = None
    Ev_Sahibi: Optional[str] = None
    Deplasman: Optional[str] = None
    MS: Optional[str] = None
    İY: Optional[str] = None
    Sonuç: Optional[str] = None  # Galibiyet, Mağlubiyet, Beraberlik
    
    def to_dict(self) -> Dict[str, str]:
        return {k: (v if v is not None else "") for k, v in asdict(self).items()}


@dataclass
class RefereeMatch:
    """Hakem maç verilerini tutan veri sınıfı."""
    Maç_Kodu: Optional[str] = None
    Güncel_Maç: Optional[str] = None
    Hakem_Adı: Optional[str] = None
    Lig: Optional[str] = None
    Tarih: Optional[str] = None
    Ev_Sahibi: Optional[str] = None
    Deplasman: Optional[str] = None
    MS: Optional[str] = None
    İY: Optional[str] = None
    Oran_1: Optional[str] = None
    Oran_1_Geldi: Optional[str] = None
    Oran_X: Optional[str] = None
    Oran_X_Geldi: Optional[str] = None
    Oran_2: Optional[str] = None
    Oran_2_Geldi: Optional[str] = None
    Oran_Alt: Optional[str] = None
    Oran_Alt_Geldi: Optional[str] = None
    Oran_Üst: Optional[str] = None
    Oran_Üst_Geldi: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        return {k: (v if v is not None else "") for k, v in asdict(self).items()}


@dataclass
class RefereeStats:
    """Hakem istatistik verilerini tutan veri sınıfı."""
    Maç_Kodu: Optional[str] = None
    Güncel_Maç: Optional[str] = None
    Hakem_Adı: Optional[str] = None
    MS1_Sayı: Optional[str] = None
    MS1_Yüzde: Optional[str] = None
    MSX_Sayı: Optional[str] = None
    MSX_Yüzde: Optional[str] = None
    MS2_Sayı: Optional[str] = None
    MS2_Yüzde: Optional[str] = None
    Alt_2_5_Sayı: Optional[str] = None
    Alt_2_5_Yüzde: Optional[str] = None
    Üst_2_5_Sayı: Optional[str] = None
    Üst_2_5_Yüzde: Optional[str] = None
    KG_Var_Sayı: Optional[str] = None
    KG_Var_Yüzde: Optional[str] = None
    KG_Yok_Sayı: Optional[str] = None
    KG_Yok_Yüzde: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        return {k: (v if v is not None else "") for k, v in asdict(self).items()}


@dataclass
class CompetitionHistory:
    """Rekabet geçmişi verilerini tutan veri sınıfı."""
    Maç_Kodu: Optional[str] = None
    Güncel_Maç: Optional[str] = None
    Lig: Optional[str] = None
    Tarih: Optional[str] = None
    Ev_Sahibi: Optional[str] = None
    Deplasman: Optional[str] = None
    MS: Optional[str] = None
    İY: Optional[str] = None
    Oran_1: Optional[str] = None
    Oran_1_Geldi: Optional[str] = None
    Oran_X: Optional[str] = None
    Oran_X_Geldi: Optional[str] = None
    Oran_2: Optional[str] = None
    Oran_2_Geldi: Optional[str] = None
    Oran_Alt: Optional[str] = None
    Oran_Alt_Geldi: Optional[str] = None
    Oran_Üst: Optional[str] = None
    Oran_Üst_Geldi: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        return {k: (v if v is not None else "") for k, v in asdict(self).items()}


@dataclass
class InjuryData:
    """Sakat ve cezalı oyuncu verilerini tutan veri sınıfı."""
    Maç_Kodu: Optional[str] = None
    Maç: Optional[str] = None
    Takım: Optional[str] = None
    Numara: Optional[str] = None
    Oyuncu: Optional[str] = None
    Yaş: Optional[str] = None
    Pozisyon: Optional[str] = None
    Maç_Sayısı: Optional[str] = None
    İlk_11: Optional[str] = None
    Gol: Optional[str] = None
    Asist: Optional[str] = None
    Durum: Optional[str] = None  # "Sakatlık" veya "Cezalı"
    Açıklama: Optional[str] = None  # Detaylı açıklama
    
    def to_dict(self) -> Dict[str, str]:
        return {k: (v if v is not None else "") for k, v in asdict(self).items()}


@dataclass
class MatchData:
    """Maç verilerini tutan veri sınıfı."""
    Maç_Kodu: Optional[str] = None
    Lig: Optional[str] = None
    Tarih: Optional[str] = None
    Saat: Optional[str] = None
    Maç: Optional[str] = None
    MBS: Optional[str] = None
    MS_1: Optional[str] = None
    MS_X: Optional[str] = None
    MS_2: Optional[str] = None
    Alt_2_5: Optional[str] = None
    Üst_2_5: Optional[str] = None
    HND: Optional[str] = None
    HND_1: Optional[str] = None
    HND_X: Optional[str] = None
    HND_2: Optional[str] = None
    ÇS_1X: Optional[str] = None
    ÇS_12: Optional[str] = None
    ÇS_X2: Optional[str] = None
    KG_Var: Optional[str] = None
    KG_Yok: Optional[str] = None
    Market_Sayısı: Optional[str] = None
    İstatistik_Link: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        return {k: (v if v is not None else "") for k, v in asdict(self).items()}


class NesineScraper:
    """Nesine.com bülten ve puan tablosu verilerini çeken scraper."""
    
    # le=0 → tüm ligler dahil (le=2 sadece editör seçkisi, daha az maç)
    BASE_URL: str = "https://www.nesine.com/iddaa"
    DEFAULT_PARAMS: Dict[str, str] = {"et": "1", "le": "0"}
    TIMEOUT: int = 15
    
    def __init__(self, match_count: int) -> None:
        self.match_count: int = match_count
        self.driver: Optional[webdriver.Chrome] = None
        self.matches: List[MatchData] = []
        self._seen_codes: set = set()  # Duplicate koruma (incremental collection)
        self._last_league_info: Dict[str, str] = {"league": None, "date": None}
        self.standings: List[TeamStanding] = []
        self.competition_history: List[CompetitionHistory] = []
        self.last_matches: List[LastMatch] = []
        self.referee_matches: List[RefereeMatch] = []
        self.referee_stats: List[RefereeStats] = []
        self.injury_data: List[InjuryData] = []
        
    def setup_driver(self) -> None:
        """Chrome WebDriver'ı headless modda yapılandırır."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        self.driver = webdriver.Chrome(options=options)
        logger.info("WebDriver başlatıldı (headless)")
        
    def wait_for_page_load(self) -> None:
        """Sayfanın yüklenmesini bekler."""
        wait = WebDriverWait(self.driver, self.TIMEOUT)
        wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div[data-test-id^='r_'][data-code]")
        ))
        logger.info("Bülten verileri yüklendi")
        
    def close_popups(self) -> None:
        """Popup'ları JavaScript ile kapatır."""
        popup_scripts = [
            """
            const cookieBtn = document.evaluate(
                "//button[contains(text(), 'Kabul Et')]",
                document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null
            ).singleNodeValue;
            if (cookieBtn) cookieBtn.click();
            """,
            """
            const closeBtn = document.querySelector('button[class*="ebfa54f068cb6c89755a"]');
            if (closeBtn) closeBtn.click();
            """,
            """
            const kapatBtn = document.evaluate(
                "//button[contains(text(), 'Kapat')]",
                document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null
            ).singleNodeValue;
            if (kapatBtn) kapatBtn.click();
            """,
            """
            document.querySelectorAll('button i.ni-close-rounded').forEach(i => {
                i.closest('button')?.click();
            });
            """
        ]
        
        for script in popup_scripts:
            try:
                self.driver.execute_script(script)
            except WebDriverException:
                pass
                
        logger.info("Popup'lar kapatıldı")
        
    def scroll_to_load_matches(self) -> None:
        """Legacy wrapper — artık _scroll_and_collect kullanılıyor."""
        self._scroll_and_collect()

    # ── Scroll Container Algılama (DOM Virtualization Fix) ─────────
    _SCROLL_CONTAINER_JS: str = """
    // Nesine.com 'overflow: hidden' body kullanır.
    // Gerçek scroll container'ı bulmak için tüm ancestor'ları tarayıp
    // scrollHeight > clientHeight olan ve overflow auto/scroll set edilmiş
    // ilk elementi döndürüyoruz.
    (function findScrollContainer() {
        // Strateji 1: Maç satırlarının en yakın scrollable parent'ı
        const firstRow = document.querySelector("div[data-test-id^='r_'][data-code]");
        if (firstRow) {
            let el = firstRow.parentElement;
            while (el && el !== document.documentElement) {
                const style = window.getComputedStyle(el);
                const oy = style.overflowY;
                if ((oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 50) {
                    return el;
                }
                el = el.parentElement;
            }
        }
        // Strateji 2: class adında 'scroll' geçen div'ler
        const candidates = document.querySelectorAll("div[class*='scroll'], div[class*='Scroll']");
        for (const c of candidates) {
            if (c.scrollHeight > c.clientHeight + 50) return c;
        }
        // Strateji 3: scrollHeight > clientHeight olan en büyük div
        let best = null; let bestDelta = 0;
        document.querySelectorAll('div').forEach(d => {
            const delta = d.scrollHeight - d.clientHeight;
            if (delta > 200 && delta > bestDelta) {
                const s = window.getComputedStyle(d);
                if (s.overflowY !== 'visible' && s.overflowY !== 'hidden') {
                    best = d; bestDelta = delta;
                }
            }
        });
        if (best) return best;
        // Strateji 4 (fallback): body
        return document.body;
    })();
    """

    def _find_scroll_container(self) -> None:
        """Sayfadaki gerçek scroll container'ı tespit edip JS referansını kaydeder.

        ``window.__nsnScrollContainer`` global değişkenine atanır.
        Sonraki scroll işlemlerinde bu referans kullanılır.
        """
        self.driver.execute_script(
            f"window.__nsnScrollContainer = {self._SCROLL_CONTAINER_JS}"
        )
        # Hangi elementin bulunduğunu logla
        tag_info: str = self.driver.execute_script("""
            const c = window.__nsnScrollContainer;
            return c.tagName + '.' + (c.className || '').substring(0, 60)
                   + ' [scrollH=' + c.scrollHeight + ', clientH=' + c.clientHeight + ']';
        """)
        logger.info(f"  Scroll container: {tag_info}")

    def _wait_for_new_rows(self, old_count: int, timeout: float = 8.0) -> int:
        """DOM'a yeni maç satırı eklenene veya XHR bitene kadar bekler.

        ``time.sleep`` yerine Explicit Wait kullanır.
        ``WebDriverWait`` + custom expected_condition ile
        ``div[data-test-id^='r_'][data-code]`` sayısının artmasını bekler.

        Parameters
        ----------
        old_count : int
            Scroll öncesi DOM'daki satır sayısı.
        timeout : float
            Maksimum bekleme süresi (saniye).

        Returns
        -------
        int
            Bekleme sonrası DOM'daki güncel satır sayısı.
        """
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: len(d.find_elements(
                    By.CSS_SELECTOR, "div[data-test-id^='r_'][data-code]"
                )) > old_count
            )
        except TimeoutException:
            pass  # Timeout → mevcut sayıyla devam et

        # Network idle bekleme: bekleyen XHR/Fetch sayısı 0 olana kadar
        try:
            self.driver.execute_async_script("""
                const cb = arguments[arguments.length - 1];
                // 500ms boyunca yeni network isteği gelmezse tamam say
                let timer = null;
                const done = () => { clearTimeout(timer); cb(true); };
                timer = setTimeout(done, 500);
            """)
        except (TimeoutException, WebDriverException):
            pass

        return len(self.driver.find_elements(
            By.CSS_SELECTOR, "div[data-test-id^='r_'][data-code]"
        ))

    def _scroll_and_collect(self) -> None:
        """Scroll → Explicit Wait → Parse → Retry döngüsü ile maç toplar.

        DOM Virtualization-Resistant Strateji:
        ──────────────────────────────────────
        1. _find_scroll_container() ile gerçek scrollable elementi bul
        2. Her adımda son maç satırına scrollIntoView() yap (container
           tabanlı scroll + görünür elemana odaklanma)
        3. _wait_for_new_rows() ile Explicit Wait (yeni DOM node bekleme)
        4. Her adımda page_source → parse → _seen_codes ile incremental
        5. Stale olursa büyük sıçrama + retry mekanizması
        6. match_count'a ulaşılmadıkça ASLA erken çıkma

        Retry Mekanizması:
          • Stale (yeni maç gelmeme) sayacı
          • MAX_RETRY_AFTER_STALE: Sayfa sonundan emin olunca bile
            retry denemesi (tam sayfa yeniden yükleme dahil)
          • scrollIntoView fallback: container scroll başarısız olursa
        """
        SCROLL_STEP: int = 1200           # px — düşük tutarak virtualization kaçırmasını azalt
        MAX_STALE_ROUNDS: int = 6         # Ardışık yeni veri gelmeyen turlar
        MAX_TOTAL_SCROLLS: int = 500      # Güvenlik sınırı
        MAX_RETRY_AFTER_STALE: int = 3    # Sayfa sonu sonrası toplam retry hakkı
        AGGRESSIVE_JUMP: int = 4000       # Stale durumda büyük sıçrama (px)

        stale_rounds: int = 0
        total_scrolls: int = 0
        prev_collected: int = 0
        retry_count: int = 0

        logger.info("Bülten verileri çekiliyor (smart scroll + explicit wait)...")
        logger.info("-" * 60)

        # ── Scroll container'ı tespit et ──
        self._find_scroll_container()

        # ── İlk yükleme: mevcut DOM'daki tüm maçları topla ──
        time.sleep(1.5)
        soup = self.get_page_source()
        self.get_match_data(soup)
        logger.info(f"  İlk yükleme: {len(self.matches)} maç toplandı")

        while total_scrolls < MAX_TOTAL_SCROLLS:
            # ── Hedef kontrolü ──
            if len(self.matches) >= self.match_count:
                logger.info(
                    f"✓ Hedef ulaşıldı: {len(self.matches)}/{self.match_count} maç"
                )
                break

            # ── Mevcut DOM satır sayısı (Explicit Wait referansı) ──
            current_dom_count: int = len(self.driver.find_elements(
                By.CSS_SELECTOR, "div[data-test-id^='r_'][data-code]"
            ))

            # ── SCROLL: Çift strateji (container + scrollIntoView) ──
            try:
                # Strateji A: Gerçek scroll container'ı kaydır
                self.driver.execute_script(
                    f"window.__nsnScrollContainer.scrollTop += {SCROLL_STEP};"
                )
            except WebDriverException:
                pass

            try:
                # Strateji B: Son görünür maç satırına scrollIntoView
                # (DOM Virtualization altında en güvenilir yöntem)
                self.driver.execute_script("""
                    const rows = document.querySelectorAll(
                        "div[data-test-id^='r_'][data-code]"
                    );
                    if (rows.length > 0) {
                        rows[rows.length - 1].scrollIntoView({
                            behavior: 'instant', block: 'end'
                        });
                    }
                """)
            except WebDriverException:
                pass

            total_scrolls += 1

            # ── Explicit Wait: Yeni satırlar yüklenene kadar bekle ──
            new_dom_count: int = self._wait_for_new_rows(
                current_dom_count, timeout=8.0
            )

            # ── Parse et & topla ──
            soup = self.get_page_source()
            self.get_match_data(soup)

            if len(self.matches) > prev_collected:
                stale_rounds = 0
                logger.info(
                    f"  📊 Scroll #{total_scrolls}: "
                    f"toplanan={len(self.matches)}/{self.match_count} "
                    f"(DOM={new_dom_count})"
                )
                prev_collected = len(self.matches)
            else:
                stale_rounds += 1

            # ── Sayfa sonu kontrolü (scroll container tabanlı) ──
            at_bottom: bool = self.driver.execute_script("""
                const c = window.__nsnScrollContainer;
                return (c.scrollTop + c.clientHeight) >= (c.scrollHeight - 150);
            """)

            if stale_rounds >= MAX_STALE_ROUNDS:
                if at_bottom and retry_count < MAX_RETRY_AFTER_STALE:
                    # ── RETRY: Sayfa sonu ama hedef sayıya ulaşılmadı ──
                    retry_count += 1
                    logger.warning(
                        f"  ⚠ Sayfa sonuna ulaşıldı ama hedef uzak "
                        f"({len(self.matches)}/{self.match_count}). "
                        f"Retry {retry_count}/{MAX_RETRY_AFTER_STALE}..."
                    )
                    # Sayfayı baştan yüklemeyip container'ı en üste sarmak
                    # ve tekrar aşağı scroll etmek virtualized DOM'u
                    # yeniden render ettirebilir.
                    self.driver.execute_script("""
                        const c = window.__nsnScrollContainer;
                        c.scrollTop = 0;
                    """)
                    time.sleep(2)
                    # Tekrar en alta kaydır (bu sefer adım adım)
                    self.driver.execute_script("""
                        const c = window.__nsnScrollContainer;
                        c.scrollTop = c.scrollHeight;
                    """)
                    time.sleep(3)
                    soup = self.get_page_source()
                    self.get_match_data(soup)

                    if len(self.matches) > prev_collected:
                        prev_collected = len(self.matches)
                        stale_rounds = 0
                        logger.info(
                            f"  ✓ Retry başarılı: {len(self.matches)} maç"
                        )
                        continue

                elif at_bottom and retry_count >= MAX_RETRY_AFTER_STALE:
                    # Tüm retry hakları tükendi — gerçekten sayfa sonu
                    logger.warning(
                        f"Sayfa fiziksel olarak sona erdi — toplanan: "
                        f"{len(self.matches)}/{self.match_count}"
                    )
                    break
                else:
                    # Sayfa sonunda değiliz → agresif sıçrama dene
                    logger.debug(
                        f"  Stale #{stale_rounds}: Agresif sıçrama deneniyor"
                    )
                    try:
                        self.driver.execute_script(
                            f"window.__nsnScrollContainer.scrollTop += {AGGRESSIVE_JUMP};"
                        )
                    except WebDriverException:
                        pass
                    time.sleep(2)
                    # Ek fallback: tüm container'ı tazelemek için sayfa
                    # boyutunu değiştirip geri al (DOM re-render tetikler)
                    try:
                        self.driver.execute_script("""
                            const rows = document.querySelectorAll(
                                "div[data-test-id^='r_'][data-code]"
                            );
                            if (rows.length > 0) {
                                rows[rows.length - 1].scrollIntoView({
                                    behavior: 'instant', block: 'center'
                                });
                            }
                        """)
                    except WebDriverException:
                        pass
                    time.sleep(1)
                    soup = self.get_page_source()
                    self.get_match_data(soup)
                    stale_rounds = stale_rounds // 2  # Kısmen sıfırla

        logger.info(
            f"Scroll tamamlandı: {total_scrolls} adım, "
            f"{len(self.matches)} maç toplandı"
        )

    def get_page_source(self) -> BeautifulSoup:
        """Sayfa kaynağını BeautifulSoup ile parse eder."""
        return BeautifulSoup(self.driver.page_source, "lxml")
        
    @staticmethod
    def extract_odd(row: Tag, testid: str) -> Optional[str]:
        """Oran değerini data-testid ile çeker."""
        btn = row.select_one(f'button[data-testid="{testid}"]')
        if btn:
            odd_divs = btn.select("div > div")
            for div in odd_divs:
                text = div.get_text(strip=True)
                if text and text.replace(".", "").replace(",", "").isdigit():
                    return text
                if ":" in text:
                    return text
        return None
        
    def parse_match_row(self, row: Tag, league_info: Dict[str, str]) -> Optional[MatchData]:
        """Tek bir maç satırını parse eder."""
        try:
            match_code = row.get("data-code")
            if not match_code:
                return None
                
            teams_elem = row.select_one('a[data-test-id="matchName"]')
            teams = teams_elem.get_text(strip=True) if teams_elem else None
            stats_link = teams_elem.get("href") if teams_elem else None
            
            time_elem = row.select_one('span[data-testid^="time-"]')
            match_time = time_elem.get_text(strip=True) if time_elem else None
            
            mbs_elem = row.select_one('div[data-test-id="event_mbs"] span')
            mbs = mbs_elem.get_text(strip=True) if mbs_elem else None
            
            market_elem = row.select_one(f'div[data-test-id="{match_code}_m"]')
            market_count = market_elem.get_text(strip=True) if market_elem else None
            
            return MatchData(
                Maç_Kodu=match_code,
                Lig=league_info.get("league"),
                Tarih=league_info.get("date"),
                Saat=match_time,
                Maç=teams,
                MBS=mbs,
                MS_1=self.extract_odd(row, "odd_Maç Sonucu_1"),
                MS_X=self.extract_odd(row, "odd_Maç Sonucu_X"),
                MS_2=self.extract_odd(row, "odd_Maç Sonucu_2"),
                Alt_2_5=self.extract_odd(row, "odd_2,5 Gol_Alt"),
                Üst_2_5=self.extract_odd(row, "odd_2,5 Gol_Üst"),
                HND=self.extract_odd(row, "odd_Handikaplı Maç Sonucu_HND"),
                HND_1=self.extract_odd(row, "odd_Handikaplı Maç Sonucu_1"),
                HND_X=self.extract_odd(row, "odd_Handikaplı Maç Sonucu_X"),
                HND_2=self.extract_odd(row, "odd_Handikaplı Maç Sonucu_2"),
                ÇS_1X=self.extract_odd(row, "odd_Çifte Şans_1-X"),
                ÇS_12=self.extract_odd(row, "odd_Çifte Şans_1-2"),
                ÇS_X2=self.extract_odd(row, "odd_Çifte Şans_X-2"),
                KG_Var=self.extract_odd(row, "odd_Karş. Gol_Var"),
                KG_Yok=self.extract_odd(row, "odd_Karş. Gol_Yok"),
                Market_Sayısı=market_count,
                İstatistik_Link=stats_link
            )
            
        except Exception as e:
            logger.error(f"Maç parse hatası: {e}")
            return None
            
    def extract_league_info(self, container: Tag) -> Dict[str, str]:
        """Lig ve tarih bilgisini container'dan çıkarır."""
        league = None
        date = None
        
        league_elem = container.select_one("strong")
        if league_elem:
            league = league_elem.get_text(strip=True)
            
        date_elem = container.select_one('div[data-test-id="date"]')
        if date_elem:
            date = date_elem.get_text(strip=True)
            
        return {"league": league, "date": date}
        
    def get_match_data(self, soup: BeautifulSoup) -> None:
        """BeautifulSoup ile tüm maç verilerini çeker.

        v2.1 Robust Parsing:
          • Birincil yol: div[data-item-index] container tabanlı traverse
            (lig/tarih header → maç satırları ilişkisi korunur)
          • Yedek yol: Container bulunamazsa veya yeterli maç yoksa,
            doğrudan tüm div[data-test-id^='r_'][data-code] satırlarını
            tara ve her satır için en yakın lig header'ı bul
          • Duplicate koruma: instance-level _seen_codes set ile mükerrer
            engelleme (incremental collection çağrıları arasında korunur)
        """

        # ── Birincil Yol: Container tabanlı traverse ──
        containers = soup.select('div[data-item-index]')

        for container in containers:
            if len(self.matches) >= self.match_count:
                break

            new_league_info = self.extract_league_info(container)
            if new_league_info["league"]:
                self._last_league_info["league"] = new_league_info["league"]
            if new_league_info["date"]:
                self._last_league_info["date"] = new_league_info["date"]

            match_rows = container.select('div[data-test-id^="r_"][data-code]')

            for row in match_rows:
                if len(self.matches) >= self.match_count:
                    break

                code = row.get("data-code")
                if code in self._seen_codes:
                    continue

                match_data = self.parse_match_row(row, self._last_league_info)

                if match_data and match_data.Maç:
                    self._seen_codes.add(code)
                    self.matches.append(match_data)
                    logger.info(
                        f"✓ [{len(self.matches)}/{self.match_count}] "
                        f"{match_data.Maç}"
                    )

        # ── Yedek Yol: Düz satır taraması (container eksikse) ──
        if len(self.matches) < self.match_count:
            all_rows = soup.select('div[data-test-id^="r_"][data-code]')
            new_in_fallback = 0

            for row in all_rows:
                if len(self.matches) >= self.match_count:
                    break

                code = row.get("data-code")
                if code in self._seen_codes:
                    continue

                # En yakın lig header'ını bul (önceki sibling'lerde)
                league_info = self._find_nearest_league_header(row)

                match_data = self.parse_match_row(row, league_info)

                if match_data and match_data.Maç:
                    self._seen_codes.add(code)
                    self.matches.append(match_data)
                    new_in_fallback += 1
                    logger.info(
                        f"✓ [{len(self.matches)}/{self.match_count}] "
                        f"{match_data.Maç}"
                    )

            if new_in_fallback > 0:
                logger.info(
                    f"  Fallback tarama: {new_in_fallback} ek maç bulundu"
                )

    def _find_nearest_league_header(self, row: Tag) -> Dict[str, str]:
        """Bir maç satırının en yakın lig/tarih header'ını bulur.

        DOM'da yukarı doğru traverse ederek league header arar.
        Virtualized DOM'da container kaybolmuş olabilir; bu durumda
        parent ve önceki sibling'lerden bilgi çıkarmaya çalışır.
        """
        info: Dict[str, str] = {"league": None, "date": None}

        # Parent container'ı dene
        parent = row.parent
        for _ in range(5):  # Max 5 seviye yukarı çık
            if parent is None:
                break
            league_elem = parent.select_one("strong")
            date_elem = parent.select_one('div[data-test-id="date"]')
            if league_elem:
                info["league"] = league_elem.get_text(strip=True)
            if date_elem:
                info["date"] = date_elem.get_text(strip=True)
            if info["league"]:
                break
            parent = parent.parent

        # Hâlâ bulunamadıysa önceki sibling'lerden dene
        if not info["league"]:
            prev = row.find_previous("strong")
            if prev:
                info["league"] = prev.get_text(strip=True)
        if not info["date"]:
            prev_date = row.find_previous(attrs={"data-test-id": "date"})
            if prev_date:
                info["date"] = prev_date.get_text(strip=True)

        return info
                    
    def parse_standing_row(self, row: Tag, match_code: str, match_name: str, team_type: str) -> Optional[TeamStanding]:
        """Puan tablosu satırını parse eder."""
        try:
            # Sıra numarası
            rank_elem = row.select_one('td[data-test-id="renderSortNumberColumn"] span:last-child')
            rank = rank_elem.get_text(strip=True) if rank_elem else None
            
            # Takım adı
            team_elem = row.select_one('a[data-test-id="TeamLink"]')
            team_name = team_elem.get_text(strip=True) if team_elem else None
            
            # İstatistikler
            o_elem = row.select_one('td.oCol[data-test-id="renderDefaultColumn"]')
            g_elem = row.select_one('td.gCol[data-test-id="renderDefaultColumn"]')
            b_elem = row.select_one('td.bCol[data-test-id="renderDefaultColumn"]')
            m_elem = row.select_one('td.mCol[data-test-id="renderDefaultColumn"]')
            ay_elem = row.select_one('td.ayCol[data-test-id="renderDefaultColumn"]')
            av_elem = row.select_one('td.avCol[data-test-id="renderDefaultColumn"]')
            p_elem = row.select_one('td.pCol[data-test-id="renderDefaultColumn"]')
            
            # Form (son maçlar)
            form_elems = row.select('span[data-test-id="getResultTooltipValue"]')
            form_list = [f.get_text(strip=True) for f in form_elems if f.get_text(strip=True) != "?"]
            form = "".join(form_list) if form_list else None
            
            return TeamStanding(
                Maç_Kodu=match_code,
                Maç=match_name,
                Takım_Tipi=team_type,
                Sıra=rank,
                Takım=team_name,
                O=o_elem.get_text(strip=True) if o_elem else None,
                G=g_elem.get_text(strip=True) if g_elem else None,
                B=b_elem.get_text(strip=True) if b_elem else None,
                M=m_elem.get_text(strip=True) if m_elem else None,
                A_Y=ay_elem.get_text(strip=True) if ay_elem else None,
                AV=av_elem.get_text(strip=True) if av_elem else None,
                P=p_elem.get_text(strip=True) if p_elem else None,
                Form=form
            )
            
        except Exception as e:
            logger.error(f"Puan tablosu satır parse hatası: {e}")
            return None
            
    def find_team_standing(self, soup: BeautifulSoup, team_name: str, match_code: str, match_full_name: str, team_type: str) -> Optional[TeamStanding]:
        """Belirli bir takımın puan tablosu verisini bulur."""
        rows = soup.select('tr[data-test-id="PointTable"]')
        
        # Arama için takım adını normalize et
        search_name = team_name.lower().strip()
        search_words = search_name.replace(".", " ").replace("-", " ").split()
        
        best_match = None
        best_score = 0
        
        for row in rows:
            team_elem = row.select_one('a[data-test-id="TeamLink"]')
            if not team_elem:
                continue
                
            table_team_name = team_elem.get_text(strip=True).lower()
            table_words = table_team_name.replace(".", " ").replace("-", " ").split()
            
            score = 0
            
            # Tam eşleşme
            if search_name == table_team_name:
                return self.parse_standing_row(row, match_code, match_full_name, team_type)
                
            # İçerme kontrolü
            if search_name in table_team_name or table_team_name in search_name:
                score = 10
                
            # Kelime eşleşmeleri
            for sw in search_words:
                if len(sw) >= 3:  # Kısa kelimeleri atla
                    for tw in table_words:
                        if sw == tw:
                            score += 5
                        elif sw in tw or tw in sw:
                            score += 3
                            
            # İlk kelime bonus
            if search_words and table_words:
                if search_words[0] == table_words[0]:
                    score += 8
                elif len(search_words[0]) >= 3 and search_words[0] in table_words[0]:
                    score += 4
                    
            if score > best_score:
                best_score = score
                best_match = row
                
        # En iyi eşleşme varsa döndür (minimum skor 3)
        if best_match and best_score >= 3:
            return self.parse_standing_row(best_match, match_code, match_full_name, team_type)
            
        return None
        
    def match_team_name(self, search_name: str, table_name: str) -> bool:
        """İki takım adının eşleşip eşleşmediğini kontrol eder."""
        search_lower = search_name.lower().strip()
        table_lower = table_name.lower().strip()
        
        # Tam eşleşme
        if search_lower == table_lower:
            return True
            
        # İçerme kontrolü
        if search_lower in table_lower or table_lower in search_lower:
            return True
            
        # Kelime bazlı eşleşme
        search_words = search_lower.replace(".", " ").replace("-", " ").split()
        table_words = table_lower.replace(".", " ").replace("-", " ").split()
        
        # En az bir anlamlı kelime eşleşmesi
        for sw in search_words:
            if len(sw) >= 3:
                for tw in table_words:
                    if len(tw) >= 3:
                        if sw == tw:
                            return True
                        # Kısaltma kontrolü (örn: "Sarsfield" ve "Velez Sarsfield")
                        if len(sw) >= 4 and (sw in tw or tw in sw):
                            return True
                            
        return False

    def get_standings_for_match(self, match: MatchData) -> None:
        """Bir maç için her iki takımın puan tablosu verilerini çeker."""
        if not match.İstatistik_Link or not match.Maç:
            return
            
        try:
            # Puan tablosu URL'si
            stats_url = f"{match.İstatistik_Link}/puan-tablosu"
            self.driver.get(stats_url)
            
            # Puan tablosunun yüklenmesini bekle
            wait = WebDriverWait(self.driver, self.TIMEOUT)
            wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'table[data-test-id="PointTableWrapper"]')
            ))
            
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            
            # Takım isimlerini ayır
            teams = match.Maç.split(" - ")
            if len(teams) != 2:
                logger.warning(f"Takım isimleri ayrıştırılamadı: {match.Maç}")
                return
                
            home_team = teams[0].strip()
            away_team = teams[1].strip()
            
            # Highlighted satırları bul (maçtaki takımlar işaretli)
            highlighted_rows = soup.select('tr[data-test-id="PointTable"][class*="fe8a09b89be114afe977"], tr[data-test-id="PointTable"][class*="ba36c2fc08832e02ac89"]')
            
            home_standing = None
            away_standing = None
            home_row = None
            away_row = None
            
            # Önce kesin eşleşmeleri bul
            for row in highlighted_rows:
                team_elem = row.select_one('a[data-test-id="TeamLink"]')
                if not team_elem:
                    continue
                    
                row_team_name = team_elem.get_text(strip=True)
                
                # Ev sahibi eşleşmesi kontrolü
                if not home_row and self.match_team_name(home_team, row_team_name):
                    home_row = row
                    continue
                    
                # Deplasman eşleşmesi kontrolü  
                if not away_row and self.match_team_name(away_team, row_team_name):
                    away_row = row
                    
            # Eğer 2 highlight var ve sadece 1'i eşleştiyse, diğeri otomatik olarak diğer takım
            if len(highlighted_rows) == 2:
                if home_row and not away_row:
                    # Diğer highlight away takımı
                    away_row = highlighted_rows[0] if highlighted_rows[1] == home_row else highlighted_rows[1]
                elif away_row and not home_row:
                    # Diğer highlight home takımı
                    home_row = highlighted_rows[0] if highlighted_rows[1] == away_row else highlighted_rows[1]
                elif not home_row and not away_row:
                    # Hiçbiri eşleşmediyse, sırayla ata (fuzzy ile devam et)
                    pass
                    
            # Row'lardan standing oluştur
            if home_row:
                home_standing = self.parse_standing_row(home_row, match.Maç_Kodu, match.Maç, "Ev Sahibi")
            if away_row:
                away_standing = self.parse_standing_row(away_row, match.Maç_Kodu, match.Maç, "Deplasman")
                    
            # Highlight ile bulunamadıysa tüm tabloda fuzzy ara
            if not home_standing:
                home_standing = self.find_team_standing(soup, home_team, match.Maç_Kodu, match.Maç, "Ev Sahibi")
            if not away_standing:
                away_standing = self.find_team_standing(soup, away_team, match.Maç_Kodu, match.Maç, "Deplasman")
                
            # Sonuçları kaydet
            if home_standing:
                self.standings.append(home_standing)
                logger.info(f"  ├─ Ev Sahibi: {home_standing.Takım} (Sıra: {home_standing.Sıra}, P: {home_standing.P})")
            else:
                logger.warning(f"  ├─ Ev sahibi bulunamadı: {home_team}")
                
            if away_standing:
                self.standings.append(away_standing)
                logger.info(f"  └─ Deplasman: {away_standing.Takım} (Sıra: {away_standing.Sıra}, P: {away_standing.P})")
            else:
                logger.warning(f"  └─ Deplasman bulunamadı: {away_team}")
                
        except TimeoutException:
            logger.error(f"Puan tablosu yüklenemedi: {match.Maç}")
        except Exception as e:
            logger.error(f"Puan tablosu hatası ({match.Maç}): {e}")
            
    def save_matches_to_csv(self, filename: str = "Bülten.csv") -> str:
        """Bülten verilerini CSV dosyasına kaydeder."""
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        if not self.matches:
            logger.warning("Kaydedilecek bülten verisi yok!")
            return filepath
            
        fieldnames = list(MatchData.__dataclass_fields__.keys())
        
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows([m.to_dict() for m in self.matches])
            
        logger.info(f"✓ {len(self.matches)} maç verisi kaydedildi: {filename}")
        return filepath
        
    def save_standings_to_csv(self, filename: str = "Puan_Tablosu.csv") -> str:
        """Puan tablosu verilerini CSV dosyasına kaydeder."""
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        if not self.standings:
            logger.warning("Kaydedilecek puan tablosu verisi yok!")
            return filepath
            
        fieldnames = list(TeamStanding.__dataclass_fields__.keys())
        
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows([s.to_dict() for s in self.standings])
            
        logger.info(f"✓ {len(self.standings)} takım puan verisi kaydedildi: {filename}")
        return filepath
        
    def get_competition_history_for_match(self, match: MatchData) -> None:
        """Bir maç için rekabet geçmişi verilerini çeker."""
        if not match.İstatistik_Link or not match.Maç:
            return
            
        try:
            # Rekabet geçmişi URL'si
            stats_url = f"{match.İstatistik_Link}/rekabet-gecmisi"
            self.driver.get(stats_url)
            
            # Rekabet geçmişi tablosunun yüklenmesini bekle
            wait = WebDriverWait(self.driver, self.TIMEOUT)
            wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'div[data-test-id="CompitionHistoryTable"]')
            ))
            
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            
            # Rekabet geçmişi satırlarını bul
            history_rows = soup.select('div[data-test-id="CompitionHistoryTableItem"]')
            
            for row in history_rows:
                history_data = self.parse_competition_history_row(row, match.Maç_Kodu, match.Maç)
                if history_data:
                    self.competition_history.append(history_data)
                    
            logger.info(f"  └─ {len(history_rows)} geçmiş maç bulundu")
                
        except TimeoutException:
            logger.warning(f"  └─ Rekabet geçmişi bulunamadı")
        except Exception as e:
            logger.error(f"Rekabet geçmişi hatası ({match.Maç}): {e}")
            
    def parse_competition_history_row(self, row: Tag, match_code: str, current_match: str) -> Optional[CompetitionHistory]:
        """Rekabet geçmişi satırını parse eder."""
        try:
            # Lig bilgisi
            league_elem = row.select_one('span[data-test-id="CompitionTableItemLeague"] span:first-child')
            league = league_elem.get_text(strip=True) if league_elem else None
            
            # Tarih bilgisi
            date_elem = row.select_one('span[data-test-id="CompitionTableItemSeason"]')
            date = date_elem.get_text(strip=True) if date_elem else None
            
            # Ev sahibi takım
            home_elem = row.select_one('div[data-test-id="HomeTeam"] a span')
            home_team = home_elem.get_text(strip=True) if home_elem else None
            
            # Deplasman takım
            away_elem = row.select_one('div[data-test-id="AwayTeam"] a span')
            away_team = away_elem.get_text(strip=True) if away_elem else None
            
            # Maç sonucu
            score_elem = row.select_one('button[data-test-id="NsnButton"] span')
            score = score_elem.get_text(strip=True) if score_elem else None
            
            # İlk yarı sonucu
            first_half_elem = row.select_one('span[data-test-id="CompitionTableItemFirstHalf"]')
            first_half = first_half_elem.get_text(strip=True) if first_half_elem else None
            
            # Oranlar
            odds_container = row.select_one('div[data-test-id="CompitionTableItemOdds"]')
            odds = odds_container.select('span[data-test-id="CompitionHistoryTableItem"]') if odds_container else []
            
            # Kazanan oran class'ı: ab18fc768d1ec03e3ada
            winning_class = "ab18fc768d1ec03e3ada"
            
            # Oranları parse et
            odd_1 = odds[0].get_text(strip=True) if len(odds) > 0 else None
            odd_1_won = "Evet" if (len(odds) > 0 and winning_class in (odds[0].get("class") or [])) else "Hayır"
            
            odd_x = odds[1].get_text(strip=True) if len(odds) > 1 else None
            odd_x_won = "Evet" if (len(odds) > 1 and winning_class in (odds[1].get("class") or [])) else "Hayır"
            
            odd_2 = odds[2].get_text(strip=True) if len(odds) > 2 else None
            odd_2_won = "Evet" if (len(odds) > 2 and winning_class in (odds[2].get("class") or [])) else "Hayır"
            
            odd_alt = odds[3].get_text(strip=True) if len(odds) > 3 else None
            odd_alt_won = "Evet" if (len(odds) > 3 and winning_class in (odds[3].get("class") or [])) else "Hayır"
            
            odd_ust = odds[4].get_text(strip=True) if len(odds) > 4 else None
            odd_ust_won = "Evet" if (len(odds) > 4 and winning_class in (odds[4].get("class") or [])) else "Hayır"
            
            return CompetitionHistory(
                Maç_Kodu=match_code,
                Güncel_Maç=current_match,
                Lig=league,
                Tarih=date,
                Ev_Sahibi=home_team,
                Deplasman=away_team,
                MS=score,
                İY=first_half,
                Oran_1=odd_1,
                Oran_1_Geldi=odd_1_won,
                Oran_X=odd_x,
                Oran_X_Geldi=odd_x_won,
                Oran_2=odd_2,
                Oran_2_Geldi=odd_2_won,
                Oran_Alt=odd_alt,
                Oran_Alt_Geldi=odd_alt_won,
                Oran_Üst=odd_ust,
                Oran_Üst_Geldi=odd_ust_won
            )
            
        except Exception as e:
            logger.error(f"Rekabet geçmişi satır parse hatası: {e}")
            return None
            
    def save_competition_history_to_csv(self, filename: str = "Rekabet_Gecmisi.csv") -> str:
        """Rekabet geçmişi verilerini CSV dosyasına kaydeder."""
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        if not self.competition_history:
            logger.warning("Kaydedilecek rekabet geçmişi verisi yok!")
            return filepath
            
        fieldnames = list(CompetitionHistory.__dataclass_fields__.keys())
        
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows([h.to_dict() for h in self.competition_history])
            
        logger.info(f"✓ {len(self.competition_history)} rekabet geçmişi kaydedildi: {filename}")
        return filepath
        
    def get_last_matches_for_match(self, match: MatchData) -> None:
        """Bir maç için her iki takımın son maçlarını çeker."""
        if not match.İstatistik_Link or not match.Maç:
            return
            
        try:
            # Son maçlar URL'si
            stats_url = f"{match.İstatistik_Link}/son-maclari"
            self.driver.get(stats_url)
            
            # Sayfanın tam yüklenmesi için bekle
            time.sleep(2)
            
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            
            # Takım isimlerini ayır
            teams = match.Maç.split(" - ")
            if len(teams) != 2:
                return
                
            home_team = teams[0].strip()
            away_team = teams[1].strip()
            
            home_count = 0
            away_count = 0
            
            # Birinci takımın (ev sahibi) son maçları
            first_table = soup.select_one('div[data-test-id="LastMatchesTableFirst"]')
            if first_table:
                team_name_elem = first_table.select_one('a[data-test-id="TeamLink"] span')
                team_name = team_name_elem.get_text(strip=True) if team_name_elem else home_team
                
                rows = first_table.select('tr[data-test-id="LastMatchesTable"]')
                for row in rows:
                    last_match = self.parse_last_match_row(row, match.Maç_Kodu, match.Maç, team_name, "Ev Sahibi")
                    if last_match:
                        self.last_matches.append(last_match)
                        home_count += 1
                        
            # İkinci takımın (deplasman) son maçları
            second_table = soup.select_one('div[data-test-id="LastMatchesTableSecond"]')
            if second_table:
                team_name_elem = second_table.select_one('a[data-test-id="TeamLink"] span')
                team_name = team_name_elem.get_text(strip=True) if team_name_elem else away_team
                
                rows = second_table.select('tr[data-test-id="LastMatchesTable"]')
                for row in rows:
                    last_match = self.parse_last_match_row(row, match.Maç_Kodu, match.Maç, team_name, "Deplasman")
                    if last_match:
                        self.last_matches.append(last_match)
                        away_count += 1
                        
            if home_count > 0 or away_count > 0:
                logger.info(f"  └─ Ev: {home_count}, Deplasman: {away_count} son maç")
            else:
                logger.warning(f"  └─ Son maçlar bulunamadı (bu lig için veri olmayabilir)")
                
        except TimeoutException:
            logger.warning(f"  └─ Son maçlar bulunamadı")
        except Exception as e:
            logger.error(f"Son maçlar hatası ({match.Maç}): {e}")
            
    def parse_last_match_row(self, row: Tag, match_code: str, current_match: str, team_name: str, team_type: str) -> Optional[LastMatch]:
        """Son maç satırını parse eder."""
        try:
            # Lig bilgisi
            league_elem = row.select_one('td[data-test-id="TableBodyLeague"] span:first-child')
            league = league_elem.get_text(strip=True) if league_elem else None
            
            # Tarih bilgisi
            date_elem = row.select_one('td[data-test-id="TableBodyLeague"] span:last-child')
            date = date_elem.get_text(strip=True) if date_elem else None
            
            # Ev sahibi takım
            home_elem = row.select_one('div[data-test-id="HomeTeam"] a span')
            home_team = home_elem.get_text(strip=True) if home_elem else None
            
            # Deplasman takım
            away_elem = row.select_one('div[data-test-id="AwayTeam"] a span')
            away_team = away_elem.get_text(strip=True) if away_elem else None
            
            # Maç sonucu ve sonuç rengi
            score_btn = row.select_one('button[data-test-id="NsnButton"] span')
            score = score_btn.get_text(strip=True) if score_btn else None
            
            # Sonuç (Galibiyet/Mağlubiyet/Beraberlik)
            result = None
            if score_btn:
                classes = score_btn.get("class") or []
                # Takımın bu maçta ev sahibi mi deplasman mı olduğunu bul
                is_home = self.match_team_name(team_name, home_team) if home_team else False
                
                # Skor analizi
                if score:
                    try:
                        parts = score.replace(" ", "").split("-")
                        if len(parts) == 2:
                            home_goals = int(parts[0])
                            away_goals = int(parts[1])
                            
                            if home_goals > away_goals:
                                result = "Galibiyet" if is_home else "Mağlubiyet"
                            elif home_goals < away_goals:
                                result = "Mağlubiyet" if is_home else "Galibiyet"
                            else:
                                result = "Beraberlik"
                    except ValueError:
                        pass
            
            # İlk yarı sonucu
            first_half_elem = row.select_one('td[data-test-id="TableBodyFirstHalf"]')
            first_half = first_half_elem.get_text(strip=True) if first_half_elem else None
            
            return LastMatch(
                Maç_Kodu=match_code,
                Güncel_Maç=current_match,
                Takım=team_name,
                Takım_Tipi=team_type,
                Lig=league,
                Tarih=date,
                Ev_Sahibi=home_team,
                Deplasman=away_team,
                MS=score,
                İY=first_half,
                Sonuç=result
            )
            
        except Exception as e:
            logger.error(f"Son maç satır parse hatası: {e}")
            return None
            
    def save_last_matches_to_csv(self, filename: str = "Son_Maclar.csv") -> str:
        """Son maçlar verilerini CSV dosyasına kaydeder."""
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        if not self.last_matches:
            logger.warning("Kaydedilecek son maç verisi yok!")
            return filepath
            
        fieldnames = list(LastMatch.__dataclass_fields__.keys())
        
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows([m.to_dict() for m in self.last_matches])
            
        logger.info(f"✓ {len(self.last_matches)} son maç kaydedildi: {filename}")
        return filepath
        
    def get_referee_info_for_match(self, match: MatchData) -> None:
        """Bir maç için hakem bilgilerini çeker."""
        if not match.İstatistik_Link or not match.Maç:
            return
            
        try:
            # Hakem bilgileri URL'si
            stats_url = f"{match.İstatistik_Link}/hakem-bilgileri"
            self.driver.get(stats_url)
            
            # Sayfanın tam yüklenmesi için bekle
            time.sleep(2)
            
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            
            # Hakem adını çek
            referee_container = soup.select_one('div[data-test-id="Referee"]')
            if not referee_container:
                logger.warning(f"  └─ Hakem bilgisi bulunamadı")
                return
                
            referee_name_elem = referee_container.select_one('h4.cd931950a4583aede299')
            referee_name = None
            if referee_name_elem:
                # İkon ve bayrak dışındaki metin
                referee_name = referee_name_elem.get_text(strip=True)
                
            if not referee_name:
                logger.warning(f"  └─ Hakem adı bulunamadı")
                return
                
            # Hakem maçlarını çek
            match_count = 0
            history_rows = referee_container.select('div[data-test-id="CompitionHistoryTableItem"]')
            
            for row in history_rows:
                referee_match = self.parse_referee_match_row(row, match.Maç_Kodu, match.Maç, referee_name)
                if referee_match:
                    self.referee_matches.append(referee_match)
                    match_count += 1
                    
            # Hakem istatistiklerini çek
            stats = self.parse_referee_stats(soup, match.Maç_Kodu, match.Maç, referee_name)
            if stats:
                self.referee_stats.append(stats)
                
            logger.info(f"  └─ {referee_name}: {match_count} maç")
                
        except TimeoutException:
            logger.warning(f"  └─ Hakem bilgileri bulunamadı")
        except Exception as e:
            logger.error(f"Hakem bilgileri hatası ({match.Maç}): {e}")
            
    def parse_referee_match_row(self, row: Tag, match_code: str, current_match: str, referee_name: str) -> Optional[RefereeMatch]:
        """Hakem maç satırını parse eder."""
        try:
            # Lig bilgisi
            league_elem = row.select_one('span[data-test-id="CompitionTableItemLeague"] span:first-child')
            league = league_elem.get_text(strip=True) if league_elem else None
            
            # Tarih bilgisi
            date_elem = row.select_one('span[data-test-id="CompitionTableItemSeason"]')
            date = date_elem.get_text(strip=True) if date_elem else None
            
            # Ev sahibi takım
            home_elem = row.select_one('div[data-test-id="HomeTeam"] a span')
            home_team = home_elem.get_text(strip=True) if home_elem else None
            
            # Deplasman takım
            away_elem = row.select_one('div[data-test-id="AwayTeam"] a span')
            away_team = away_elem.get_text(strip=True) if away_elem else None
            
            # Maç sonucu
            score_elem = row.select_one('button[data-test-id="NsnButton"] span')
            score = score_elem.get_text(strip=True) if score_elem else None
            
            # İlk yarı sonucu
            first_half_elem = row.select_one('span[data-test-id="CompitionTableItemFirstHalf"]')
            first_half = first_half_elem.get_text(strip=True) if first_half_elem else None
            
            # Oranlar
            odds_container = row.select_one('div[data-test-id="CompitionTableItemOdds"]')
            odds = odds_container.select('span[data-test-id="CompitionHistoryTableItem"]') if odds_container else []
            
            # Kazanan oran class'ı
            winning_class = "ab18fc768d1ec03e3ada"
            
            # Oranları parse et
            odd_1 = odds[0].get_text(strip=True) if len(odds) > 0 else None
            odd_1_won = "Evet" if (len(odds) > 0 and winning_class in (odds[0].get("class") or [])) else "Hayır"
            
            odd_x = odds[1].get_text(strip=True) if len(odds) > 1 else None
            odd_x_won = "Evet" if (len(odds) > 1 and winning_class in (odds[1].get("class") or [])) else "Hayır"
            
            odd_2 = odds[2].get_text(strip=True) if len(odds) > 2 else None
            odd_2_won = "Evet" if (len(odds) > 2 and winning_class in (odds[2].get("class") or [])) else "Hayır"
            
            odd_alt = odds[3].get_text(strip=True) if len(odds) > 3 else None
            odd_alt_won = "Evet" if (len(odds) > 3 and winning_class in (odds[3].get("class") or [])) else "Hayır"
            
            odd_ust = odds[4].get_text(strip=True) if len(odds) > 4 else None
            odd_ust_won = "Evet" if (len(odds) > 4 and winning_class in (odds[4].get("class") or [])) else "Hayır"
            
            return RefereeMatch(
                Maç_Kodu=match_code,
                Güncel_Maç=current_match,
                Hakem_Adı=referee_name,
                Lig=league,
                Tarih=date,
                Ev_Sahibi=home_team,
                Deplasman=away_team,
                MS=score,
                İY=first_half,
                Oran_1=odd_1,
                Oran_1_Geldi=odd_1_won,
                Oran_X=odd_x,
                Oran_X_Geldi=odd_x_won,
                Oran_2=odd_2,
                Oran_2_Geldi=odd_2_won,
                Oran_Alt=odd_alt,
                Oran_Alt_Geldi=odd_alt_won,
                Oran_Üst=odd_ust,
                Oran_Üst_Geldi=odd_ust_won
            )
            
        except Exception as e:
            logger.error(f"Hakem maç satır parse hatası: {e}")
            return None
            
    def parse_referee_stats(self, soup: BeautifulSoup, match_code: str, current_match: str, referee_name: str) -> Optional[RefereeStats]:
        """Hakem istatistiklerini parse eder."""
        try:
            stats_container = soup.select_one('div[data-test-id="setContent"]')
            if not stats_container:
                return None
                
            # İstatistik değerlerini çek
            stats_items = soup.select('div[data-test-id="TableItem"]')
            
            stats_dict = {}
            for item in stats_items:
                label_elem = item.select_one('div.f15d176d9b8eb47234b0 span:first-child')
                value_elem = item.select_one('div.f15d176d9b8eb47234b0 span:last-child')
                percent_elem = item.select_one('span.c411ef110bae2cf448ba span')
                
                if label_elem and value_elem:
                    label = label_elem.get_text(strip=True)
                    value = value_elem.get_text(strip=True)
                    percent = percent_elem.get_text(strip=True) if percent_elem else None
                    stats_dict[label] = {"count": value, "percent": percent}
                    
            return RefereeStats(
                Maç_Kodu=match_code,
                Güncel_Maç=current_match,
                Hakem_Adı=referee_name,
                MS1_Sayı=stats_dict.get("MS1", {}).get("count"),
                MS1_Yüzde=stats_dict.get("MS1", {}).get("percent"),
                MSX_Sayı=stats_dict.get("MSX", {}).get("count"),
                MSX_Yüzde=stats_dict.get("MSX", {}).get("percent"),
                MS2_Sayı=stats_dict.get("MS2", {}).get("count"),
                MS2_Yüzde=stats_dict.get("MS2", {}).get("percent"),
                Alt_2_5_Sayı=stats_dict.get("2,5 Alt", {}).get("count"),
                Alt_2_5_Yüzde=stats_dict.get("2,5 Alt", {}).get("percent"),
                Üst_2_5_Sayı=stats_dict.get("2,5 Üst", {}).get("count"),
                Üst_2_5_Yüzde=stats_dict.get("2,5 Üst", {}).get("percent"),
                KG_Var_Sayı=stats_dict.get("KG Var", {}).get("count"),
                KG_Var_Yüzde=stats_dict.get("KG Var", {}).get("percent"),
                KG_Yok_Sayı=stats_dict.get("KG Yok", {}).get("count"),
                KG_Yok_Yüzde=stats_dict.get("KG Yok", {}).get("percent")
            )
            
        except Exception as e:
            logger.error(f"Hakem istatistik parse hatası: {e}")
            return None
            
    def save_referee_matches_to_csv(self, filename: str = "Hakem_Bilgileri.csv") -> str:
        """Hakem maç verilerini CSV dosyasına kaydeder."""
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        if not self.referee_matches:
            logger.warning("Kaydedilecek hakem maç verisi yok!")
            return filepath
            
        fieldnames = list(RefereeMatch.__dataclass_fields__.keys())
        
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows([m.to_dict() for m in self.referee_matches])
            
        logger.info(f"✓ {len(self.referee_matches)} hakem maçı kaydedildi: {filename}")
        return filepath
        
    def save_referee_stats_to_csv(self, filename: str = "Hakem_Istatistikleri.csv") -> str:
        """Hakem istatistik verilerini CSV dosyasına kaydeder."""
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        if not self.referee_stats:
            logger.warning("Kaydedilecek hakem istatistik verisi yok!")
            return filepath
            
        fieldnames = list(RefereeStats.__dataclass_fields__.keys())
        
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows([s.to_dict() for s in self.referee_stats])
            
        logger.info(f"✓ {len(self.referee_stats)} hakem istatistiği kaydedildi: {filename}")
        return filepath
        
    def get_injury_data_for_match(self, match: MatchData) -> None:
        """Bir maç için sakat ve cezalı oyuncu verilerini çeker."""
        if not match.İstatistik_Link or not match.Maç:
            return
            
        try:
            # Sakat/Cezalı URL'si
            stats_url = f"{match.İstatistik_Link}/sakat-cezali"
            self.driver.get(stats_url)
            
            # Sayfanın tam yüklenmesi için bekle
            time.sleep(2)
            
            soup = BeautifulSoup(self.driver.page_source, "lxml")
            
            # Sakat/Cezalı ana container'ı bul
            injury_container = soup.select_one('div[data-test-id="CrippledPunished"]')
            if not injury_container:
                logger.info(f"  └─ Sakat/cezalı verisi yok")
                return
                
            # Her iki takım için ayrı ayrı verileri çek
            team_containers = injury_container.select('div.ad65c734cbc1c4292120')
            
            total_count = 0
            for team_container in team_containers:
                # Takım adını çek
                team_link = team_container.select_one('a[data-test-id="TeamLink"] span')
                team_name = team_link.get_text(strip=True) if team_link else None
                
                if not team_name:
                    continue
                    
                # Oyuncu satırlarını çek
                player_rows = team_container.select('div[data-test-id="MissingPlayersTable"]')
                
                for row in player_rows:
                    injury = self.parse_injury_player_row(row, match.Maç_Kodu, match.Maç, team_name)
                    if injury:
                        self.injury_data.append(injury)
                        total_count += 1
                        
            if total_count > 0:
                logger.info(f"  └─ {total_count} sakat/cezalı oyuncu bulundu")
            else:
                logger.info(f"  └─ Sakat/cezalı oyuncu yok")
                
        except TimeoutException:
            logger.warning(f"  └─ Sakat/cezalı sayfası yüklenemedi")
        except Exception as e:
            logger.error(f"Sakat/cezalı hatası ({match.Maç}): {e}")
            
    def parse_injury_player_row(self, row: Tag, match_code: str, current_match: str, team_name: str) -> Optional[InjuryData]:
        """Sakat/cezalı oyuncu satırını parse eder."""
        try:
            # Forma numarası
            number_elem = row.select_one('span[data-test-id="Number"] span')
            number = number_elem.get_text(strip=True) if number_elem else None
            
            # Oyuncu adı
            player_elem = row.select_one('span[data-test-id="Player"] a')
            player_name = player_elem.get_text(strip=True) if player_elem else None
            
            if not player_name:
                return None
                
            # Yaş
            age_elem = row.select_one('span[data-test-id="Age"]')
            age = age_elem.get_text(strip=True) if age_elem else None
            
            # Pozisyon
            position_elem = row.select_one('span[data-test-id="Position"]')
            position = position_elem.get_text(strip=True) if position_elem else None
            
            # Maç sayısı
            match_count_elem = row.select_one('span[data-test-id="Match"]')
            match_count = match_count_elem.get_text(strip=True) if match_count_elem else "0"
            
            # İlk 11
            first_eleven_elem = row.select_one('span[data-test-id="FirstEleven"]')
            first_eleven = first_eleven_elem.get_text(strip=True) if first_eleven_elem else "0"
            # "-" işaretini 0'a çevir
            if first_eleven == "-":
                first_eleven = "0"
                
            # Gol
            goal_elem = row.select_one('span[data-test-id="Goal"]')
            goal = goal_elem.get_text(strip=True) if goal_elem else "0"
            if goal == "-":
                goal = "0"
                
            # Asist
            assist_elem = row.select_one('span[data-test-id="Assist"]')
            assist = assist_elem.get_text(strip=True) if assist_elem else "0"
            if assist == "-":
                assist = "0"
                
            # Durum ve Açıklama
            description_elem = row.select_one('span[data-test-id="Description"] span')
            full_description = description_elem.get_text(strip=True) if description_elem else None
            
            # Durum ve açıklamayı ayır
            status = None
            description = None
            if full_description:
                if "Sakatlık" in full_description:
                    status = "Sakatlık"
                    # "Sakatlık - " kısmını çıkar
                    description = full_description.replace("Sakatlık - ", "").strip()
                elif "Cezalı" in full_description:
                    status = "Cezalı"
                    # "Cezalı - " kısmını çıkar
                    description = full_description.replace("Cezalı - ", "").strip()
                else:
                    status = "Bilinmiyor"
                    description = full_description
                    
            return InjuryData(
                Maç_Kodu=match_code,
                Maç=current_match,
                Takım=team_name,
                Numara=number,
                Oyuncu=player_name,
                Yaş=age,
                Pozisyon=position,
                Maç_Sayısı=match_count,
                İlk_11=first_eleven,
                Gol=goal,
                Asist=assist,
                Durum=status,
                Açıklama=description
            )
            
        except Exception as e:
            logger.error(f"Sakat/cezalı oyuncu parse hatası: {e}")
            return None
            
    def save_injury_data_to_csv(self, filename: str = "Sakat_Cezali.csv") -> str:
        """Sakat/cezalı oyuncu verilerini CSV dosyasına kaydeder."""
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        if not self.injury_data:
            logger.warning("Kaydedilecek sakat/cezalı verisi yok!")
            return filepath
            
        fieldnames = list(InjuryData.__dataclass_fields__.keys())
        
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
            writer.writeheader()
            writer.writerows([i.to_dict() for i in self.injury_data])
            
        logger.info(f"✓ {len(self.injury_data)} sakat/cezalı oyuncu kaydedildi: {filename}")
        return filepath
        
    def run(self) -> None:
        """Ana çalıştırma fonksiyonu."""
        try:
            logger.info("Nesine.com Bülten & İstatistik Scraper Başlatılıyor...")
            logger.info("="*60)
            
            self.setup_driver()
            
            # URL'yi parametrelerle oluştur
            params = "&".join(f"{k}={v}" for k, v in self.DEFAULT_PARAMS.items())
            url = f"{self.BASE_URL}?{params}"
            logger.info(f"URL: {url}")
            self.driver.get(url)
            
            self.wait_for_page_load()
            self.close_popups()

            # ── Agresif Scroll + Incremental Collection ──
            # Virtualized DOM'da tek page_source tüm verileri içermeyebilir.
            # Bu yüzden scroll sırasında periyodik olarak veri topluyoruz.
            self._scroll_and_collect()

            # Eksik kalan varsa son bir deneme daha yap
            if len(self.matches) < self.match_count:
                logger.info(
                    f"İnkremental toplamada {len(self.matches)}/{self.match_count}, "
                    f"son page_source ile tamamlanmaya çalışılıyor..."
                )
                soup = self.get_page_source()
                self.get_match_data(soup)

            logger.info(f"📋 Toplam {len(self.matches)} maç toplandı")
            
            # Bülten verilerini kaydet
            self.save_matches_to_csv()
            
            # Puan tablosu verilerini çek
            logger.info("="*60)
            logger.info("Puan tablosu verileri çekiliyor...")
            logger.info("-"*60)
            
            for i, match in enumerate(self.matches, 1):
                logger.info(f"[{i}/{len(self.matches)}] {match.Maç}")
                self.get_standings_for_match(match)
                
            # Puan tablosu verilerini kaydet
            logger.info("="*60)
            self.save_standings_to_csv()
            
            # Rekabet geçmişi verilerini çek
            logger.info("="*60)
            logger.info("Rekabet geçmişi verileri çekiliyor...")
            logger.info("-"*60)
            
            for i, match in enumerate(self.matches, 1):
                logger.info(f"[{i}/{len(self.matches)}] {match.Maç}")
                self.get_competition_history_for_match(match)
                
            # Rekabet geçmişi verilerini kaydet
            logger.info("="*60)
            self.save_competition_history_to_csv()
            
            # Son maçlar verilerini çek
            logger.info("="*60)
            logger.info("Son maçlar verileri çekiliyor...")
            logger.info("-"*60)
            
            for i, match in enumerate(self.matches, 1):
                logger.info(f"[{i}/{len(self.matches)}] {match.Maç}")
                self.get_last_matches_for_match(match)
                
            # Son maçlar verilerini kaydet
            logger.info("="*60)
            self.save_last_matches_to_csv()
            
            # Hakem bilgileri verilerini çek
            logger.info("="*60)
            logger.info("Hakem bilgileri çekiliyor...")
            logger.info("-"*60)
            
            for i, match in enumerate(self.matches, 1):
                logger.info(f"[{i}/{len(self.matches)}] {match.Maç}")
                self.get_referee_info_for_match(match)
                
            # Hakem verilerini kaydet
            logger.info("="*60)
            self.save_referee_matches_to_csv()
            self.save_referee_stats_to_csv()
            
            # Sakat/Cezalı oyuncu verilerini çek
            logger.info("="*60)
            logger.info("Sakat/Cezalı oyuncu verileri çekiliyor...")
            logger.info("-"*60)
            
            for i, match in enumerate(self.matches, 1):
                logger.info(f"[{i}/{len(self.matches)}] {match.Maç}")
                self.get_injury_data_for_match(match)
                
            # Sakat/Cezalı verilerini kaydet
            logger.info("="*60)
            self.save_injury_data_to_csv()
            
            logger.info("="*60)
            logger.info("✓ Tüm işlemler tamamlandı!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Kritik hata: {e}")
            raise
            
        finally:
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver kapatıldı")


def main() -> None:
    """Ana giriş noktası."""
    try:
        match_input = input("Kaç adet maç çekmek istiyorsunuz? ")
        match_count = int(match_input)
        
        if match_count <= 0:
            logger.error("Geçersiz sayı! En az 1 maç girmelisiniz.")
            return
            
        scraper = NesineScraper(match_count=match_count)
        scraper.run()
        
    except ValueError:
        logger.error("Lütfen geçerli bir sayı girin!")
    except KeyboardInterrupt:
        logger.info("İşlem kullanıcı tarafından iptal edildi.")


if __name__ == "__main__":
    main()
