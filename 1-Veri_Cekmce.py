from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os

class NesineScraper:
    def __init__(self):
        self.driver = None
        self.matches_data = []
        self.league_table_data = []
        self.competition_history_data = []
        self.last_matches_data = []

    def setup_driver(self, headless=True):
        print("ChromeDriver yükleniyor (arka plan modu)...")
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            print("Chrome arka planda çalışacak")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-images')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.set_window_size(1920, 1080)
            print("✅ ChromeDriver başarıyla başlatıldı!")
            return True
        except Exception as e:
            print(f"❌ ChromeDriver hatası: {e}")
            return False

    def close_live_events(self):
        try:
            print("Canlı maçlar kapatılıyor...")
            time.sleep(1)
            selectors = [
                ".close-col a",
                ".close-col i.ni-close-rounded",
                "a[href*='LiveEventsFilterClick']"
            ]
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        if element.is_displayed():
                            self.driver.execute_script("arguments[0].click();", element)
                            print("✅ Canlı maçlar kapatıldı")
                            time.sleep(1)
                            return True
                except:
                    continue
            print("ℹ️ Canlı maçlar zaten kapalı")
            return False
        except Exception as e:
            print(f"⚠️ Canlı maçlar kapatma hatası: {e}")
            return False

    def get_exact_matches(self, target_count):
        print(f"🎯 Hedef: Tam {target_count} maç")
        scroll_attempts = 0
        max_scroll_attempts = 15
        no_change_count = 0
        while scroll_attempts < max_scroll_attempts:
            matches = self.driver.find_elements(By.CSS_SELECTOR, ".event-list.pre-event")
            current_count = len(matches)
            print(f"📊 Mevcut maç sayısı: {current_count}")
            if current_count >= target_count:
                print(f"✅ Hedef sayıya ulaşıldı: {current_count} maç")
                break
            previous_count = current_count
            self.driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(1.5)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_matches = self.driver.find_elements(By.CSS_SELECTOR, ".event-list.pre-event")
            if len(new_matches) == previous_count:
                no_change_count += 1
                if no_change_count >= 3:
                    print("⚠️ Daha fazla maç yüklenmiyor")
                    break
            else:
                no_change_count = 0
            scroll_attempts += 1
        final_matches = self.driver.find_elements(By.CSS_SELECTOR, ".event-list.pre-event")
        print(f"📈 Toplam yüklenen maç: {len(final_matches)}")
        return len(final_matches)

    def extract_all_odds(self, match_element):
        odds_data = {
            "MS_1": "-", "MS_X": "-", "MS_2": "-",
            "Alt": "-", "Üst": "-",
            "HND": "-", "HND_1": "-", "HND_X": "-", "HND_2": "-",
            "CS_1X": "-", "CS_12": "-", "CS_X2": "-",
            "KG_Var": "-", "KG_Yok": "-"
        }
        try:
            event_rows = match_element.find_elements(By.CSS_SELECTOR, ".event-row")
            col_02_sections = []
            col_03_sections = []
            col_04_sections = []
            for row in event_rows:
                class_list = row.get_attribute("class")
                if "col-02" in class_list:
                    col_02_sections.append(row)
                elif "col-03" in class_list:
                    col_03_sections.append(row)
                elif "col-04" in class_list:
                    col_04_sections.append(row)
            if len(col_03_sections) >= 1:
                try:
                    ms_odds = col_03_sections[0].find_elements(By.CSS_SELECTOR, ".cell.outcome .odd")
                    if len(ms_odds) >= 3:
                        odds_data["MS_1"] = ms_odds[0].text.strip()
                        odds_data["MS_X"] = ms_odds[1].text.strip()
                        odds_data["MS_2"] = ms_odds[2].text.strip()
                except: pass
            if len(col_02_sections) >= 1:
                try:
                    alt_ust_odds = col_02_sections[0].find_elements(By.CSS_SELECTOR, ".cell.outcome .odd")
                    if len(alt_ust_odds) >= 2:
                        odds_data["Alt"] = alt_ust_odds[0].text.strip()
                        odds_data["Üst"] = alt_ust_odds[1].text.strip()
                except: pass
            if len(col_04_sections) >= 1:
                try:
                    hnd_section = col_04_sections[0]
                    hnd_value_elem = hnd_section.find_element(By.CSS_SELECTOR, ".cell.empty-odd.hnd .odd")
                    odds_data["HND"] = hnd_value_elem.text.strip()
                    hnd_odds = hnd_section.find_elements(By.CSS_SELECTOR, ".cell.outcome .odd")
                    if len(hnd_odds) >= 3:
                        odds_data["HND_1"] = hnd_odds[0].text.strip()
                        odds_data["HND_X"] = hnd_odds[1].text.strip()
                        odds_data["HND_2"] = hnd_odds[2].text.strip()
                except: pass
            if len(col_03_sections) >= 2:
                try:
                    cs_odds = col_03_sections[1].find_elements(By.CSS_SELECTOR, ".cell.outcome .odd")
                    if len(cs_odds) >= 3:
                        odds_data["CS_1X"] = cs_odds[0].text.strip()
                        odds_data["CS_12"] = cs_odds[1].text.strip()
                        odds_data["CS_X2"] = cs_odds[2].text.strip()
                except: pass
            if len(col_02_sections) >= 2:
                try:
                    kg_odds = col_02_sections[1].find_elements(By.CSS_SELECTOR, ".cell.outcome .odd")
                    if len(kg_odds) >= 2:
                        odds_data["KG_Var"] = kg_odds[0].text.strip()
                        odds_data["KG_Yok"] = kg_odds[1].text.strip()
                except: pass
        except Exception as e:
            print(f"⚠️ Oran çıkarma hatası: {e}")
        return odds_data

    def extract_match_data(self, target_count):
        print(f"📊 Veri çıkarma başlıyor - Hedef: {target_count} maç")
        matches = self.driver.find_elements(By.CSS_SELECTOR, ".event-list.pre-event")
        matches_to_process = matches[:target_count]
        print(f"🔄 İşlenecek maç sayısı: {len(matches_to_process)}")
        current_league = ""
        for i, match in enumerate(matches_to_process):
            try:
                print(f"⚽ Maç {i+1}/{target_count} işleniyor...")
                try:
                    league = self.driver.execute_script("""
                        var match = arguments[0];
                        var previous = match.previousElementSibling;
                        while (previous) {
                            if (previous.classList.contains('lea-hdr')) {
                                var nameElement = previous.querySelector('.name');
                                return nameElement ? nameElement.textContent.trim() : '';
                            }
                            previous = previous.previousElementSibling;
                        }
                        return '';
                    """, match)
                    if league:
                        current_league = league
                except: pass
                match_time = "-"
                try:
                    time_element = match.find_element(By.CSS_SELECTOR, ".time span")
                    match_time = time_element.text.strip()
                except: pass
                teams = "-"
                match_link = "-"
                try:
                    name_element = match.find_element(By.CSS_SELECTOR, ".name a")
                    teams = name_element.text.strip()
                    match_link = name_element.get_attribute("href")
                    if not match_link:
                        match_link = "-"
                except: pass
                odds_data = self.extract_all_odds(match)
                match_data = {
                    "Saat": match_time,
                    "Lig": current_league,
                    "Takımlar": teams,
                    "Link": match_link,
                    **odds_data
                }
                self.matches_data.append(match_data)
                print(f"✅ Maç {i+1}: {teams} - {match_time}")
                print(f"   🔗 Link: {match_link}")
                print(f"   📈 MS: {odds_data['MS_1']}/{odds_data['MS_X']}/{odds_data['MS_2']}")
                print(f"   🔄 Alt/Üst: {odds_data['Alt']}/{odds_data['Üst']}")
                print(f"   ⚖️ HND: {odds_data['HND']} - {odds_data['HND_1']}/{odds_data['HND_X']}/{odds_data['HND_2']}")
                print(f"   🎯 Çifte Şans: {odds_data['CS_1X']}/{odds_data['CS_12']}/{odds_data['CS_X2']}")
                print(f"   ⚽ KG: {odds_data['KG_Var']}/{odds_data['KG_Yok']}")
                if len(self.matches_data) >= target_count:
                    print(f"🎯 Hedef sayıya ulaşıldı: {len(self.matches_data)} maç")
                    break
            except Exception as e:
                print(f"❌ Maç {i+1} işlenirken hata: {e}")
                continue
        print(f"📊 Veri çıkarma tamamlandı. Toplam: {len(self.matches_data)} maç")

    def scrape_league_table(self, match_link, teams, league):
        try:
            print(f"🔍 İstatistik sayfasına gidiliyor: {match_link}")
            self.driver.get(match_link)
            time.sleep(3)
            try:
                wait = WebDriverWait(self.driver, 10)
                stats_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-test-id="MenuItem"][href*="puan-tablosu"]')))
                self.driver.execute_script("arguments[0].click();", stats_button)
                print("✅ Puan tablosu sekmesine tıklandı")
                time.sleep(3)
            except Exception as e:
                print(f"⚠️ Puan tablosu sekmesi bulunamadı: {e}")
                return False
            try:
                rows = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-test-id="PointTable"]')))
                for i, row in enumerate(rows):
                    try:
                        rank = "-"
                        try:
                            rank_elem = row.find_element(By.CSS_SELECTOR, '[data-test-id="renderSortNumberColumn"] .ab38fbca15d2af4bc139')
                            rank = rank_elem.text.strip()
                        except: pass
                        team_name = "-"
                        try:
                            team_elem = row.find_element(By.CSS_SELECTOR, '[data-test-id="TeamLink"]')
                            team_name = team_elem.text.strip()
                        except: pass
                        oynanan = "-"
                        try:
                            oynanan = row.find_element(By.CSS_SELECTOR, '.oCol').text.strip()
                        except: pass
                        galibiyet = "-"
                        try:
                            galibiyet = row.find_element(By.CSS_SELECTOR, '.gCol').text.strip()
                        except: pass
                        beraberlik = "-"
                        try:
                            beraberlik = row.find_element(By.CSS_SELECTOR, '.bCol').text.strip()
                        except: pass
                        maglubiyet = "-"
                        try:
                            maglubiyet = row.find_element(By.CSS_SELECTOR, '.mCol').text.strip()
                        except: pass
                        atilan_yenilen = "-"
                        atilan = "-"
                        yenilen = "-"
                        try:
                            atilan_yenilen = row.find_element(By.CSS_SELECTOR, '.ayCol').text.strip()
                            if ":" in atilan_yenilen:
                                atilan, yenilen = atilan_yenilen.split(":")
                        except: pass
                        averaj = "-"
                        try:
                            averaj = row.find_element(By.CSS_SELECTOR, '.avCol').text.strip()
                        except: pass
                        puan = "-"
                        try:
                            puan = row.find_element(By.CSS_SELECTOR, '.pCol').text.strip()
                        except: pass
                        team_data = {
                            "Maç": teams,
                            "Lig": league,
                            "Sıra": rank,
                            "Takım": team_name,
                            "Oynanan": oynanan,
                            "Galibiyet": galibiyet,
                            "Beraberlik": beraberlik,
                            "Mağlubiyet": maglubiyet,
                            "Atılan": atilan,
                            "Yenilen": yenilen,
                            "Averaj": averaj,
                            "Puan": puan
                        }
                        self.league_table_data.append(team_data)
                    except Exception as e:
                        print(f"⚠️ Satır {i+1} işlenirken hata: {e}")
                        continue
                print(f"✅ Puan tablosu verisi çekildi: {len(rows)} takım")
                return True
            except Exception as e:
                print(f"⚠️ Puan tablosu verisi çekilemedi: {e}")
                return False
        except Exception as e:
            print(f"❌ İstatistik sayfası hatası: {e}")
            return False

    def scrape_competition_history(self, match_link, teams, league):
        try:
            print(f"🔍 Rekabet geçmişi sayfasına gidiliyor: {match_link}")
            self.driver.get(match_link)
            time.sleep(3)
            try:
                wait = WebDriverWait(self.driver, 10)
                history_button = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, '[data-test-id="MenuItem"][href*="rekabet-gecmisi"]')
                ))
                self.driver.execute_script("arguments[0].click();", history_button)
                print("✅ Rekabet geçmişi sekmesine tıklandı")
                time.sleep(3)
            except Exception as e:
                print(f"⚠️ Rekabet geçmişi sekmesi bulunamadı: {e}")
                return False
            try:
                history_items = self.driver.find_elements(By.CSS_SELECTOR, '[data-test-id="CompitionHistoryTableItem"]')
                for item in history_items:
                    try:
                        lig = "-"
                        sezon = "-"
                        try:
                            lig_sezon_elem = item.find_element(By.CSS_SELECTOR, '[data-test-id="CompitionTableItemLeague"]')
                            spans = lig_sezon_elem.find_elements(By.TAG_NAME, "span")
                            if len(spans) >= 2:
                                lig = spans[0].text.strip()
                                sezon = spans[1].text.strip()
                            elif len(spans) == 1:
                                lig = spans[0].text.strip()
                        except: pass
                        tarih = "-"
                        try:
                            tarih_elem = item.find_element(By.CSS_SELECTOR, '[data-test-id="CompitionTableItemSeason"]')
                            tarih = tarih_elem.text.strip()
                        except: pass
                        ms = "-"
                        try:
                            ms_elem = item.find_element(By.CSS_SELECTOR, '[data-test-id="NsnButton"] span')
                            ms = ms_elem.text.strip()
                        except: pass
                        iy = "-"
                        try:
                            iy_elem = item.find_element(By.CSS_SELECTOR, '[data-test-id="CompitionTableItemFirstHalf"]')
                            iy = iy_elem.text.strip()
                        except: pass
                        oranlar = ["-", "-", "-", "-", "-"]
                        try:
                            odds_elem = item.find_element(By.CSS_SELECTOR, '[data-test-id="CompitionTableItemOdds"]')
                            odds_spans = odds_elem.find_elements(By.CSS_SELECTOR, 'span[data-test-id="CompitionHistoryTableItem"]')
                            for i, odd in enumerate(odds_spans[:5]):
                                oranlar[i] = odd.text.strip()
                            if not odds_spans:
                                p_tags = odds_elem.find_elements(By.TAG_NAME, "p")
                                if p_tags and "oranlarına erişilememektedir" in p_tags[0].text:
                                    oranlar = ["-"]*5
                        except: pass

                        if lig != "-" and tarih != "-" and ms != "-":
                            history_data = {
                                "Ana_Maç": teams,
                                "Ana_Lig": league,
                                "Lig": lig,
                                "Sezon": sezon,
                                "Tarih": tarih,
                                "MS": ms,
                                "İY": iy,
                                "1": oranlar[0],
                                "X": oranlar[1],
                                "2": oranlar[2],
                                "Alt": oranlar[3],
                                "Üst": oranlar[4]
                            }
                            self.competition_history_data.append(history_data)
                    except Exception as e:
                        print(f"⚠️ Rekabet geçmişi satırı işlenirken hata: {e}")
                        continue
                print(f"✅ Rekabet geçmişi verisi çekildi: {len(self.competition_history_data)} maç")
                return True
            except Exception as e:
                print(f"⚠️ Rekabet geçmişi verisi çekilemedi: {e}")
                return False
        except Exception as e:
            print(f"❌ Rekabet geçmişi sayfası hatası: {e}")
            return False

    def scrape_last_matches(self, match_link, teams, league):
        try:
            print(f"🕑 Son Maçlar sekmesine gidiliyor: {match_link}")
            self.driver.get(match_link)
            time.sleep(3)
            wait = WebDriverWait(self.driver, 10)
            try:
                last_matches_button = wait.until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, '[data-test-id="MenuItem"][href*="son-maclari"]')
                    )
                )
                self.driver.execute_script("arguments[0].click();", last_matches_button)
                print("✅ Son Maçlar sekmesine tıklandı")
                time.sleep(3)
            except Exception as e:
                print(f"⚠️ Son Maçlar sekmesi bulunamadı: {e}")
                return False

            try:
                last_matches_sections = self.driver.find_elements(By.CSS_SELECTOR, '[data-test-id^="LastMatchesTable"]')
                if not last_matches_sections:
                    print("⚠️ Son maçlar tablosu bulunamadı!")
                    return False

                for section in last_matches_sections:
                    try:
                        team_name = "-"
                        try:
                            team_link_elem = section.find_element(By.CSS_SELECTOR, '[data-test-id="TeamLink"] span')
                            team_name = team_link_elem.text.strip()
                        except: pass
                        matches_table = section.find_element(By.CSS_SELECTOR, '[data-test-id="LastMatchesTable"] table tbody')
                        match_rows = matches_table.find_elements(By.CSS_SELECTOR, '[data-test-id="LastMatchesTable"]')
                        for row in match_rows:
                            try:
                                lig = "-"
                                tarih = "-"
                                try:
                                    lig_span = row.find_element(By.CSS_SELECTOR, '[data-test-id="TableBodyLeague"] span.a0cdb4268c7ee710adc6')
                                    lig = lig_span.text.strip()
                                    date_spans = row.find_elements(By.CSS_SELECTOR, '[data-test-id="TableBodyLeague"] span')
                                    if len(date_spans) > 1:
                                        tarih = date_spans[1].text.strip()
                                except: pass
                                ms = "-"
                                try:
                                    ms_elem = row.find_element(By.CSS_SELECTOR, '[data-test-id="TableBodyMatch"] .nsn-btn span')
                                    ms = ms_elem.text.strip()
                                except: pass
                                iy = "-"
                                try:
                                    iy_elem = row.find_element(By.CSS_SELECTOR, '[data-test-id="TableBodyFirstHalf"]')
                                    iy = iy_elem.text.strip()
                                except: pass
                                ev_sahibi = "-"
                                deplasman = "-"
                                try:
                                    ev_sahibi = row.find_element(By.CSS_SELECTOR, '[data-test-id="HomeTeam"] span').text.strip()
                                except: pass
                                try:
                                    deplasman = row.find_element(By.CSS_SELECTOR, '[data-test-id="AwayTeam"] span').text.strip()
                                except: pass

                                last_match_data = {
                                    "Ana_Maç": teams,
                                    "Ana_Lig": league,
                                    "Takım": team_name,
                                    "Son_Lig": lig,
                                    "Tarih": tarih,
                                    "Ev_Sahibi": ev_sahibi,
                                    "Deplasman": deplasman,
                                    "MS": ms,
                                    "İY": iy
                                }
                                self.last_matches_data.append(last_match_data)
                            except Exception as e:
                                print(f"⚠️ Son maç satırı ayrıştırılırken hata: {e}")
                                continue
                    except Exception as e:
                        print(f"⚠️ Takım son maç tablosu ayrıştırılamadı: {e}")
                        continue
                print(f"✅ Son maçlar verisi çekildi: {len(self.last_matches_data)} satır")
                return True
            except Exception as e:
                print(f"⚠️ Son maçlar verisi çekilemedi: {e}")
                return False
        except Exception as e:
            print(f"❌ Son maçlar sayfası hatası: {e}")
            return False

    def extract_statistics_data(self):
        print(f"\n📈 İstatistiksel veri çekme başlıyor...")
        print(f"🎯 Hedef: {len(self.matches_data)} maçın puan tablosu, rekabet geçmişi ve son maçlar verisi")
        main_url = "https://www.nesine.com/iddaa?et=1&le=3&ocg=MS-2%2C5&gt=Pop%C3%BCler"
        self.last_matches_data = []
        for i, match_data in enumerate(self.matches_data):
            try:
                print(f"\n📊 Maç {i+1}/{len(self.matches_data)} istatistikleri çekiliyor...")
                match_link = match_data.get("Link", "-")
                teams = match_data.get("Takımlar", "-")
                league = match_data.get("Lig", "-")
                if match_link == "-" or match_link == "":
                    print(f"⚠️ Maç {i+1} için link bulunamadı, atlanıyor...")
                    continue
                self.scrape_league_table(match_link, teams, league)
                self.scrape_competition_history(match_link, teams, league)
                self.scrape_last_matches(match_link, teams, league)
                if i < len(self.matches_data) - 1:
                    print("🔄 Ana sayfaya dönülüyor...")
                    self.driver.get(main_url)
                    time.sleep(2)
            except Exception as e:
                print(f"❌ Maç {i+1} istatistik hatası: {e}")
                continue
        print(f"\n📊 İstatistik çekme tamamlandı!")
        print(f"📈 Puan tablosu verisi: {len(self.league_table_data)} satır")
        print(f"🏆 Rekabet geçmişi verisi: {len(self.competition_history_data)} satır")
        print(f"📅 Son maçlar verisi: {len(self.last_matches_data)} satır")

    def save_to_csv(self):
        if not self.matches_data:
            print("❌ Kaydedilecek veri yok!")
            return False
        try:
            df = pd.DataFrame(self.matches_data)
            columns_order = [
                "Saat", "Lig", "Takımlar", "Link",
                "MS_1", "MS_X", "MS_2",
                "Alt", "Üst",
                "HND", "HND_1", "HND_X", "HND_2",
                "CS_1X", "CS_12", "CS_X2",
                "KG_Var", "KG_Yok"
            ]
            df = df[columns_order]
            csv_paths = [
                "Maclar.csv",
                os.path.join(os.path.expanduser("~"), "Desktop", "Maclar.csv"),
                os.path.join(os.getcwd(), "Maclar.csv")
            ]
            for csv_path in csv_paths:
                try:
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                        print(f"✅ {len(self.matches_data)} maç verisi '{csv_path}' dosyasına kaydedildi")
                        print(f"📁 Dosya boyutu: {os.path.getsize(csv_path)} bytes")
                        print("\n📋 İlk 3 maç (özet):")
                        display_columns = ['Saat', 'Takımlar', 'Link', 'MS_1', 'MS_X', 'MS_2', 'Alt', 'Üst']
                        print(df.head(3)[display_columns].to_string(index=False))
                        print(f"\n📊 Toplam sütun sayısı: {len(df.columns)}")
                        print(f"🏷️ Sütunlar: {', '.join(df.columns.tolist())}")
                        return True
                except Exception as e:
                    continue
            print("❌ Dosya kaydedilemedi!")
            return False
        except Exception as e:
            print(f"❌ CSV kaydetme hatası: {e}")
            return False

    def save_league_table_to_csv(self):
        if not self.league_table_data:
            print("❌ Kaydedilecek puan tablosu verisi yok!")
            return False
        try:
            df = pd.DataFrame(self.league_table_data)
            csv_paths = [
                "Puan_tablosu.csv",
                os.path.join(os.path.expanduser("~"), "Desktop", "Puan_tablosu.csv"),
                os.path.join(os.getcwd(), "Puan_tablosu.csv")
            ]
            for csv_path in csv_paths:
                try:
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                        print(f"✅ {len(self.league_table_data)} puan tablosu verisi '{csv_path}' dosyasına kaydedildi")
                        print(f"📁 Dosya boyutu: {os.path.getsize(csv_path)} bytes")
                        print("\n📋 İlk 5 takım (özet):")
                        display_columns = ['Maç', 'Lig', 'Takım', 'Oynanan', 'Galibiyet', 'Beraberlik', 'Mağlubiyet', 'Puan']
                        print(df.head(5)[display_columns].to_string(index=False))
                        print(f"\n📊 Toplam sütun sayısı: {len(df.columns)}")
                        print(f"🏷️ Sütunlar: {', '.join(df.columns.tolist())}")
                        return True
                except Exception as e:
                    continue
            print("❌ Puan tablosu dosyası kaydedilemedi!")
            return False
        except Exception as e:
            print(f"❌ Puan tablosu CSV kaydetme hatası: {e}")
            return False

    def save_competition_history_to_csv(self):
        if not self.competition_history_data:
            print("❌ Kaydedilecek rekabet geçmişi verisi yok!")
            return False
        try:
            columns = ["Ana_Maç", "Ana_Lig", "Lig", "Sezon", "Tarih", "MS", "İY", "1", "X", "2", "Alt", "Üst"]
            df = pd.DataFrame(self.competition_history_data)
            if not df.empty:
                df = df[columns]
            csv_paths = [
                "Rekabet_gecmisi.csv",
                os.path.join(os.path.expanduser("~"), "Desktop", "Rekabet_gecmisi.csv"),
                os.path.join(os.getcwd(), "Rekabet_gecmisi.csv")
            ]
            for csv_path in csv_paths:
                try:
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                        print(f"✅ {len(df)} rekabet geçmişi verisi '{csv_path}' dosyasına kaydedildi")
                        print(f"📁 Dosya boyutu: {os.path.getsize(csv_path)} bytes")
                        print("\n📋 İlk 5 geçmiş maç (özet):")
                        display_columns = ["Ana_Maç", "Lig", "Sezon", "Tarih", "MS", "İY", "1", "X", "2", "Alt", "Üst"]
                        print(df.head(5)[display_columns].to_string(index=False))
                        print(f"\n📊 Toplam sütun sayısı: {len(df.columns)}")
                        print(f"🏷️ Sütunlar: {', '.join(df.columns.tolist())}")
                        return True
                except Exception as e:
                    continue
            print("❌ Rekabet geçmişi dosyası kaydedilemedi!")
            return False
        except Exception as e:
            print(f"❌ Rekabet geçmişi CSV kaydetme hatası: {e}")
            return False

    def save_last_matches_to_csv(self):
        if not self.last_matches_data:
            print("❌ Kaydedilecek son maçlar verisi yok!")
            return False
        try:
            seen = set()
            unique_last_matches = []
            for row in self.last_matches_data:
                key = (row['Tarih'], row['Ev_Sahibi'], row['Deplasman'], row['MS'])
                if key not in seen:
                    unique_last_matches.append(row)
                    seen.add(key)
            df = pd.DataFrame(unique_last_matches)
            csv_paths = [
                "Son_maclar.csv",
                os.path.join(os.path.expanduser("~"), "Desktop", "Son_maclar.csv"),
                os.path.join(os.getcwd(), "Son_maclar.csv")
            ]
            for csv_path in csv_paths:
                try:
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                        print(f"✅ {len(df)} son maç verisi '{csv_path}' dosyasına kaydedildi")
                        print(f"📋 İlk 5 maç (özet):")
                        display_columns = ['Ana_Maç', 'Takım', 'Son_Lig', 'Tarih', 'Ev_Sahibi', 'Deplasman', 'MS', 'İY']
                        print(df.head(5)[display_columns].to_string(index=False))
                        return True
                except Exception as e:
                    continue
            print("❌ Son maçlar dosyası kaydedilemedi!")
            return False
        except Exception as e:
            print(f"❌ Son maçlar CSV kaydetme hatası: {e}")
            return False

    def scrape_matches(self):
        try:
            while True:
                try:
                    target_count = int(input("Kaç adet maçın verisini çekmek istiyorsunuz? "))
                    if target_count > 0:
                        break
                    else:
                        print("Lütfen 0'dan büyük bir sayı girin.")
                except ValueError:
                    print("Lütfen geçerli bir sayı girin.")
            extract_stats = input("İstatistiksel verileri de çekmek istiyor musunuz? (e/h): ").lower().strip()
            extract_stats = extract_stats in ['e', 'evet', 'yes', 'y']
            print(f"\n🎯 Hedef: TAM {target_count} maç (TÜM ORANLAR + HREF LİNKLERİ)")
            if extract_stats:
                print(f"📊 + İSTATİSTİKSEL VERİLER VE PUAN TABLOLARI")
                print(f"🏆 + REKABET GEÇMİŞİ VE İDDİAA ORANLARI")
                print(f"📅 + SON MAÇLAR TABLOSU")
            if not self.setup_driver(headless=True):
                print("❌ ChromeDriver başlatılamadı!")
                return
            print("🌐 Nesine.com'a gidiliyor...")
            self.driver.get("https://www.nesine.com/iddaa?et=1&le=3&ocg=MS-2%2C5&gt=Pop%C3%BCler")
            print("⏳ Sayfa yükleniyor...")
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
            time.sleep(3)
            self.close_live_events()
            time.sleep(2)
            self.get_exact_matches(target_count)
            self.extract_match_data(target_count)
            success = self.save_to_csv()
            if extract_stats and success:
                self.extract_statistics_data()
                self.save_league_table_to_csv()
                self.save_competition_history_to_csv()
                self.save_last_matches_to_csv()
            if success:
                print(f"\n🎉 İşlem tamamlandı!")
                print(f"📊 İstenen: {target_count} | Çekilen: {len(self.matches_data)}")
                print(f"📈 Çıkarılan oran türleri: MS, Alt/Üst, HND, Çifte Şans, KG")
                print(f"🔗 Maç detay linkleri de dahil edildi!")
                if extract_stats:
                    print(f"📊 Puan tablosu verileri: {len(self.league_table_data)} satır")
                    print(f"🏆 Rekabet geçmişi verileri: {len(self.competition_history_data)} satır")
                    print(f"📅 Son maçlar verileri: {len(pd.DataFrame(self.last_matches_data).drop_duplicates(subset=['Tarih','Ev_Sahibi','Deplasman','MS']))} satır")
            else:
                print("❌ CSV kaydetme başarısız!")
        except Exception as e:
            print(f"❌ Hata: {e}")
        finally:
            if self.driver:
                print("🔄 Tarayıcı kapatılıyor...")
                self.driver.quit()

if __name__ == "__main__":
    print("=" * 80)
    print("🏈 NESİNE.COM TÜM İDDİAA ORANLARI VERİ ÇEKME")
    print("📊 MS | Alt/Üst | HND | Çifte Şans | Karşılıklı Gol")
    print("🔗 + HREF LİNKLERİ + İSTATİSTİKSEL VERİLER")
    print("📋 + GÜNCELLENMİŞ HTML YAPISI İLE PUAN TABLOSU")
    print("🏆 + REKABET GEÇMİŞİ VE İDDİAA ORANLARI")
    print("📅 + SON MAÇLAR TABLOSU")
    print("=" * 80)
    scraper = NesineScraper()
    scraper.scrape_matches()