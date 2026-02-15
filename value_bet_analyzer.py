"""
🎯 Value Bet Analyzer - Profesyonel Futbol Bahis Analiz Sistemi
Nesine.com verileri üzerinde istatistiksel analiz ve value bet tespiti

Analiz Faktörleri:
- Form Analizi (%20): Son maç performansları ve puan tablosu
- Hakem Faktörü (%15): Hakem istatistikleri ve eğilimleri  
- H2H Rekabet Geçmişi (%15): Takımlar arası geçmiş maçlar
- Oran Değerlendirmesi (%20): Value bet tespiti için oran analizi
- Lig Pozisyonu (%15): Puan tablosu sıralaması ve averaj
- Sakat/Cezalı Analizi (%15): Eksik oyuncuların takım gücüne etkisi
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
import warnings
warnings.filterwarnings('ignore')


# Çalışma dizini
WORK_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class InjuryData:
    """Sakat ve cezalı oyuncu verilerini tutan sınıf"""
    mac_kodu: str
    mac: str
    takim: str
    numara: int
    oyuncu: str
    yas: int
    pozisyon: str
    mac_sayisi: int
    ilk_11: int
    gol: int
    asist: int
    durum: str  # Sakatlık veya Cezalı
    aciklama: str
    
    @property
    def onem_puani(self) -> float:
        """Oyuncunun takımdaki önem puanını hesaplar"""
        puan = 0.0
        
        # İlk 11 bazlı önem (en kritik faktör)
        if self.ilk_11 >= 15:
            puan += 10  # Vazgeçilmez oyuncu
        elif self.ilk_11 >= 10:
            puan += 7   # Çok önemli oyuncu
        elif self.ilk_11 >= 5:
            puan += 4   # Önemli oyuncu
        else:
            puan += 1   # Rotasyon oyuncusu
        
        # Skorer katkısı
        skor_katkisi = self.gol + self.asist
        if skor_katkisi >= 10:
            puan += 5   # Yıldız skorer
        elif skor_katkisi >= 5:
            puan += 3   # İyi skorer
        elif skor_katkisi >= 2:
            puan += 1   # Katkı sağlayan
        
        # Pozisyon bazlı ağırlık
        if self.pozisyon.lower() in ['forvet', 'santrafor']:
            puan *= 1.2  # Forvetler daha kritik
        elif self.pozisyon.lower() in ['orta saha', 'ortasaha']:
            puan *= 1.1  # Orta sahalar önemli
        elif self.pozisyon.lower() in ['kaleci']:
            puan *= 1.5  # Kaleci çok kritik
        
        return puan


@dataclass
class MatchAnalysis:
    """Maç analiz sonuçlarını tutan veri sınıfı"""
    mac_kodu: str
    mac: str
    lig: str
    tarih: str
    saat: str
    
    # Oranlar
    ms_1: float
    ms_x: float
    ms_2: float
    alt_2_5: float
    ust_2_5: float
    
    # Ev Sahibi İstatistikleri
    ev_sahibi: str
    ev_sira: int
    ev_puan: int
    ev_form_puan: float
    ev_son_mac_trend: str
    
    # Deplasman İstatistikleri
    deplasman: str
    dep_sira: int
    dep_puan: int
    dep_form_puan: float
    dep_son_mac_trend: str
    
    # Hakem İstatistikleri
    hakem_adi: str
    hakem_ev_yuzde: float
    hakem_x_yuzde: float
    hakem_dep_yuzde: float
    hakem_ust_yuzde: float
    
    # H2H İstatistikleri
    h2h_mac_sayisi: int
    h2h_ev_galibiyet: int
    h2h_beraberlik: int
    h2h_dep_galibiyet: int
    h2h_son_trend: str
    
    # Analiz Puanları
    form_puani: float
    hakem_puani: float
    h2h_puani: float
    oran_value_puani: float
    lig_pozisyon_puani: float
    
    # Sakat/Cezalı Analizi
    ev_eksik_puan: float  # Ev sahibi eksik oyuncu ceza puanı
    dep_eksik_puan: float  # Deplasman eksik oyuncu ceza puanı
    ev_eksik_sayisi: int
    dep_eksik_sayisi: int
    ev_kritik_eksikler: str  # Kritik eksik oyuncu isimleri
    dep_kritik_eksikler: str
    eksik_puani: float  # Sakat/cezalı faktörü toplam puanı
    
    # Genel Sonuç
    toplam_guven_puani: float
    tahmin: str
    value_bet: str
    risk_seviyesi: str
    aciklama: str


def fuzzy_match(s1: str, s2: str) -> float:
    """İki string arasındaki benzerlik oranını hesaplar"""
    if not s1 or not s2:
        return 0.0
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    
    # Direkt eşleşme
    if s1 == s2:
        return 1.0
    
    # İçerme kontrolü
    if s1 in s2 or s2 in s1:
        return 0.9
    
    # Kelime bazlı eşleşme
    words1 = set(s1.replace(".", " ").replace("-", " ").split())
    words2 = set(s2.replace(".", " ").replace("-", " ").split())
    
    if len(words1) > 0 and len(words2) > 0:
        intersection = words1.intersection(words2)
        if len(intersection) >= 1:
            # Anlamlı kelime eşleşmesi (3+ karakter)
            meaningful = [w for w in intersection if len(w) >= 3]
            if meaningful:
                return 0.85
    
    # SequenceMatcher ile benzerlik
    return SequenceMatcher(None, s1, s2).ratio()


def find_best_match(target: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
    """En iyi eşleşmeyi bulur"""
    if not target or not candidates:
        return None
    
    best_match = None
    best_score = threshold
    
    for candidate in candidates:
        score = fuzzy_match(target, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match


def form_to_points(form_str: str) -> float:
    """Form stringini puana çevirir (G=3, B=1, M=0)"""
    if not form_str or pd.isna(form_str):
        return 0.0
    
    points = 0
    count = 0
    for char in str(form_str).upper():
        if char == 'G':
            points += 3
            count += 1
        elif char == 'B':
            points += 1
            count += 1
        elif char == 'M':
            count += 1
    
    # Normalize (maks 15 puan = 5 galibiyet)
    return (points / 15) * 100 if count > 0 else 0.0


def analyze_form_trend(form_str: str) -> str:
    """Son maçlardaki trendi analiz eder"""
    if not form_str or pd.isna(form_str) or len(str(form_str)) < 2:
        return "Belirsiz"
    
    form = str(form_str).upper()
    
    # Son 3 maç ağırlıklı
    recent = form[:3] if len(form) >= 3 else form
    
    g_count = recent.count('G')
    m_count = recent.count('M')
    
    if g_count >= 2:
        return "⬆️ Yükseliş"
    elif m_count >= 2:
        return "⬇️ Düşüş"
    else:
        return "➡️ Stabil"


def calculate_implied_probability(odd: float) -> float:
    """Orandan ima edilen olasılığı hesaplar"""
    if odd <= 0:
        return 0.0
    return (1 / odd) * 100


def calculate_value_bet_score(predicted_prob: float, market_prob: float) -> float:
    """Value bet skorunu hesaplar"""
    if market_prob <= 0:
        return 0.0
    
    edge = predicted_prob - market_prob
    # Pozitif edge = value var
    return edge


def parse_score(score_str: str) -> Tuple[int, int]:
    """Skor stringini parse eder"""
    if not score_str or pd.isna(score_str):
        return (-1, -1)
    
    try:
        parts = str(score_str).replace(" ", "").split("-")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except:
        pass
    return (-1, -1)


def load_data() -> Dict[str, pd.DataFrame]:
    """Tüm CSV dosyalarını yükler"""
    data = {}
    
    files = {
        'bulten': 'Bülten.csv',
        'puan_tablosu': 'Puan_Tablosu.csv',
        'son_maclar': 'Son_Maclar.csv',
        'rekabet_gecmisi': 'Rekabet_Gecmisi.csv',
        'hakem_bilgileri': 'Hakem_Bilgileri.csv',
        'hakem_istatistikleri': 'Hakem_Istatistikleri.csv',
        'sakat_cezali': 'Sakat_Cezali.csv'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(WORK_DIR, filename)
        if os.path.exists(filepath):
            # Sakat_Cezali.csv virgül ayracı kullanıyor, diğerleri noktalı virgül
            separator = ',' if key == 'sakat_cezali' else ';'
            data[key] = pd.read_csv(filepath, sep=separator, encoding='utf-8-sig')
            print(f"✓ {filename} yüklendi ({len(data[key])} satır)")
        else:
            print(f"✗ {filename} bulunamadı!")
            data[key] = pd.DataFrame()
    
    return data


def load_injury_data(df: pd.DataFrame) -> Dict[str, List[InjuryData]]:
    """Sakat/Cezalı verilerini maç koduna göre gruplar"""
    injury_dict = {}
    
    if df.empty:
        return injury_dict
    
    for _, row in df.iterrows():
        try:
            mac_kodu = str(row.get('Maç_Kodu', '')).strip()
            if not mac_kodu:
                continue
            
            injury = InjuryData(
                mac_kodu=mac_kodu,
                mac=str(row.get('Maç', '')),
                takim=str(row.get('Takım', '')),
                numara=int(row.get('Numara', 0)) if pd.notna(row.get('Numara')) else 0,
                oyuncu=str(row.get('Oyuncu', '')),
                yas=int(row.get('Yaş', 0)) if pd.notna(row.get('Yaş')) else 0,
                pozisyon=str(row.get('Pozisyon', '')),
                mac_sayisi=int(row.get('Maç_Sayısı', 0)) if pd.notna(row.get('Maç_Sayısı')) else 0,
                ilk_11=int(row.get('İlk_11', 0)) if pd.notna(row.get('İlk_11')) else 0,
                gol=int(row.get('Gol', 0)) if pd.notna(row.get('Gol')) else 0,
                asist=int(row.get('Asist', 0)) if pd.notna(row.get('Asist')) else 0,
                durum=str(row.get('Durum', '')),
                aciklama=str(row.get('Açıklama', ''))
            )
            
            if mac_kodu not in injury_dict:
                injury_dict[mac_kodu] = []
            injury_dict[mac_kodu].append(injury)
            
        except Exception as e:
            continue
    
    return injury_dict


def calculate_missing_player_penalty(injuries: List[InjuryData], ev_sahibi: str, deplasman: str) -> Dict:
    """
    Sakat/cezalı oyuncuların takıma etkisini hesaplar.
    
    Returns:
        Dict: {
            'ev_ceza': float,  # Ev sahibi ceza puanı (negatif etki)
            'dep_ceza': float,  # Deplasman ceza puanı (negatif etki)
            'ev_sayisi': int,  # Ev sahibi eksik sayısı
            'dep_sayisi': int,  # Deplasman eksik sayısı
            'ev_kritik': str,  # Ev sahibi kritik eksikler
            'dep_kritik': str,  # Deplasman kritik eksikler
            'score': float  # Normalize edilmiş skor (0-100)
        }
    """
    result = {
        'ev_ceza': 0.0,
        'dep_ceza': 0.0,
        'ev_sayisi': 0,
        'dep_sayisi': 0,
        'ev_kritik': '',
        'dep_kritik': '',
        'score': 50.0  # Nötr başlangıç
    }
    
    if not injuries:
        return result
    
    ev_kritik_list = []
    dep_kritik_list = []
    
    for injury in injuries:
        onem = injury.onem_puani
        
        # Takım eşleştirmesi (fuzzy match kullan)
        ev_match = fuzzy_match(injury.takim, ev_sahibi)
        dep_match = fuzzy_match(injury.takim, deplasman)
        
        if ev_match > dep_match and ev_match > 0.6:
            result['ev_ceza'] += onem
            result['ev_sayisi'] += 1
            if onem >= 7:  # Kritik oyuncu
                ev_kritik_list.append(injury.oyuncu)
        elif dep_match > ev_match and dep_match > 0.6:
            result['dep_ceza'] += onem
            result['dep_sayisi'] += 1
            if onem >= 7:  # Kritik oyuncu
                dep_kritik_list.append(injury.oyuncu)
    
    # Kritik eksikler listesi
    result['ev_kritik'] = ', '.join(ev_kritik_list[:3]) if ev_kritik_list else ''
    result['dep_kritik'] = ', '.join(dep_kritik_list[:3]) if dep_kritik_list else ''
    
    # Score hesaplama: Eksik farkına göre puan
    # Ev sahibi eksikse puan düşer, deplasman eksikse puan artar
    ceza_farki = result['dep_ceza'] - result['ev_ceza']
    
    # -50 ile +50 arası farkı 0-100'e normalize et
    result['score'] = min(max(50 + (ceza_farki * 2), 0), 100)
    
    return result


def get_team_standing(puan_df: pd.DataFrame, mac_kodu: str, takim_tipi: str) -> Dict:
    """Belirli bir takımın puan tablosu verilerini döndürür"""
    result = {
        'sira': 0, 'puan': 0, 'form': '', 'form_puan': 0.0,
        'o': 0, 'g': 0, 'b': 0, 'm': 0, 'av': 0, 'takim': ''
    }
    
    if puan_df.empty:
        return result
    
    mask = (puan_df['Maç_Kodu'].astype(str) == str(mac_kodu)) & \
           (puan_df['Takım_Tipi'] == takim_tipi)
    
    row = puan_df[mask]
    
    if not row.empty:
        row = row.iloc[0]
        result['sira'] = int(row['Sıra']) if pd.notna(row.get('Sıra')) else 0
        result['puan'] = int(row['P']) if pd.notna(row.get('P')) else 0
        result['form'] = str(row['Form']) if pd.notna(row.get('Form')) else ''
        result['form_puan'] = form_to_points(row.get('Form', ''))
        result['o'] = int(row['O']) if pd.notna(row.get('O')) else 0
        result['g'] = int(row['G']) if pd.notna(row.get('G')) else 0
        result['b'] = int(row['B']) if pd.notna(row.get('B')) else 0
        result['m'] = int(row['M']) if pd.notna(row.get('M')) else 0
        result['av'] = int(row['AV']) if pd.notna(row.get('AV')) else 0
        result['takim'] = str(row['Takım']) if pd.notna(row.get('Takım')) else ''
    
    return result


def get_referee_stats(hakem_stat_df: pd.DataFrame, mac_kodu: str) -> Dict:
    """Hakem istatistiklerini döndürür"""
    result = {
        'hakem_adi': '', 'ms1_yuzde': 0.0, 'msx_yuzde': 0.0, 'ms2_yuzde': 0.0,
        'alt_yuzde': 0.0, 'ust_yuzde': 0.0, 'kg_var_yuzde': 0.0
    }
    
    if hakem_stat_df.empty:
        return result
    
    row = hakem_stat_df[hakem_stat_df['Maç_Kodu'].astype(str) == str(mac_kodu)]
    
    if not row.empty:
        row = row.iloc[0]
        result['hakem_adi'] = str(row['Hakem_Adı']) if pd.notna(row.get('Hakem_Adı')) else ''
        
        # Yüzde değerlerini parse et
        for col, key in [('MS1_Yüzde', 'ms1_yuzde'), ('MSX_Yüzde', 'msx_yuzde'), 
                         ('MS2_Yüzde', 'ms2_yuzde'), ('Alt_2_5_Yüzde', 'alt_yuzde'),
                         ('Üst_2_5_Yüzde', 'ust_yuzde'), ('KG_Var_Yüzde', 'kg_var_yuzde')]:
            val = row.get(col)
            if pd.notna(val):
                try:
                    result[key] = float(str(val).replace('%', ''))
                except:
                    pass
    
    return result


def get_h2h_stats(rekabet_df: pd.DataFrame, mac_kodu: str) -> Dict:
    """Rekabet geçmişi istatistiklerini döndürür"""
    result = {
        'mac_sayisi': 0, 'ev_galibiyet': 0, 'beraberlik': 0, 
        'dep_galibiyet': 0, 'son_trend': '', 'ust_orani': 0.0
    }
    
    if rekabet_df.empty:
        return result
    
    matches = rekabet_df[rekabet_df['Maç_Kodu'].astype(str) == str(mac_kodu)]
    
    if matches.empty:
        return result
    
    result['mac_sayisi'] = len(matches)
    
    ust_count = 0
    for _, row in matches.iterrows():
        ms = str(row.get('MS', '')).replace(" ", "")
        if ms and '-' in ms:
            try:
                parts = ms.split('-')
                home_goals = int(parts[0])
                away_goals = int(parts[1])
                total = home_goals + away_goals
                
                if total > 2.5:
                    ust_count += 1
                
                if home_goals > away_goals:
                    result['ev_galibiyet'] += 1
                elif home_goals < away_goals:
                    result['dep_galibiyet'] += 1
                else:
                    result['beraberlik'] += 1
            except:
                pass
    
    if result['mac_sayisi'] > 0:
        result['ust_orani'] = (ust_count / result['mac_sayisi']) * 100
        
        # Son trend
        if result['ev_galibiyet'] > result['dep_galibiyet']:
            result['son_trend'] = "🏠 Ev Sahibi Üstün"
        elif result['dep_galibiyet'] > result['ev_galibiyet']:
            result['son_trend'] = "✈️ Deplasman Üstün"
        else:
            result['son_trend'] = "⚖️ Dengeli"
    
    return result


def calculate_form_score(ev_form: float, dep_form: float) -> Tuple[float, str]:
    """Form puanını hesaplar (%25 ağırlık)"""
    # Ev sahibi formu ağırlıklı (ev avantajı)
    ev_weighted = ev_form * 1.1  # %10 ev avantajı
    
    diff = ev_weighted - dep_form
    
    # -100 ile +100 arasında normalize
    score = min(max((diff + 100) / 2, 0), 100)
    
    if diff > 20:
        prediction = "1"
    elif diff < -20:
        prediction = "2"
    else:
        prediction = "X"
    
    return score, prediction


def calculate_referee_score(hakem_stats: Dict, tahmin: str) -> float:
    """Hakem faktörü puanını hesaplar (%20 ağırlık)"""
    if not hakem_stats['hakem_adi']:
        return 50.0  # Nötr puan
    
    if tahmin == "1":
        return hakem_stats['ms1_yuzde']
    elif tahmin == "X":
        return hakem_stats['msx_yuzde']
    else:
        return hakem_stats['ms2_yuzde']


def calculate_h2h_score(h2h_stats: Dict, tahmin: str) -> float:
    """H2H puanını hesaplar (%20 ağırlık)"""
    if h2h_stats['mac_sayisi'] == 0:
        return 50.0  # Nötr puan
    
    total = h2h_stats['mac_sayisi']
    
    if tahmin == "1":
        return (h2h_stats['ev_galibiyet'] / total) * 100
    elif tahmin == "X":
        return (h2h_stats['beraberlik'] / total) * 100
    else:
        return (h2h_stats['dep_galibiyet'] / total) * 100


def calculate_value_score(predicted_prob: float, ms_1: float, ms_x: float, ms_2: float, tahmin: str) -> Tuple[float, str]:
    """Value bet skorunu hesaplar (%20 ağırlık)"""
    if tahmin == "1" and ms_1 > 0:
        market_prob = calculate_implied_probability(ms_1)
        edge = predicted_prob - market_prob
        value_type = f"MS1 ({ms_1})"
    elif tahmin == "X" and ms_x > 0:
        market_prob = calculate_implied_probability(ms_x)
        edge = predicted_prob - market_prob
        value_type = f"MSX ({ms_x})"
    else:
        if ms_2 > 0:
            market_prob = calculate_implied_probability(ms_2)
            edge = predicted_prob - market_prob
            value_type = f"MS2 ({ms_2})"
        else:
            return 50.0, ""
    
    # Edge'i 0-100 skalasına çevir
    score = min(max((edge + 20) * 2.5, 0), 100)
    
    if edge > 5:
        return score, f"✅ VALUE BET: {value_type} (Edge: +{edge:.1f}%)"
    else:
        return score, ""


def calculate_league_position_score(ev_sira: int, ev_puan: int, ev_av: int,
                                    dep_sira: int, dep_puan: int, dep_av: int) -> float:
    """Lig pozisyonu puanını hesaplar (%15 ağırlık)"""
    if ev_sira == 0 and dep_sira == 0:
        return 50.0  # Nötr
    
    # Sıra farkı (düşük sıra = daha iyi)
    sira_diff = dep_sira - ev_sira  # Pozitif = ev sahibi daha iyi
    
    # Puan farkı
    puan_diff = ev_puan - dep_puan
    
    # Averaj farkı
    av_diff = ev_av - dep_av
    
    # Normalize ve ağırlıklı toplam
    score = 50 + (sira_diff * 2) + (puan_diff * 0.5) + (av_diff * 0.5)
    
    return min(max(score, 0), 100)


def determine_risk_level(guven_puani: float, h2h_count: int, hakem_adi: str) -> str:
    """Risk seviyesini belirler"""
    risk_factors = 0
    
    if guven_puani < 55:
        risk_factors += 2
    elif guven_puani < 65:
        risk_factors += 1
    
    if h2h_count < 3:
        risk_factors += 1
    
    if not hakem_adi:
        risk_factors += 1
    
    if risk_factors >= 3:
        return "🔴 Yüksek Risk"
    elif risk_factors >= 2:
        return "🟡 Orta Risk"
    else:
        return "🟢 Düşük Risk"


def generate_explanation(analysis: MatchAnalysis) -> str:
    """Analiz açıklaması oluşturur"""
    parts = []
    
    # Form analizi
    if analysis.ev_form_puan > analysis.dep_form_puan + 20:
        parts.append(f"📊 {analysis.ev_sahibi} formda ({analysis.ev_son_mac_trend})")
    elif analysis.dep_form_puan > analysis.ev_form_puan + 20:
        parts.append(f"📊 {analysis.deplasman} formda ({analysis.dep_son_mac_trend})")
    
    # Sakat/Cezalı analizi
    if analysis.ev_eksik_sayisi > 0 and analysis.ev_kritik_eksikler:
        parts.append(f"🏥 {analysis.ev_sahibi}: {analysis.ev_eksik_sayisi} eksik ({analysis.ev_kritik_eksikler})")
    if analysis.dep_eksik_sayisi > 0 and analysis.dep_kritik_eksikler:
        parts.append(f"🏥 {analysis.deplasman}: {analysis.dep_eksik_sayisi} eksik ({analysis.dep_kritik_eksikler})")
    
    # Hakem etkisi
    if analysis.hakem_adi:
        if analysis.hakem_ev_yuzde >= 50:
            parts.append(f"👨‍⚖️ Hakem ev sahibine yatkın (%{analysis.hakem_ev_yuzde:.0f})")
        elif analysis.hakem_dep_yuzde >= 40:
            parts.append(f"👨‍⚖️ Hakem deplasmanı destekliyor (%{analysis.hakem_dep_yuzde:.0f})")
    
    # H2H
    if analysis.h2h_mac_sayisi >= 3:
        parts.append(f"📜 H2H: {analysis.h2h_son_trend}")
    
    # Lig pozisyonu
    if analysis.ev_sira > 0 and analysis.dep_sira > 0:
        if analysis.ev_sira < analysis.dep_sira - 3:
            parts.append(f"📈 {analysis.ev_sahibi} ligde üstün ({analysis.ev_sira}. vs {analysis.dep_sira}.)")
        elif analysis.dep_sira < analysis.ev_sira - 3:
            parts.append(f"📈 {analysis.deplasman} ligde üstün ({analysis.dep_sira}. vs {analysis.ev_sira}.)")
    
    return " | ".join(parts) if parts else "Detaylı analiz için veri yetersiz"


def analyze_match(row: pd.Series, data: Dict[str, pd.DataFrame], injury_dict: Dict[str, List[InjuryData]]) -> Optional[MatchAnalysis]:
    """Tek bir maçı analiz eder"""
    try:
        mac_kodu = str(row['Maç_Kodu'])
        mac = str(row.get('Maç', ''))
        
        if not mac or '-' not in mac:
            return None
        
        teams = mac.split(' - ')
        if len(teams) != 2:
            return None
        
        ev_sahibi = teams[0].strip()
        deplasman = teams[1].strip()
        
        # Oranları parse et
        def safe_float(val):
            try:
                return float(str(val).replace(',', '.'))
            except:
                return 0.0
        
        ms_1 = safe_float(row.get('MS_1'))
        ms_x = safe_float(row.get('MS_X'))
        ms_2 = safe_float(row.get('MS_2'))
        alt_2_5 = safe_float(row.get('Alt_2_5'))
        ust_2_5 = safe_float(row.get('Üst_2_5'))
        
        # Puan tablosu verilerini al
        ev_standing = get_team_standing(data['puan_tablosu'], mac_kodu, 'Ev Sahibi')
        dep_standing = get_team_standing(data['puan_tablosu'], mac_kodu, 'Deplasman')
        
        # Hakem istatistiklerini al
        hakem_stats = get_referee_stats(data['hakem_istatistikleri'], mac_kodu)
        
        # H2H istatistiklerini al
        h2h_stats = get_h2h_stats(data['rekabet_gecmisi'], mac_kodu)
        
        # Sakat/Cezalı analizi
        injuries = injury_dict.get(mac_kodu, [])
        injury_penalty = calculate_missing_player_penalty(injuries, ev_sahibi, deplasman)
        
        # Form puanları
        ev_form_puan = ev_standing['form_puan']
        dep_form_puan = dep_standing['form_puan']
        
        # Eksik oyuncu etkisini form puanına uygula
        # Ev sahibi eksikse form puanı düşer, deplasman eksikse artar
        ev_form_adjusted = max(0, ev_form_puan - injury_penalty['ev_ceza'] * 2)
        dep_form_adjusted = max(0, dep_form_puan - injury_penalty['dep_ceza'] * 2)
        
        # Form bazlı tahmin (düzeltilmiş formlarla)
        form_score, form_tahmin = calculate_form_score(ev_form_adjusted, dep_form_adjusted)
        
        # Lig pozisyonu puanı
        lig_score = calculate_league_position_score(
            ev_standing['sira'], ev_standing['puan'], ev_standing['av'],
            dep_standing['sira'], dep_standing['puan'], dep_standing['av']
        )
        
        # Lig pozisyonu bazlı tahmin ayarlaması
        if lig_score > 65 and form_tahmin != "1":
            final_tahmin = "1"
        elif lig_score < 35 and form_tahmin != "2":
            final_tahmin = "2"
        else:
            final_tahmin = form_tahmin
        
        # Eksik oyuncu etkisiyle tahmin revizyonu
        # Bir takımın çok fazla eksiği varsa tahmin değişebilir
        if injury_penalty['ev_ceza'] > 20 and final_tahmin == "1":
            final_tahmin = "X"  # Ev sahibi çok zayıfladı
        elif injury_penalty['dep_ceza'] > 20 and final_tahmin == "2":
            final_tahmin = "X"  # Deplasman çok zayıfladı
        
        # Hakem puanı
        hakem_score = calculate_referee_score(hakem_stats, final_tahmin)
        
        # H2H puanı
        h2h_score = calculate_h2h_score(h2h_stats, final_tahmin)
        
        # Eksik oyuncu puanı
        eksik_score = injury_penalty['score']
        
        # Tahmini olasılık hesapla (yeni ağırlıklar: Form %20, Hakem %15, H2H %15, Value %20, Lig %15, Eksik %15)
        predicted_prob = (form_score * 0.20 + hakem_score * 0.15 + 
                         h2h_score * 0.15 + lig_score * 0.15 + 
                         eksik_score * 0.15 + 50 * 0.20)
        
        # Value bet puanı
        value_score, value_bet = calculate_value_score(
            predicted_prob, ms_1, ms_x, ms_2, final_tahmin
        )
        
        # Toplam güven puanı (yeni ağırlıklar)
        toplam_puan = (
            form_score * 0.20 +
            hakem_score * 0.15 +
            h2h_score * 0.15 +
            value_score * 0.20 +
            lig_score * 0.15 +
            eksik_score * 0.15
        )
        
        # Risk seviyesi
        risk = determine_risk_level(toplam_puan, h2h_stats['mac_sayisi'], hakem_stats['hakem_adi'])
        
        # Analiz objesi oluştur
        analysis = MatchAnalysis(
            mac_kodu=mac_kodu,
            mac=mac,
            lig=str(row.get('Lig', '')),
            tarih=str(row.get('Tarih', '')),
            saat=str(row.get('Saat', '')),
            ms_1=ms_1,
            ms_x=ms_x,
            ms_2=ms_2,
            alt_2_5=alt_2_5,
            ust_2_5=ust_2_5,
            ev_sahibi=ev_sahibi,
            ev_sira=ev_standing['sira'],
            ev_puan=ev_standing['puan'],
            ev_form_puan=ev_form_puan,
            ev_son_mac_trend=analyze_form_trend(ev_standing['form']),
            deplasman=deplasman,
            dep_sira=dep_standing['sira'],
            dep_puan=dep_standing['puan'],
            dep_form_puan=dep_form_puan,
            dep_son_mac_trend=analyze_form_trend(dep_standing['form']),
            hakem_adi=hakem_stats['hakem_adi'],
            hakem_ev_yuzde=hakem_stats['ms1_yuzde'],
            hakem_x_yuzde=hakem_stats['msx_yuzde'],
            hakem_dep_yuzde=hakem_stats['ms2_yuzde'],
            hakem_ust_yuzde=hakem_stats['ust_yuzde'],
            h2h_mac_sayisi=h2h_stats['mac_sayisi'],
            h2h_ev_galibiyet=h2h_stats['ev_galibiyet'],
            h2h_beraberlik=h2h_stats['beraberlik'],
            h2h_dep_galibiyet=h2h_stats['dep_galibiyet'],
            h2h_son_trend=h2h_stats['son_trend'],
            form_puani=form_score,
            hakem_puani=hakem_score,
            h2h_puani=h2h_score,
            oran_value_puani=value_score,
            lig_pozisyon_puani=lig_score,
            ev_eksik_puan=injury_penalty['ev_ceza'],
            dep_eksik_puan=injury_penalty['dep_ceza'],
            ev_eksik_sayisi=injury_penalty['ev_sayisi'],
            dep_eksik_sayisi=injury_penalty['dep_sayisi'],
            ev_kritik_eksikler=injury_penalty['ev_kritik'],
            dep_kritik_eksikler=injury_penalty['dep_kritik'],
            eksik_puani=eksik_score,
            toplam_guven_puani=toplam_puan,
            tahmin=final_tahmin,
            value_bet=value_bet,
            risk_seviyesi=risk,
            aciklama=""
        )
        
        analysis.aciklama = generate_explanation(analysis)
        
        return analysis
        
    except Exception as e:
        print(f"Analiz hatası ({row.get('Maç', 'Bilinmeyen')}): {e}")
        return None


def run_analysis() -> List[MatchAnalysis]:
    """Ana analiz fonksiyonu"""
    print("="*80)
    print("🎯 VALUE BET ANALYZER - Profesyonel Futbol Bahis Analiz Sistemi")
    print("="*80)
    print()
    
    # Verileri yükle
    print("📂 Veriler yükleniyor...")
    data = load_data()
    print()
    
    # Sakat/Cezalı verilerini yükle
    injury_dict = load_injury_data(data.get('sakat_cezali', pd.DataFrame()))
    if injury_dict:
        print(f"🏥 {len(injury_dict)} maç için sakat/cezalı verisi yüklendi")
    
    if data['bulten'].empty:
        print("❌ Bülten verisi bulunamadı!")
        return []
    
    # Tüm maçları analiz et
    print("🔍 Maçlar analiz ediliyor...")
    print("-"*80)
    
    analyses = []
    for _, row in data['bulten'].iterrows():
        analysis = analyze_match(row, data, injury_dict)
        if analysis:
            analyses.append(analysis)
    
    print(f"\n✓ {len(analyses)} maç analiz edildi")
    
    # Güven puanına göre sırala
    analyses.sort(key=lambda x: x.toplam_guven_puani, reverse=True)
    
    return analyses


def print_report(analyses: List[MatchAnalysis]):
    """Analiz raporunu yazdırır"""
    print()
    print("="*80)
    print("📊 VALUE BET ANALİZ RAPORU")
    print("="*80)
    
    # Value bet olan maçlar
    value_bets = [a for a in analyses if a.value_bet and a.toplam_guven_puani >= 60]
    
    if value_bets:
        print()
        print("🎯 EN İYİ VALUE BET ÖNERİLERİ")
        print("-"*80)
        
        for i, a in enumerate(value_bets[:10], 1):
            print(f"\n{i}. {a.mac}")
            print(f"   📅 {a.tarih} {a.saat} | 🏆 {a.lig}")
            print(f"   💰 Oranlar: 1={a.ms_1:.2f} | X={a.ms_x:.2f} | 2={a.ms_2:.2f}")
            print(f"   📈 Güven Puanı: {a.toplam_guven_puani:.1f}/100 | Tahmin: MS{a.tahmin}")
            print(f"   {a.value_bet}")
            print(f"   {a.risk_seviyesi}")
            print(f"   💡 {a.aciklama}")
            
            # Detay tablosu
            print(f"\n   ┌{'─'*36}┬{'─'*36}┐")
            print(f"   │ {'EV SAHİBİ':^34} │ {'DEPLASMAN':^34} │")
            print(f"   ├{'─'*36}┼{'─'*36}┤")
            print(f"   │ {a.ev_sahibi:<34} │ {a.deplasman:<34} │")
            print(f"   │ Sıra: {a.ev_sira:<28} │ Sıra: {a.dep_sira:<28} │")
            print(f"   │ Puan: {a.ev_puan:<28} │ Puan: {a.dep_puan:<28} │")
            print(f"   │ Form: {a.ev_form_puan:.1f}/100 {a.ev_son_mac_trend:<17} │ Form: {a.dep_form_puan:.1f}/100 {a.dep_son_mac_trend:<17} │")
            print(f"   └{'─'*36}┴{'─'*36}┘")
            
            if a.hakem_adi:
                print(f"\n   👨‍⚖️ Hakem: {a.hakem_adi}")
                print(f"      MS1: %{a.hakem_ev_yuzde:.0f} | MSX: %{a.hakem_x_yuzde:.0f} | MS2: %{a.hakem_dep_yuzde:.0f} | Üst: %{a.hakem_ust_yuzde:.0f}")
            
            if a.h2h_mac_sayisi > 0:
                print(f"\n   📜 H2H ({a.h2h_mac_sayisi} maç): {a.h2h_ev_galibiyet}G-{a.h2h_beraberlik}B-{a.h2h_dep_galibiyet}M")
                print(f"      {a.h2h_son_trend}")
            
            print()
    else:
        print("\n⚠️ Güçlü value bet bulunamadı. Mevcut verilerle yüksek güvenli tahmin yapılamıyor.")
    
    # Tüm analizlerin özeti
    print()
    print("="*80)
    print("📋 TÜM MAÇLAR ÖZETİ")
    print("="*80)
    print()
    print(f"{'Maç':<45} {'Tahmin':<8} {'Güven':<10} {'Risk':<15}")
    print("-"*80)
    
    for a in analyses[:30]:  # İlk 30 maç
        tahmin_str = f"MS{a.tahmin}"
        if a.value_bet:
            tahmin_str += " 💰"
        print(f"{a.mac[:44]:<45} {tahmin_str:<8} {a.toplam_guven_puani:.1f}/100   {a.risk_seviyesi}")
    
    # Alt/Üst analizi
    print()
    print("="*80)
    print("⚽ ALT/ÜST ANALİZİ (Hakem & H2H bazlı)")
    print("="*80)
    print()
    
    ust_candidates = []
    for a in analyses:
        if a.hakem_ust_yuzde >= 50 and a.ust_2_5 > 1.5:
            ust_candidates.append((a, a.hakem_ust_yuzde))
    
    ust_candidates.sort(key=lambda x: x[1], reverse=True)
    
    if ust_candidates:
        print("🔼 ÜST 2.5 Gol Önerileri (Hakem istatistiklerine göre):")
        print("-"*60)
        for a, hakm_ust in ust_candidates[:5]:
            print(f"  • {a.mac}")
            print(f"    Oran: {a.ust_2_5:.2f} | Hakem Üst: %{hakm_ust:.0f}")
            print()
    
    # İstatistiksel özet
    print()
    print("="*80)
    print("📊 İSTATİSTİKSEL ÖZET")
    print("="*80)
    
    if analyses:
        avg_guven = sum(a.toplam_guven_puani for a in analyses) / len(analyses)
        high_conf = len([a for a in analyses if a.toplam_guven_puani >= 65])
        med_conf = len([a for a in analyses if 55 <= a.toplam_guven_puani < 65])
        low_conf = len([a for a in analyses if a.toplam_guven_puani < 55])
        
        print(f"""
  Toplam Analiz Edilen Maç: {len(analyses)}
  Ortalama Güven Puanı: {avg_guven:.1f}/100
  
  Güven Dağılımı:
    🟢 Yüksek Güven (65+): {high_conf} maç
    🟡 Orta Güven (55-65): {med_conf} maç
    🔴 Düşük Güven (<55): {low_conf} maç
  
  Value Bet Sayısı: {len(value_bets)}
        """)


def save_report(analyses: List[MatchAnalysis]):
    """Analiz sonuçlarını CSV olarak kaydeder (MSX tahminleri hariç)"""
    if not analyses:
        return
    
    report_data = []
    for a in analyses:
        # MSX tahminlerini filtrele
        if a.tahmin == 'X':
            continue
            
        report_data.append({
            'Maç_Kodu': a.mac_kodu,
            'Maç': a.mac,
            'Lig': a.lig,
            'Tarih': a.tarih,
            'Saat': a.saat,
            'MS_1': a.ms_1,
            'MS_X': a.ms_x,
            'MS_2': a.ms_2,
            'Tahmin': f"MS{a.tahmin}",
            'Güven_Puanı': round(a.toplam_guven_puani, 1),
            'Form_Puanı': round(a.form_puani, 1),
            'Hakem_Puanı': round(a.hakem_puani, 1),
            'H2H_Puanı': round(a.h2h_puani, 1),
            'Value_Puanı': round(a.oran_value_puani, 1),
            'Lig_Puanı': round(a.lig_pozisyon_puani, 1),
            'Eksik_Puanı': round(a.eksik_puani, 1),
            'Value_Bet': a.value_bet,
            'Risk': a.risk_seviyesi,
            'Ev_Sahibi': a.ev_sahibi,
            'Ev_Sıra': a.ev_sira,
            'Ev_Form': a.ev_form_puan,
            'Ev_Eksik_Sayısı': a.ev_eksik_sayisi,
            'Ev_Kritik_Eksikler': a.ev_kritik_eksikler,
            'Deplasman': a.deplasman,
            'Dep_Sıra': a.dep_sira,
            'Dep_Form': a.dep_form_puan,
            'Dep_Eksik_Sayısı': a.dep_eksik_sayisi,
            'Dep_Kritik_Eksikler': a.dep_kritik_eksikler,
            'Hakem': a.hakem_adi,
            'H2H_Maç_Sayısı': a.h2h_mac_sayisi,
            'Açıklama': a.aciklama
        })
    
    df = pd.DataFrame(report_data)
    filepath = os.path.join(WORK_DIR, 'Value_Bet_Raporu.csv')
    df.to_csv(filepath, index=False, sep=';', encoding='utf-8-sig')
    print(f"\n✓ Rapor kaydedildi: Value_Bet_Raporu.csv (MSX tahminleri filtrelendi)")


def main():
    """Ana fonksiyon"""
    try:
        analyses = run_analysis()
        
        if analyses:
            print_report(analyses)
            save_report(analyses)
        else:
            print("❌ Analiz edilecek maç bulunamadı!")
        
        print("\n" + "="*80)
        print("🏁 Analiz tamamlandı!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Kritik hata: {e}")
        raise


if __name__ == "__main__":
    main()
