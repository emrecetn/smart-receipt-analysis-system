import requests
import json

def fis_verilerini_cikar(ocr_metni):
    # Ollama'nın çalıştığı adres
    url = "http://localhost:11434/api/generate"
    
    # İŞTE O KUSURSUZ MÜHENDİSLİK PROMPTU
    system_prompt = """Sen uzman bir finansal veri çıkarma asistanısın.
Görev: Aşağıda verilen karmaşık OCR metnini analiz et ve bir alışveriş fişindeki temel bilgileri çıkar.

Çıkarılacak Alanlar ve Kesin Kurallar:
1. "merchant_name": Dükkanın TAM adı. KURAL: Fişin en üstündeki ilk 2-3 satırı birleştir. Eğer "SAN.TIC.LTD.STI" gibi şirket ünvanları varsa bunları KESİNLİKLE dahil et. (Örn: "6N MARKET MERDILAY GIDA SAN.TIC.LTD.STI.")
2. "tax_id": Vergi numarası (V.N, V.NO veya V.D yazan yerin yanındaki 10-11 haneli sayı).
3. "date": Tarih (TARIH yazan yer, GG/AA/YYYY formatında).
4. "receipt_no": Fiş numarası (FIS NO yazan yer).
5. "total_amount": Ödenen toplam tutar. KURAL: "TOPLAM" yazısının altındaki veya yanındaki sayıyı al. Sadece rakam olsun (Örn: 428.37).
6. "tax_amount": Fişteki toplam KDV. KURAL: Metinde "TOPKDV", "TOPLAM KDV" veya "KDV" yazısını ara ve altındaki/yanındaki rakamı al.
7. "tax_rate_1": %1 KDV'li ürünlerin kümülatif toplamı. KURAL: Fişin altında açıkça bir "KDV Özeti" tablosu yoksa KESİNLİKLE null ver. Kendi kendine ürünlerin fiyatını TOPLAMA, dil modelleri matematik yapamaz.
8. "tax_rate_10": %10 KDV kümülatif toplamı. (Özeti yoksa null ver).
9. "tax_rate_20": %20 KDV kümülatif toplamı. (Özeti yoksa null ver).

GENEL KURALLAR:
- SADECE geçerli bir JSON objesi döndür.
- JSON dışında hiçbir açıklama, selamlama veya markdown (```json) karakteri KULLANMA.
- Sayılardan TL, *, % gibi işaretleri temizle ve virgülleri noktaya çevir (Örn: *4,40 yerine 4.40 yaz).

OCR METNİ:
"""

    # Prompt ile OCR metnini birleştiriyoruz
    tam_prompt = system_prompt + ocr_metni

    # Ollama'ya gönderilecek paket
    payload = {
        "model": "llama3", # Eğer Ollama'da modelin adı farklıysa (örn: llama3:8b) burayı güncelle
        "prompt": tam_prompt,
        "format": "json",  # Modelin gevezelik etmesini kesin olarak engeller
        "stream": False,
        "options": {
            "temperature": 0.0 # Yaratıcılığı sıfırlıyoruz ki uydurmasın, sadece gördüğünü yazsın
        }
    }

    print("Llama 3 fişi inceliyor, lütfen bekleyin...")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Gelen cevabı al ve JSON'a çevir
        cevap_metni = response.json()["response"]
        json_verisi = json.loads(cevap_metni)
        
        return json_verisi
        
    except requests.exceptions.RequestException as e:
        print(f"Ollama'ya ulaşılamadı! Ollama'nın arka planda açık olduğundan emin ol. Hata: {e}")
        return None
    except json.JSONDecodeError:
        print("Model JSON formatını bozdu! Gelen ham cevap:")
        print(cevap_metni)
        return None

# Test için PaddleOCR'dan az önce aldığımız o çıktı metnini buraya yapıştırıyoruz
ornek_ocr_metni = """
6N MARKET
MERDILAY GIDA SAN.TIC.LTD.STI.
BAHÇELIEVLER MAH. 100. YIL CAD.
NO:110
ALTIEYLUL/BALIKESIR
KARESI V.D V.NO:6170182590
TEL: 0(266)243 54 54
MERSIS NO: 0617018259000010
WWW.GNMARKET.COM
TARIH: 04/02/2026
FIS NO: 00154
SAAT : 12:14:11
6N ALIŞVER IŞ POŞET
%20
*1,00
EKER 1 LT ORMAN MEYV
%1
*97,50
TUNA YUFKA 5'LI
%1
*125,00
TUNALAR YUMURTA 15'L
%1
*73,00
TATAL 800 GR TRABZON
%1
*60,00
0,575 KILOGRAM X 36,90 TL
PIRASA KG
%1
*21,22
1,080 KILOGRAM X 46,90 TL
PORTAKAL KG
%1
*50,65
TOPKDV
*4,40
TOPLAM
*428,37
KREDI KARTI
*428,37
"""

# Fonksiyonu çalıştır ve sonucu ekrana bas
sonuc_json = fis_verilerini_cikar(ornek_ocr_metni)

if sonuc_json:
    print("\n✅ İŞTE MÜKEMMEL JSON ÇIKTISI:\n")
    # JSON'ı okunaklı ve renkli formatta yazdırır
    print(json.dumps(sonuc_json, indent=4, ensure_ascii=False))