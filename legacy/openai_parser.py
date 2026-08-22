from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

# OpenAI İstemcisini başlat (key .env dosyasından okunur)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def fis_verilerini_cikar_openai(ocr_metni):
    # Llama 3'te kullandığımız o mükemmel ve katı Prompt'un aynısı
    system_prompt = """Sen uzman bir finansal veri çıkarma asistanısın.
Görev: Aşağıda verilen karmaşık OCR metnini analiz et ve bir alışveriş fişindeki temel bilgileri çıkar.

Çıkarılacak Alanlar ve Kesin Kurallar:
1. "merchant_name": Dükkanın TAM adı. KURAL: Fişin en üstündeki ilk 2-3 satırı birleştir. Eğer "SAN.TIC.LTD.STI" gibi şirket ünvanları varsa bunları KESİNLİKLE dahil et.
2. "tax_id": Vergi numarası (V.N, V.NO veya V.D yazan yerin yanındaki 10-11 haneli sayı).
3. "date": Tarih (TARIH yazan yer, GG/AA/YYYY formatında).
4. "receipt_no": Fiş numarası (FIS NO yazan yer).
5. "total_amount": Ödenen toplam tutar. KURAL: "TOPLAM" yazısının altındaki veya yanındaki sayıyı al. Sadece rakam olsun (Örn: 428.37).
6. "tax_amount": Fişteki toplam KDV. KURAL: Metinde "TOPKDV", "TOPLAM KDV" veya "KDV" yazısını ara ve altındaki/yanındaki rakamı al.
7. "tax_rate_1": %1 KDV'li ürünlerin kümülatif toplamı. KURAL: Fişin altında açıkça bir "KDV Özeti" tablosu yoksa KESİNLİKLE null ver.
8. "tax_rate_10": %10 KDV kümülatif toplamı. (Özeti yoksa null ver).
9. "tax_rate_20": %20 KDV kümülatif toplamı. (Özeti yoksa null ver).

GENEL KURALLAR:
- SADECE geçerli bir JSON objesi döndür.
- Sayılardan TL, *, % gibi işaretleri temizle ve virgülleri noktaya çevir (Örn: *4,40 yerine 4.40 yaz).
"""

    print("OpenAI (gpt-4o-mini) fişi inceliyor, lütfen bekleyin...")
    
    try:
        # OpenAI API'sine istek atıyoruz
        response = client.chat.completions.create(
            model="gpt-4o-mini", # En hızlı ve en ucuz model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"OCR METNİ:\n{ocr_metni}"}
            ],
            temperature=0.0, # Yaratıcılık SIFIR, sadece gerçekler
            response_format={ "type": "json_object" } # SİHİRLİ DOKUNUŞ: Kesin JSON garantisi
        )
        
        # Gelen string formatındaki JSON'ı Python sözlüğüne çevir
        cevap_metni = response.choices[0].message.content
        json_verisi = json.loads(cevap_metni)
        
        return json_verisi
        
    except Exception as e:
        print(f"HATA OLUŞTU: {e}")
        return None

# Test için PaddleOCR'dan aldığımız çıktı metni
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

# Fonksiyonu ateşle
sonuc_json = fis_verilerini_cikar_openai(ornek_ocr_metni)

if sonuc_json:
    print("\n✅ BULUTTAN GELEN KUSURSUZ JSON ÇIKTISI:\n")
    print(json.dumps(sonuc_json, indent=4, ensure_ascii=False))