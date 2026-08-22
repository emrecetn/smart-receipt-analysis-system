from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Optional, List
from ultralytics import YOLO
from supabase import create_client, Client
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from PIL import Image
import pillow_heif
import numpy as np
import os
import cv2
import hashlib
import io
import json
import time
import base64

load_dotenv()

# iPhone'ların varsayılan fotoğraf formatı HEIC; Pillow bunu bu opener kayıtlı
# olmadan açamaz. cv2.imdecode HEIC'i hiç desteklemiyor.
pillow_heif.register_heif_opener()

MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(title="Akıllı Fiş Analiz Sistemi | API & Dev Portal", version="5.1")

# --- 1. SUPABASE ADMIN BAĞLANTISI ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
# service_role key .env dosyasından okunur (RLS'i bypass ettiği için asla koda gömülmez)
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase: Optional[Client] = None
try:
    # Service role kullanarak RLS engellerini aşıyoruz
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("[SİSTEM] Supabase Admin Bağlantısı Başarılı.")
except Exception as e:
    print(f"[HATA] Supabase bağlantısı kurulamadı: {e}")

# --- 2. CORS AYARLARI ---
# Bu API üçüncü parti (B2B) müşterilerin kendi domainlerinden çağırması için
# tasarlandı, bu yüzden allow_origins=["*"] kasıtlı. allow_credentials=False:
# kimlik doğrulama cookie değil X-API-Key header'ı ile yapılıyor, ayrıca
# tarayıcılar "*" origin ile credentials=true kombinasyonunu zaten reddeder.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def hash_api_key(raw_key: str) -> str:
    """API anahtarlarını veritabanına yazmadan önce hash'lemek için kullanılır.
    Frontend tarafında da (Web Crypto SubtleCrypto ile) aynı SHA-256 algoritması
    uygulanıyor; böylece veritabanında hiçbir zaman ham anahtar tutulmaz."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# --- 3. GÜVENLİK KAPISI (Dinamik API Key Kontrolü) ---
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Hem sabit test anahtarını hem de Supabase'deki canlı anahtarları doğrular."""

    # Durum A: Sabit Test Anahtarı
    if x_api_key == "sk_test_demo123456789":
        return "Demo Müşteri"

    # Durum B: Canlı Geliştirici Anahtarı (sk_live_...)
    if x_api_key.startswith("sk_live_"):
        if supabase is None:
            raise HTTPException(status_code=503, detail="Kimlik doğrulama servisi şu anda kullanılamıyor.")
        try:
            # Veritabanında ham anahtar değil, hash'i saklanıyor; aynı şekilde sorgula
            key_hash = hash_api_key(x_api_key)
            response = supabase.from_("api_keys").select("company_name").eq("api_key_hash", key_hash).execute()

            if response.data and len(response.data) > 0:
                # Anahtar bulundu, şirket adını döndür
                return response.data[0]["company_name"]
        except Exception as e:
            print(f"[HATA] API Key sorgulanırken hata oluştu: {e}")
            raise HTTPException(status_code=500, detail="Doğrulama servisi hatası.")

    # Durum C: Geçersiz Anahtar
    raise HTTPException(status_code=403, detail="Geçersiz API Anahtarı! Lütfen Geliştirici Panelinden kontrol edin.")

SYSTEM_PROMPT = """Sen uzman bir finansal veri çıkarma asistanısın.
Sana verilen fiş veya fatura görselini dikkatlice incele ve aşağıdaki kurallara harfiyen uyarak bir JSON objesi döndür.

ÇIKARILACAK ALANLAR VE KESİN KURALLAR:

1. "merchant_name" (String): Fişi kesen kurumun TAM TİCARİ UNVANI (Örn: "MİGROS TİC. A.Ş." veya "BARIŞ ECZANESİ"). Fişin en üstünde yer alır. Eğer fişte hem işletme adı hem de şahıs adı (Örn: eczane sahibinin adı) varsa, her zaman İŞLETME ADINI / TİCARİ UNVANI tercih et. Şahıs isimlerini yoksay.

2. "tax_id" (String): Vergi Kimlik Numarası (VKN). ÇOK ÖNEMLİ KURAL: Fişte hem "Vergi No" (VKN / V.D. ibaresi yanındaki) hem de "TC Kimlik Numarası" (TCKN) bulunuyorsa, KESİNLİKLE Vergi Numarasını (VKN) seç ve TC Kimlik Numarasını yoksay. Sadece ve sadece fişte hiçbir Vergi Numarası yoksa (şahıs şirketi durumu) TC'yi al. Sadece rakamları al, VD ismini alma.

3. "date" (String): Fişin tarihi. Saat bilgisini alma. Çıktıyı SADECE "GG.AA.YYYY" formatında ver (Örn: "25.04.2026").

4. "receipt_no" (String): Fiş, Fatura veya Belge numarası. ("Fiş No", "Belge No", "Fis" gibi etiketlerin yanındaki numara).

5. "total_amount" (Float): Toplam ödenecek tutar. KESİNLİKLE float formatında olmalı ve ondalık ayracı olarak VİRGÜL DEĞİL, NOKTA kullanılmalıdır (Örn: 125.50). "TL", "*", "₺" gibi sembolleri temizle.

6. "tax_amount" (Float): Toplam KDV tutarı. Fişin üzerinde açıkça "TOPKDV", "TOP. KDV" veya "TOPLAM KDV" yazan satırı bul ve sadece tam karşısındaki sayıyı oku. Kendin hesaplama yapma. Float formatında ve nokta ile ayırarak yaz (Örn: 61.97).

7. "tax_breakdown" (Array): Fişteki KDV oranlarının (%1, %10, %20 vb.) ve bu oranlara ait KDV tutarlarının detaylı dökümü. Eğer fişte böyle bir ayrım varsa, Fişin alt kısımlarındaki KDV detay tablosunu dikkatlice oku. Fişte birden fazla KDV oranı varsa (Örn: hem %1 hem %20 gibi), hiçbirini atlamadan hepsini listele.
- Tablodaki satırlarda en solda yazan KDV oranını ("rate") ve onunla aynı satırda en sağda yazan KDV tutarını ("amount") eşleştir.
- Ortada yazan vergisiz matrah tutarlarını KDV sanıp alma.
- Sütunlar arasındaki geniş boşluklar seni yanıltmasın, aynı satırı takip et.
Şu formatta JSON listesi olarak ver: [{"rate": 1, "amount": 0.34}, {"rate": 20, "amount": 61.63}].
Oran ("rate") integer (sadece sayı), KDV tutarı ("amount") float olmalı ve nokta kullanılmalı. Eğer fişte KDV ayrımı yoksa boş liste [] döndür.

GENEL KURALLAR:
- SADECE geçerli bir JSON objesi döndür, dışına hiçbir metin veya açıklama yazma.
- Okunamayan, fişte bulunmayan veya emin olamadığın alanlar için değer uydurma, `null` değerini ata."""


class TaxBreakdownItem(BaseModel):
    rate: int
    amount: float


class ReceiptData(BaseModel):
    merchant_name: Optional[str] = None
    tax_id: Optional[str] = None
    date: Optional[str] = None
    receipt_no: Optional[str] = None
    total_amount: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_breakdown: List[TaxBreakdownItem] = []


class Durations(BaseModel):
    yolo_seconds: float
    openai_seconds: float
    total_seconds: float


class ExtractReceiptResponse(BaseModel):
    status: str
    durations: Durations
    yolo_detected: bool
    cropped_image_base64: str
    data: ReceiptData


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    supabase_connected: bool


def decode_image(content: bytes):
    """Yüklenen dosyayı BGR numpy dizisine çevirir. Önce hızlı yol olan OpenCV'yi
    dener; JPEG/PNG/WEBP dışındaki formatlarda (örn. HEIC) None dönerse Pillow'a
    düşer. İkisi de başarısız olursa None döner (bozuk/desteklenmeyen dosya)."""
    if not content:
        return None

    arr = np.frombuffer(content, dtype=np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is not None:
        return img
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# --- 4. MODELLERİ YÜKLEME ---
print("[BAŞLATICI] YOLO Yükleniyor...")
yolo_model = YOLO('ml/runs/detect/fis_tespit_modeli/weights/best.pt')

# OpenAI Bağlantısı
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
print("[BAŞLATICI] OpenAI Vision Modülü Hazır.")

# --- 5. ENDPOINTLER ---

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("index.html")


@app.get("/health", response_model=HealthResponse, tags=["Sistem"], summary="Servis sağlık kontrolü")
async def health_check():
    return HealthResponse(
        status="ok",
        yolo_loaded=yolo_model is not None,
        supabase_connected=supabase is not None,
    )


@app.post(
    "/api/v1/extract-receipt",
    response_model=ExtractReceiptResponse,
    tags=["Fiş Analizi"],
    summary="Fiş/fatura görselinden finansal veri çıkar",
)
async def extract_receipt(
    file: UploadFile = File(...),
    company_name: str = Depends(verify_api_key)
):
    print(f"\n[{file.filename}] >>> İSTEK GELDİ | Firma: {company_name}")
    start_time_total = time.time()

    # AŞAMA 0: DOSYA DOĞRULAMA (boyut + okunabilirlik)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Boş dosya yüklendi.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"Dosya çok büyük (maksimum {MAX_FILE_SIZE_MB}MB).")

    img = decode_image(content)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Görüntü okunamadı veya bozuk. Desteklenen formatlar: JPEG, PNG, WEBP, HEIC."
        )

    # AŞAMA 1: YOLO İLE FİŞİ ARKA PLANDAN İZOLE ET
    try:
        yolo_start = time.time()
        yolo_sonuclar = yolo_model(img, verbose=False)
    except Exception as e:
        print(f"[HATA] YOLO tespiti başarısız: {e}")
        raise HTTPException(status_code=500, detail="Fiş tespiti sırasında bir hata oluştu.")

    # YOLO tespit ederse ve kutu geçerliyse kırp; aksi halde orijinal görselle devam et
    yolo_detected = False
    if len(yolo_sonuclar) > 0 and len(yolo_sonuclar[0].boxes) > 0:
        kutu = yolo_sonuclar[0].boxes[0]
        x1, y1, x2, y2 = map(int, kutu.xyxy[0])
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]
            yolo_detected = True

    yolo_sure = round(time.time() - yolo_start, 2)

    # Optimizasyon: Boyutlandırma
    max_boyut = 1500
    h, w = img.shape[:2]
    if max(h, w) > max_boyut:
        oran = max_boyut / float(max(h, w))
        img = cv2.resize(img, (int(w * oran), int(h * oran)))

    ok, buffer = cv2.imencode('.jpg', img)
    if not ok:
        raise HTTPException(status_code=500, detail="Görüntü işlenirken bir hata oluştu.")
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # AŞAMA 2: OPENAI VISION ANALİZİ
    ai_start = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
    except OpenAIError as e:
        print(f"[HATA] OpenAI isteği başarısız: {e}")
        raise HTTPException(status_code=502, detail="Yapay zeka servisine şu anda ulaşılamıyor. Lütfen tekrar deneyin.")

    try:
        json_verisi = json.loads(response.choices[0].message.content)
        receipt_data = ReceiptData(**json_verisi)
    except (json.JSONDecodeError, ValidationError, IndexError, AttributeError, TypeError) as e:
        print(f"[HATA] Model çıktısı ayrıştırılamadı/doğrulanamadı: {e}")
        raise HTTPException(status_code=502, detail="Yapay zeka modeli geçersiz formatta yanıt döndürdü.")

    ai_sure = round(time.time() - ai_start, 2)
    total_sure = round(time.time() - start_time_total, 2)

    return ExtractReceiptResponse(
        status="success",
        durations=Durations(yolo_seconds=yolo_sure, openai_seconds=ai_sure, total_seconds=total_sure),
        yolo_detected=yolo_detected,
        cropped_image_base64=base64_image,
        data=receipt_data,
    )