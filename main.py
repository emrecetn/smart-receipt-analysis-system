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

# iPhone's default photo format is HEIC; Pillow can't open it without this
# opener registered. cv2.imdecode doesn't support HEIC at all.
pillow_heif.register_heif_opener()

MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(title="Smart Receipt Analysis System | API & Dev Portal", version="5.1")

# --- 1. SUPABASE ADMIN CONNECTION ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
# service_role key is read from .env (never hardcoded, since it bypasses RLS)
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase: Optional[Client] = None
try:
    # Using the service role to bypass RLS restrictions
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("[SYSTEM] Supabase Admin Connection Successful.")
except Exception as e:
    print(f"[ERROR] Could not establish Supabase connection: {e}")

# --- 2. CORS SETTINGS ---
# This API is designed to be called by third-party (B2B) customers from
# their own domains, so allow_origins=["*"] is intentional. allow_credentials=False
# because auth is done via the X-API-Key header, not cookies — and browsers
# reject the "*" origin + credentials=true combination anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def hash_api_key(raw_key: str) -> str:
    """Used to hash API keys before writing them to the database.
    The frontend applies the same SHA-256 algorithm (via Web Crypto
    SubtleCrypto), so the raw key is never stored in the database."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# --- 3. SECURITY GATE (Dynamic API Key Check) ---
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Validates both the fixed demo key and live keys stored in Supabase."""

    # Case A: Fixed demo key
    if x_api_key == "sk_test_demo123456789":
        return "Demo Müşteri"

    # Case B: Live developer key (sk_live_...)
    if x_api_key.startswith("sk_live_"):
        if supabase is None:
            raise HTTPException(status_code=503, detail="Kimlik doğrulama servisi şu anda kullanılamıyor.")
        try:
            # Only the hash is stored in the database, not the raw key; query the same way
            key_hash = hash_api_key(x_api_key)
            response = supabase.from_("api_keys").select("company_name").eq("api_key_hash", key_hash).execute()

            if response.data and len(response.data) > 0:
                # Key found, return the company name
                return response.data[0]["company_name"]
        except Exception as e:
            print(f"[ERROR] Error occurred while querying API Key: {e}")
            raise HTTPException(status_code=500, detail="Doğrulama servisi hatası.")

    # Case C: Invalid key
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
    """Converts the uploaded file into a BGR numpy array. Tries the fast path
    (OpenCV) first; falls back to Pillow if it returns None for formats other
    than JPEG/PNG/WEBP (e.g. HEIC). Returns None if both fail (corrupt/unsupported file)."""
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


# --- 4. LOADING MODELS ---
print("[STARTUP] Loading YOLO...")
yolo_model = YOLO('ml/runs/detect/fis_tespit_modeli/weights/best.pt')

# OpenAI connection
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
print("[STARTUP] OpenAI Vision Module Ready.")

# --- 5. ENDPOINTS ---

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Service health check")
async def health_check():
    return HealthResponse(
        status="ok",
        yolo_loaded=yolo_model is not None,
        supabase_connected=supabase is not None,
    )


@app.post(
    "/api/v1/extract-receipt",
    response_model=ExtractReceiptResponse,
    tags=["Receipt Analysis"],
    summary="Extract financial data from a receipt/invoice image",
)
async def extract_receipt(
    file: UploadFile = File(...),
    company_name: str = Depends(verify_api_key)
):
    print(f"\n[{file.filename}] >>> REQUEST RECEIVED | Company: {company_name}")
    start_time_total = time.time()

    # STAGE 0: FILE VALIDATION (size + readability)
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

    # STAGE 1: ISOLATE THE RECEIPT FROM THE BACKGROUND WITH YOLO
    try:
        yolo_start = time.time()
        yolo_sonuclar = yolo_model(img, verbose=False)
    except Exception as e:
        print(f"[ERROR] YOLO detection failed: {e}")
        raise HTTPException(status_code=500, detail="Fiş tespiti sırasında bir hata oluştu.")

    # Crop if YOLO detected something and the box is valid; otherwise continue with the original image
    yolo_detected = False
    if len(yolo_sonuclar) > 0 and len(yolo_sonuclar[0].boxes) > 0:
        kutu = yolo_sonuclar[0].boxes[0]
        x1, y1, x2, y2 = map(int, kutu.xyxy[0])
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]
            yolo_detected = True

    yolo_sure = round(time.time() - yolo_start, 2)

    # Optimization: resizing
    max_boyut = 1500
    h, w = img.shape[:2]
    if max(h, w) > max_boyut:
        oran = max_boyut / float(max(h, w))
        img = cv2.resize(img, (int(w * oran), int(h * oran)))

    ok, buffer = cv2.imencode('.jpg', img)
    if not ok:
        raise HTTPException(status_code=500, detail="Görüntü işlenirken bir hata oluştu.")
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # STAGE 2: OPENAI VISION ANALYSIS
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
        print(f"[ERROR] OpenAI request failed: {e}")
        raise HTTPException(status_code=502, detail="Yapay zeka servisine şu anda ulaşılamıyor. Lütfen tekrar deneyin.")

    try:
        json_verisi = json.loads(response.choices[0].message.content)
        receipt_data = ReceiptData(**json_verisi)
    except (json.JSONDecodeError, ValidationError, IndexError, AttributeError, TypeError) as e:
        print(f"[ERROR] Could not parse/validate model output: {e}")
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