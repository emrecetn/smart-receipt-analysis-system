from paddleocr import PaddleOCR
import cv2
import logging

# Logları sustur
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('root').setLevel(logging.ERROR)

print("PaddleOCR Yükleniyor...")
ocr = PaddleOCR(use_textline_orientation=True, lang='tr', enable_mkldnn=False)

img_path = 'testgorselleri/IMG_7130.jpg' 
img = cv2.imread(img_path)

if img is None:
    print("Hata: Resim okunamadı!")
    exit()

# Akıllı Boyutlandırma
max_boyut = 1500
h, w = img.shape[:2]
if max(h, w) > max_boyut:
    oran = max_boyut / float(max(h, w))
    img = cv2.resize(img, (int(w * oran), int(h * oran)))

gecici_yol = 'gecici_test.jpg'
cv2.imwrite(gecici_yol, img)

print("\n--- Fiş Okunuyor ---")
sonuclar = ocr.predict(gecici_yol)

tam_metin = ""

# Röntgenden öğrendiğimiz bilgiyle doğrudan 'rec_texts' listesini çekiyoruz
if sonuclar:
    for sonuc in sonuclar:
        # Veri obje formatındaysa
        if hasattr(sonuc, 'rec_texts') and sonuc.rec_texts:
            tam_metin += "\n".join(sonuc.rec_texts) + "\n"
        # Veri sözlük formatındaysa
        elif isinstance(sonuc, dict) and 'rec_texts' in sonuc:
            tam_metin += "\n".join(sonuc['rec_texts']) + "\n"

print("\n[TEMİZ OCR ÇIKTISI]:")
print("=" * 40)
print(tam_metin)
print("=" * 40)