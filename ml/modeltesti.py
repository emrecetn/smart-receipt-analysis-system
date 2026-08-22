from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Eğittiğin modeli yükle (Yolun doğruluğunu terminaldeki çıktından aldım)
MODEL_PATH = 'runs/detect/fis_tespit_modeli/weights/best.pt'
model = YOLO(MODEL_PATH)

# 2. Test etmek istediğin fotoğrafın yolu
test_fotografi = 'testgorselleri/testgorsel2.jpeg' # Buraya test edeceğin görselin adını yaz

print(f"[{test_fotografi}] analiz ediliyor...")

# 3. Modele tahmin yaptır (conf=0.40 -> %40'tan fazla eminse çiz)
sonuclar = model.predict(source=test_fotografi, conf=0.40)

# 4. Modelin çizdiği kutuyu orijinal fotoğrafın üzerine işle
cizimli_resim = sonuclar[0].plot()

# OpenCV formatını Matplotlib'in düzgün göstereceği formata çevir
cizimli_resim_rgb = cv2.cvtColor(cizimli_resim, cv2.COLOR_BGR2RGB)

# 5. Ekranda göster
plt.figure(figsize=(10, 8))
plt.imshow(cizimli_resim_rgb)
plt.axis('off')
plt.title("Fiş Tespit Kutusu Kontrolü", fontsize=15, fontweight='bold')
plt.show()

# Eğer konsolda da bilgi görmek istersen:
if len(sonuclar[0].boxes) > 0:
    guven = float(sonuclar[0].boxes.conf[0]) * 100
    print(f"✅ Fiş bulundu! Modelin eminlik oranı: %{guven:.2f}")
else:
    print("❌ Model bu fotoğrafta fiş bulamadı.")