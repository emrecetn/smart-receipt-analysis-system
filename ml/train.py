from ultralytics import YOLO

# 1. Modeli yükle (yolov8n.pt küçük ve hızlıdır, yolov8m.pt daha doğrudur)
model = YOLO('yolov8n.pt') 

# 2. Modeli eğit
results = model.train(
    data='dataset/data.yaml',  # data.yaml dosyanın yolu
    epochs=100,                # Eğitim turu (başta 50-100 yeter)
    imgsz=640,                 # Görüntü boyutu
    batch=16,                  # Batch size (GPU belleğine göre değiştir)
    name='fis_tespit_modeli'   # Proje adı
)
