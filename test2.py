import numpy as np
import cv2
import os
import mlist
import random # Rastgele seçim için
from stm_ai_runner import AiRunner

# --- AYARLAR ---
COM_PORT = 'COM7'
BAUD_RATE = 115200
dataset_dir = "MNIST-dataset"
test_img_path = os.path.join(dataset_dir, "t10k-images.idx3-ubyte")
test_label_path = os.path.join(dataset_dir, "t10k-labels.idx1-ubyte")

def run_hu_moment_test():
    runner = AiRunner()
    
    if runner.connect('serial', port=COM_PORT, baudrate=BAUD_RATE):
        print("HATA: Bağlantı kurulamadı!")
        return

    try:
        # 1. Model Keşfi
        names = runner._drv.discover()
        if not names: return
        model_name = names[0]

        # 2. Verileri Yükle
        test_images = mlist.load_images(test_img_path)
        test_labels = mlist.load_labels(test_label_path)

        # ---------------------------------------------------------
        # İSTEDİĞİN SAYIYI SEÇME BÖLÜMÜ
        # ---------------------------------------------------------
        # SEÇENEK A: Rastgele bir resim seç (Her seferinde farklı gelir)
        idx = random.randint(0, len(test_images) - 1)
        
        # SEÇENEK B: Elinle sayı gir (Örn: 123. resmi denemek istersen alttaki satırı aç)
        # idx = 123 
        
        test_img = test_images[idx]
        actual_label = test_labels[idx]
        # ---------------------------------------------------------

        # 3. Hu Moments Hesaplama (Senin extraction mantığın)
        moments = cv2.moments(test_img, True) 
        huMoments = cv2.HuMoments(moments).flatten()
        input_data = np.array(huMoments, dtype=np.float32).reshape(1, 7)

        print(f"\n--- TEST BAŞLADI (İndeks: {idx}) ---")
        print(f"Gerçek Rakam: {actual_label}")

        # 4. Tahmin
        outputs, profiler = runner._drv.invoke_sample([input_data], name=model_name)

        # 5. Sonuç
        if outputs:
            predictions = outputs[0].flatten()
            predicted_class = np.argmax(predictions)
            score = predictions[predicted_class]

            

            print("\n" + "⭐" * 20)
            print(f" GERÇEK ETİKET     : {actual_label}")
            print(f" STM32 TAHMİNİ     : {predicted_class}")
            print(f" GÜVEN ORANI       : %{score*100:.2f}")
            print("⭐" * 20)
            
            # Eğer tahmin yanlışsa uyarı ver
            if actual_label != predicted_class:
                print("⚠️  Tahmin yanlış çıktı! Hu Moments bu rakamda zorlanmış olabilir.")
        
    except Exception as e:
        print(f"Hata: {e}")
    finally:
        runner.disconnect()

if __name__ == "__main__":
    run_hu_moment_test()