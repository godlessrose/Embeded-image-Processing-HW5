import numpy as np
import scipy.signal as sig
from stm_ai_runner import AiRunner
from mfcc_func import create_mfcc_features
import time

# --- KULLANICI AYARLARI ---
COM_PORT = 'COM7'
BAUD_RATE = 115200
TEST_FILE = "C:/Users/godlessrose/Desktop/HW5/recordings/7_nicolas_29.wav"

# MFCC PARAMETRELERİ (Seninkilerle aynı)
FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13
window = sig.get_window("hamming", FFTSize)

def run_inference():
    # 1. Runner nesnesini oluştur
    runner = AiRunner()
    
    print(f"Bağlantı kuruluyor: {COM_PORT} ({BAUD_RATE} baud)...")
    
    # İSTEDİĞİN BAĞLANTI YAPISI:
    # AiRunner bu parametreleri arka plandaki sürücüye (pyserial) iletir.
    if  runner.connect('serial', port=COM_PORT, baudrate=BAUD_RATE):
        print("HATA: Bağlantı kurulamadı. Portu kontrol et pampişko!")
        return

    try:
        # 2. MODELİ KEŞFET (Senin istediğin yapı)
        print("Karttaki model aranıyor...")
        names = runner._drv.discover()
        if not names:
            print("HATA: Kartta model bulunamadı (Model not found).")
            return
        
        model_name = names[0]
        print(f"   Model bulundu: {model_name}")

        # 3. SES DOSYASINI İŞLE VE SHAPE AYARLA
        features, _ = create_mfcc_features([TEST_FILE], FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window)
        
        # Giriş şeklini (1, 26) olarak zorluyoruz (Senin input shape bilgin)
        # np.float32 çok önemli, STM32 bunu bekler.
        input_data_array = np.array(features[0], dtype=np.float32).reshape(1, 26)
        
        # AiRunner girdi olarak liste bekler
        input_data = [input_data_array]

        print(f"Veri hazırlandı. Shape: {input_data_array.shape}")

        # 4. MODELİ KOŞTUR (INVOKE)
        print(f"STM32 üzerinde {model_name} koşturuluyor...")
        # Runner'ın üst katman invoke metodunu kullanmak hata yönetimini kolaylaştırır
        outputs, profiler = runner._drv.invoke_sample(input_data, name=model_name)
        # 5. SONUÇLARI GÖSTER
        if outputs:
            # Gelen sonuç dizisini (1, 10) düzleştiriyoruz
            predictions = outputs[0].flatten()
            predicted_class = np.argmax(predictions)
            score = predictions[predicted_class]

            print("\n" + "🚀" * 15)
            print(f" TAHMİN EDİLEN RAKAM : {predicted_class}")
            print(f" GÜVEN (PROBABILITY) : %{score*100:.2f}")
            if profiler and 'duration_ms' in profiler:
                print(f" İŞLEM SÜRESİ        : {profiler['duration_ms']:.2f} ms")
            print("🚀" * 15)
        else:
            print("Sonuç alınamadı.")

    except Exception as e:
        print(f"\nSürücü seviyesinde bir aksilik çıktı: {e}")
    
    finally:
        # Bağlantıyı temiz bir şekilde kapat
        runner.disconnect()
        print("\nBağlantı sonlandırıldı.")

if __name__ == "__main__":
    run_inference()