import tensorflow as tf
import os

# 1. Eğitilmiş modeli yükle
model_path = "mlp_fsdd_model.h5"
if not os.path.exists(model_path):
    print(f"Hata: {model_path} bulunamadı. Lütfen önce training.py dosyasını çalıştırın.")
    exit()

model = tf.keras.models.load_model(model_path)

# 2. Modeli TensorFlow Lite (.tflite) formatına çevir
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# İsteğe bağlı: Optimizasyon (Model boyutunu küçültmek için)
# Bu satır modeli float32'den int8'e falan çevirmez ama gereksiz işlemleri atar.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# TFLite dosyasını kaydet (Kontrol amaçlı)
tflite_model_path = "model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite modeli oluşturuldu: {tflite_model_path}")

# 3. TFLite dosyasını C Array (Hex Dizisi) formatına çevir
def tflite_to_c_source(tflite_data, variable_name="g_model"):
    hex_array = [f"0x{val:02x}" for val in tflite_data]
    
    # C dosyasının içeriğini oluştur
    c_str = f"// Bu dosya otomatik oluşturulmuştur.\n"
    c_str += f"#include <stdint.h>\n\n"
    c_str += f"const unsigned int {variable_name}_len = {len(tflite_data)};\n"
    c_str += f"const unsigned char {variable_name}[] = {{\n"
    
    # Okunabilirlik için satır satır ekle (her satırda 12 byte)
    for i in range(0, len(hex_array), 12):
        line = ", ".join(hex_array[i:i+12])
        c_str += f"  {line},\n"
    
    c_str += "};\n"
    return c_str

# C kodunu oluştur ve kaydet
c_header_content = tflite_to_c_source(tflite_model, "g_model")
c_header_path = "model_data.h"

with open(c_header_path, "w", encoding="utf-8") as f:
    f.write(c_header_content)

print(f"Başarılı! Model C dizisine çevrildi: {c_header_path}")
print("Bu dosyayı (model_data.h) mikrodenetleyici projenize dahil edebilirsiniz.")