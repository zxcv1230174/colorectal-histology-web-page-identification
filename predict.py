import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.stdout.reconfigure(encoding='utf-8')

# 定義類別標籤
labels = ['tumour_epithelium', 'simple_stroma', 'complex_stroma', 'immune_cell_conglomerates',
          'debris_and_mucus', 'mucosal_glands', 'adipose_tissue', 'background','unknown']

# 載入模型
def load_model():
    try:
        current_dir = os.path.dirname(__file__)
        relative_path = os.path.join(current_dir, 'DenseNet121_model_with_unknown.keras')
        model = tf.keras.models.load_model(relative_path)

        return model
    except Exception as e:
        print(f"錯誤：無法載入模型 - {e}")
        raise

# 載入並處理圖片
def load_and_prepare_image(img_path):
    try:
        # 載入圖片並調整大小
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = img / 255.0  # 圖像標準化

        # 計算每個通道的均值和標準差
        mean = tf.reduce_mean(image, axis=(0, 1), keepdims=True) #計算影像中每個通道（R、G、B）的均值（mean）
        std = tf.math.reduce_std(image, axis=(0, 1), keepdims=True) #計算影像中每個通道的標準差
        
        # 色彩標準化
        image = (image - mean) / (std + 1e-7) #此步驟通過從每個像素值中減去該通道的均值，並除以標準差來標準化影像的每個通道，1e-7的作用是防止標準差為零時出現除零錯誤。
        
        # 增加一個維度以符合模型輸入要求
        img_array = tf.expand_dims(img, axis=0)
        return img_array
    except Exception as e:
        print(f"錯誤：無法載入圖片 - {e}")
        raise

# 辨識
def predict(img_path):
    try:
        model = load_model()
        img_array = load_and_prepare_image(img_path)
        # 禁用進度條輸出
        prediction = model.predict(img_array, verbose=0)
        #print(prediction)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        return labels[predicted_class], confidence
    except Exception as e:
        print(f"辨識過程中發生錯誤：{e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: predict.py <圖片路徑>", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]

    try:
        predicted_label, recognition_rate = predict(img_path)
        if recognition_rate < 0.8:
            predicted_label = 'unknown'
        print(f"{predicted_label},{recognition_rate:.2%}")
    except Exception as e:
        print(f"發生錯誤：{e}", file=sys.stderr)
