import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 定義類別標籤
labels = ['tumour epithelium', 'simple stroma', 'complex stroma', 'immune cell conglomerates',
          'debris and mucus', 'mucosal glands', 'adipose tissue', 'background','unknown']

# 載入模型
def load_model():
    try:
        model = tf.keras.models.load_model('C:\\Users\\ab090\\Desktop\\TEST\\DenseNet121_model_with_unknown.keras')

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
        
        # 計算每個通道的平均值和標準差
        mean = tf.reduce_mean(img, axis=(0, 1), keepdims=True)
        std = tf.math.reduce_std(img, axis=(0, 1), keepdims=True)
        
        # 色彩標準化
        img = (img - mean) / (std + 1e-7)
        
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
        print(f"{predicted_label},{recognition_rate:.2%}")
    except Exception as e:
        print(f"發生錯誤：{e}", file=sys.stderr)
