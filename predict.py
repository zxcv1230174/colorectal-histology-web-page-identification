import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只顯示警告和錯誤信息

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 定義分類標籤
labels = ['tumour epithelium', 'simple stroma', 'complex stroma', 'immune cell conglomerates',
          'debris and mucus', 'mucosal glands', 'adipose tissue', 'background']

# 載入模型和權重
def load_model_and_weights():
    input_shapes = (224, 224, 3)
    DN121_model = tf.keras.applications.DenseNet121(input_shape=input_shapes, include_top=False, weights='imagenet')
    DN121_model.trainable = False
    
    model_4 = tf.keras.models.Sequential([
        DN121_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    
    model_4.build((None, 224, 224, 3))  # 明確地建構模型
    model_4.load_weights(os.path.join(os.path.dirname(__file__), 'DenseNet121_model.weights.h5'))  # 載入模型權重
    
    return model_4

# 載入並準備圖片
def load_and_prepare_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # 圖像歸一化
        return img_array
    except Exception as e:
        raise RuntimeError(f"Error: Unable to load image - {e}")

# 執行預測
def predict(img_path):
    try:
        model = load_model_and_weights()
        img_array = load_and_prepare_image(img_path)
        prediction = model.predict(img_array)
        return prediction
    except Exception as e:
        raise RuntimeError(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: predict1.py <picture path>", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]

    try:
        prediction = predict(img_path)
        predicted_class = np.argmax(prediction)
        print(labels[predicted_class])
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)