import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, callbacks

# 設置環境變數
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 載入數據集
[train_ds, valid_ds, test_ds], info = tfds.load('colorectal_histology',
                                                split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
                                                shuffle_files=True,
                                                with_info=True)

# 參數設定
input_shapes = (224,224,3)
img_size = input_shapes[:2]
batch_size = 32
num_classes = info.features['label'].num_classes
epoch = 50
current_dir = os.path.dirname(__file__)
model_architecture = os.path.join(current_dir, 'DenseNet121_model.keras')

# 預處理函數
def preprocess_data(image_dict):
    image = tf.image.resize(image_dict['image'], img_size)
    image = tf.cast(image, tf.float32) / 255.0 #將原本資料集的資料類型從uint8轉換成float32是為了符合我們平常使用的圖片
    
    # 計算每個通道的均值和標準差
    mean = tf.reduce_mean(image, axis=(0, 1), keepdims=True) #計算影像中每個通道（R、G、B）的均值（mean）
    std = tf.math.reduce_std(image, axis=(0, 1), keepdims=True) #計算影像中每個通道的標準差
    
    # 色彩標準化
    image = (image - mean) / (std + 1e-7) #此步驟通過從每個像素值中減去該通道的均值，並除以標準差來標準化影像的每個通道，1e-7的作用是防止標準差為零時出現除零錯誤。
    
    label = image_dict['label']
    return image, label

# 將預處理函數應用到數據集，並將資料分批
train_data = train_ds.map(preprocess_data).batch(batch_size)
valid_data = valid_ds.map(preprocess_data).batch(batch_size)
test_data = test_ds.map(preprocess_data).batch(batch_size)

# EarlyStopping設定
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 建立 DenseNet121 模型
DN121_model = tf.keras.applications.DenseNet121(input_shape=input_shapes, include_top=False, weights='imagenet')
DN121_model.trainable = False # 凍結 DenseNet121 的權重

# 建立完整模型使用 Functional API
inputs = tf.keras.Input(shape=input_shapes)
x = DN121_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(train_data, epochs=epoch, validation_data=valid_data, callbacks=[early_stopping])

# 保存完整模型
model.save(model_architecture)

# 測試集評估
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
