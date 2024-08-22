import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, callbacks
from pathlib import Path

# 設置環境變數
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# EarlyStopping 設定
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 調整圖像大小和數據類型，並進行色彩標準化
def preprocess_data(image_dict):
    image = tf.image.resize(image_dict['image'], img_size)
    image = tf.cast(image, tf.float32) / 255.0
    
    # 計算每個通道的均值和標準差
    mean = tf.reduce_mean(image, axis=(0, 1), keepdims=True)
    std = tf.math.reduce_std(image, axis=(0, 1), keepdims=True)
    
    # 色彩標準化
    image = (image - mean) / (std + 1e-7)
    
    label = image_dict['label']
    return image, label

train_ds = train_ds.map(preprocess_data)
valid_ds = valid_ds.map(preprocess_data)
test_ds = test_ds.map(preprocess_data)

# 載入 unknown 資料集
data_dir = 'C:\\Users\\ab090\\Desktop\\TEST\\unknown'

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    
    # 計算每個通道的均值和標準差
    mean = tf.reduce_mean(image, axis=(0, 1), keepdims=True)
    std = tf.math.reduce_std(image, axis=(0, 1), keepdims=True)
    
    # 色彩標準化
    image = (image - mean) / (std + 1e-7)
    
    return image

# 取得 unknown 類別的影像路徑
unknown_image_paths = [str(p) for p in Path(data_dir).glob('*.png')]
unknown_images = [load_and_preprocess_image(img_path) for img_path in unknown_image_paths]
unknown_images = tf.stack(unknown_images, axis=0)

# 為 unknown 類別標籤設置為新的類別索引
unknown_labels = tf.convert_to_tensor([num_classes] * len(unknown_image_paths), dtype=tf.int64)

# 建立 unknown 的數據集，並與主數據集合併
unknown_dataset = tf.data.Dataset.from_tensor_slices((unknown_images, unknown_labels))
unknown_dataset = unknown_dataset.shuffle(buffer_size=len(unknown_image_paths), reshuffle_each_iteration=False)

# 分割 unknown 數據集並合併到主數據集中
train_ds = train_ds.concatenate(unknown_dataset.take(int(0.7 * len(unknown_image_paths))))
valid_ds = valid_ds.concatenate(unknown_dataset.skip(int(0.7 * len(unknown_image_paths))).take(int(0.15 * len(unknown_image_paths))))
test_ds = test_ds.concatenate(unknown_dataset.skip(int(0.85 * len(unknown_image_paths))))

# 分批次處理並加速數據加載
train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 載入已經訓練好的模型
model = tf.keras.models.load_model('C:\\Users\\ab090\\Desktop\\TEST\\DenseNet121_model.keras')

# 移除原有的輸出層並添加新的輸出層
x = model.layers[-3].output  # 取得 GlobalAveragePooling2D 之後的層
x = layers.Dense(256, activation='relu', name='new_dense')(x)  # 指定新的名稱
x = layers.Dropout(0.5, name='new_dropout')(x)  # 指定新的名稱
new_outputs = layers.Dense(num_classes + 1, activation='softmax', name='new_output')(x)

# 建立新的模型
new_model = models.Model(inputs=model.input, outputs=new_outputs)

# 編譯新模型
new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練新模型
new_model.fit(train_ds, epochs=epoch, validation_data=valid_ds, callbacks=[early_stopping])

# 保存微調後的模型
new_model.save('C:\\Users\\ab090\\Desktop\\TEST\\DenseNet121_model_with_unknown.keras')

# 測試集評估
test_loss, test_acc = new_model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
