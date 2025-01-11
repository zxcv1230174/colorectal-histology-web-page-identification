import os
from PIL import Image
from pathlib import Path

current_dir = os.path.dirname(__file__)
# 來源目錄
source_dir = os.path.join(current_dir, 'unknown')
# 目標目錄
target_dir = os.path.join(current_dir, 'unknown_dataset')

# 建立目標目錄如果不存在
Path(target_dir).mkdir(parents=True, exist_ok=True)

# 轉換所有圖片為 PNG 並存儲到目標目錄
for image_path in Path(source_dir).rglob('*.*'):  # 匹配所有文件
    try:
        with Image.open(image_path) as img:
            # 獲取檔名，不帶副檔名
            file_name = image_path.stem
            # 保存 PNG 文件到目標目錄
            img.save(os.path.join(target_dir, f'{file_name}.png'))
            print(f"Converted: {image_path} -> {file_name}.png")
    except Exception as e:
        print(f"Failed to convert {image_path}: {e}")
