import os
import random
import shutil
from glob import glob

# --- 1. 請在這裡修改您的配置 ---

# 原始數據集文件夾路徑
SOURCE_DATA_DIR = os.path.expanduser("~/datasets/img/2024_liyu_lake_voc/train")

# 新的劃分後數據集存放路徑
OUTPUT_DATA_DIR = os.path.expanduser("~/datasets/img/2024_liyu_lake_voc_split")

# 數據集劃分比例 [訓練集, 驗證集, 測試集]
# 確保三者相加等於 1
SPLIT_RATIOS = [0.8, 0.1, 0.1]

# 支持的圖片文件擴展名
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# 設定隨機種子以確保每次劃分結果可重現
random.seed(42)

# --- 2. 腳本主要邏輯 (通常無需修改) ---


def split_yolo_dataset():
    """
    將 YOLO 格式的數據集隨機劃分為訓練、驗證和測試集。
    """
    print("--- 開始處理數據集劃分 ---")

    # 檢查源文件夾路徑是否存在
    source_images_dir = os.path.join(SOURCE_DATA_DIR, "images")
    source_labels_dir = os.path.join(SOURCE_DATA_DIR, "labels")

    if not os.path.isdir(source_images_dir):
        print(f"錯誤: 圖片文件夾不存在于 '{source_images_dir}'")
        return
    if not os.path.isdir(source_labels_dir):
        print(f"錯誤: 標籤文件夾不存在于 '{source_labels_dir}'")
        return

    # 獲取所有圖片文件的路徑
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(glob(os.path.join(source_images_dir, f"*{ext}")))

    if not image_files:
        print("錯誤: 在源文件夾中沒有找到任何圖片文件。")
        return

    print(f"總共找到 {len(image_files)} 張圖片。")

    # 隨機打亂文件列表
    random.shuffle(image_files)

    # 計算各個集合的數量
    total_files = len(image_files)
    train_count = int(total_files * SPLIT_RATIOS[0])
    valid_count = int(total_files * SPLIT_RATIOS[1])
    # 剩下的都給測試集
    test_count = total_files - train_count - valid_count

    print(
        f"數據集劃分數量: 訓練集={train_count}, 驗證集={valid_count}, 測試集={test_count}"
    )

    # 分配文件路徑到不同集合
    train_files = image_files[:train_count]
    valid_files = image_files[train_count : train_count + valid_count]
    test_files = image_files[train_count + valid_count :]

    # 創建目標文件夾結構
    sets = {"train": train_files, "valid": valid_files, "test": test_files}
    for set_name in sets.keys():
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, set_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, set_name, "labels"), exist_ok=True)

    # 複製文件
    for set_name, files in sets.items():
        print(f"\n--- 正在複製 {set_name} 數據 ---")
        total_files_in_set = len(files)
        for i, image_path in enumerate(files):
            # 1. 獲取文件名和擴展名
            base_filename = os.path.basename(image_path)
            name_part, _ = os.path.splitext(base_filename)
            label_filename = f"{name_part}.txt"

            # 2. 構造標籤文件的源路徑
            label_path = os.path.join(source_labels_dir, label_filename)

            # 3. 構造目標路徑
            dest_image_path = os.path.join(
                OUTPUT_DATA_DIR, set_name, "images", base_filename
            )
            dest_label_path = os.path.join(
                OUTPUT_DATA_DIR, set_name, "labels", label_filename
            )

            # 4. 執行複製
            shutil.copy2(image_path, dest_image_path)
            if os.path.exists(label_path):
                shutil.copy2(label_path, dest_label_path)
            else:
                print(
                    f"警告: 圖片 '{base_filename}' 缺少對應的標籤文件 '{label_filename}'"
                )

            # 打印進度，使用 '\r' 讓光標回到行首，實現原地更新
            print(f"  進度: {i + 1}/{total_files_in_set}", end="\r")

        # 複製完成後打印一個換行符，避免下一行輸出覆蓋進度信息
        print(f"\n  '{set_name}' 集合複製完成。")

    print("\n--- 數據集劃分完成！---")
    print(f"數據已保存至: '{os.path.abspath(OUTPUT_DATA_DIR)}'")


if __name__ == "__main__":
    split_yolo_dataset()
