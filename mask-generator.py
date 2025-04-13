import os
import shutil
import glob
from PIL import Image
import numpy as np
import cv2

# 設定
data_dir = "plane"  # データディレクトリのパスを指定
original_mask_path = "mask.png"  # 元になるマスク画像のパス

# 画像フォルダのリスト
image_folders = ["images", "images_2", "images_4", "images_8"]

# 対応するマスクフォルダを作成・設定
for folder in image_folders:
    # マスクフォルダのパスを作成
    mask_folder = os.path.join(data_dir, f"masks_{folder.split('_')[-1] if '_' in folder else ''}")
    
    # フォルダが存在しない場合は作成
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
        print(f"Created directory: {mask_folder}")
    
    # 元の画像フォルダのパス
    image_folder_path = os.path.join(data_dir, folder)
    
    # 画像ファイルのリストを取得
    image_files = glob.glob(os.path.join(image_folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder_path, "*.png")) + \
                  glob.glob(os.path.join(image_folder_path, "*.jpeg"))
    
    # 元のマスク画像を読み込む
    original_mask = Image.open(original_mask_path).convert("L")

    original_mask = np.array(original_mask)
    #binary_mask = (original_mask > 127).astype(np.uint8)  # 0 or 1
    
    # 各画像に対応するマスクを作成
    for img_path in image_files:
        # ファイル名を取得
        img_filename = os.path.basename(img_path)
        
        # ターゲット画像のサイズを取得
        target_img = Image.open(img_path)
        target_size = target_img.size
        
        
        # マスクをリサイズ
        #resized_mask = original_mask.resize(target_size)

        resized_mask = cv2.resize(original_mask, (target_size), interpolation=cv2.INTER_NEAREST)
        
        # マスクの保存パス
        mask_save_path = os.path.join(mask_folder, img_filename)
        
        # マスクを保存
        #resized_mask.save(mask_save_path)
        cv2.imwrite(mask_save_path, resized_mask)
        
        print(f"Created mask: {mask_save_path}")

print("All masks created successfully!")
