import cv2
import numpy as np
import os
import argparse

def apply_mask_and_colorize(image_path, mask_path, output_path, color=[0, 0, 255]):
    """
    マスク画像の黒色領域に該当する部分を、元画像上で指定色に置き換える
    
    Parameters:
    image_path (str): 元画像のパス
    mask_path (str): マスク画像のパス
    output_path (str): 出力画像のパス
    color (list): 置換する色 [B, G, R]形式
    """
    # 画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像ファイル '{image_path}' を読み込めませんでした")
    
    # マスクの読み込み
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"マスクファイル '{mask_path}' を読み込めませんでした")
    
    # マスクのサイズが画像と異なる場合はリサイズ
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # マスクを2値化 (黒=0, 白=255になるようにする)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 黒色領域(0)を特定するために反転 (黒=255, 白=0)
    inverted_mask = cv2.bitwise_not(binary_mask)
    
    # 赤色の領域を作成
    colored_region = np.zeros_like(image)
    colored_region[:] = color  # [B, G, R]
    
    # マスクの黒色領域(inverted_maskの255)に対応する部分を赤色で塗る
    result = image.copy()
    result[inverted_mask == 255] = colored_region[inverted_mask == 255]
    
    # 出力ディレクトリがなければ作成
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 結果を保存
    cv2.imwrite(output_path, result)
    
    return result

def process_folder(input_folder, mask_path, output_folder, color=[0, 0, 255]):
    """
    フォルダ内の全画像に対してマスク処理を適用する
    
    Parameters:
    input_folder (str): 入力画像フォルダのパス
    mask_path (str): マスク画像のパス
    output_folder (str): 出力先フォルダのパス
    color (list): 置換する色 [B, G, R]形式
    """
    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)
    
    # 対応する画像形式
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 入力フォルダ内の画像ファイルをリストアップ
    processed_count = 0
    for filename in os.listdir(input_folder):
        base, ext = os.path.splitext(filename)
        if ext.lower() in img_extensions:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                apply_mask_and_colorize(input_path, mask_path, output_path, color)
                processed_count += 1
                print(f"処理完了: {input_path} -> {output_path}")
            except Exception as e:
                print(f"エラー ({filename}): {e}")
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description='マスク画像の黒色領域を元画像上で赤色に置換するツール')
    parser.add_argument('input_folder', help='処理する画像が入ったフォルダのパス')
    parser.add_argument('mask_path', help='マスク画像のパス')
    parser.add_argument('output_folder', help='処理後の画像を保存するフォルダのパス')
    parser.add_argument('--color', nargs=3, type=int, default=[0, 0, 255], 
                        help='置換する色をBGR形式で指定 (例: --color 0 0 255 で赤色)')
    
    args = parser.parse_args()
    
    try:
        count = process_folder(args.input_folder, args.mask_path, args.output_folder, args.color)
        print(f"処理完了: {count}個のファイルを処理しました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

#usage: python mask_colorize.py ./images/ ./mask.png ./output_images/ [--color 255(B) 0(G) 0(R)]

if __name__ == "__main__":
    main()