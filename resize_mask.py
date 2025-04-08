import cv2
import numpy as np
import argparse
import os

def resize_binary_mask_cv2(mask_path, output_path, target_width, target_height):
    """
    2値マスク画像を指定したサイズにリサイズする関数
    
    Parameters:
    mask_path (str): 入力マスク画像のパス
    output_path (str): 出力マスク画像のパス
    target_width (int): 目標の幅
    target_height (int): 目標の高さ
    
    Returns:
    numpy.ndarray: リサイズされたマスク画像
    """
    # マスク画像を読み込む
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise FileNotFoundError(f"ファイル '{mask_path}' を読み込めませんでした")
    
    # 2値化（すでに2値の場合でも確実に2値になるように）
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # リサイズ（INTER_NEARESTを使用して2値を維持）
    resized_mask = cv2.resize(binary_mask, (target_width, target_height), 
                             interpolation=cv2.INTER_NEAREST)
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 保存
    cv2.imwrite(output_path, resized_mask)
    
    return resized_mask

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='2値マスク画像をリサイズするツール')
    parser.add_argument('input', help='入力マスク画像のパス')
    parser.add_argument('output', help='出力マスク画像のパス')
    parser.add_argument('--width', type=int, required=True, help='目標の幅')
    parser.add_argument('--height', type=int, required=True, help='目標の高さ')
    
    args = parser.parse_args()
    
    try:
        # リサイズ実行
        resized = resize_binary_mask_cv2(args.input, args.output, args.width, args.height)
        
        # 元のサイズと新しいサイズを取得
        original_size = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE).shape[:2]
        new_size = resized.shape[:2]
        
        print(f"リサイズ成功: {args.input} ({original_size[1]}x{original_size[0]}) -> {args.output} ({args.width}x{args.height})")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

#Usage: python resize_mask.py input_mask.png output_mask.png --width 800 --height 600

if __name__ == "__main__":
    main()