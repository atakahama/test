#!/usr/bin/env python3
"""
Nerfstudio用のtransforms.jsonファイルにマスク画像のパス情報を追加するスクリプト

このスクリプトは元の画像パスから対応するマスクパスを生成し、
transforms.jsonファイルを更新します。

元の画像: images/filename.JPG
マスク画像: images/mask/filename_mask.JPG

使用方法:
python add_mask_paths.py --input transforms.json --output transforms_with_masks.json
"""

import json
import argparse
import os
from pathlib import Path


def add_mask_paths_to_transforms(input_file, output_file, mask_dir="mask", mask_suffix="_mask"):
    """
    transforms.jsonファイルにマスクパス情報を追加する

    Args:
        input_file (str): 入力のtransforms.jsonファイルパス
        output_file (str): 出力するtransforms.jsonファイルパス
        mask_dir (str): マスク画像が格納されているディレクトリ名
        mask_suffix (str): マスク画像のファイル名に追加される接尾辞
    """
    # JSONファイルを読み込む
    with open(input_file, 'r') as f:
        transforms_data = json.load(f)
    
    # framesにマスクパス情報を追加
    for frame in transforms_data.get('frames', []):
        if 'file_path' in frame:
            # 現在の画像パスから対応するマスクパスを生成
            file_path = frame['file_path']
            
            # 画像パスからファイル名と拡張子を取得
            path_obj = Path(file_path)
            parent_dir = path_obj.parent
            filename = path_obj.stem  # 拡張子なしのファイル名
            extension = path_obj.suffix  # .JPGなど
            
            # マスクパスを生成
            # 例: images/image001.JPG -> images/mask/image001_mask.JPG
            mask_path = str(parent_dir / mask_dir / f"{filename}{mask_suffix}{extension}")
            
            # マスクパス情報をフレームに追加
            frame['mask_path'] = mask_path
    
    # 更新されたデータを保存
    with open(output_file, 'w') as f:
        json.dump(transforms_data, f, indent=2)
    
    print(f"マスクパス情報を追加して {output_file} に保存しました")
    print(f"総フレーム数: {len(transforms_data.get('frames', []))}")
    
    # ファイルの先頭数行を表示して確認
    print("\n更新されたJSONの最初のフレーム例:")
    if transforms_data.get('frames'):
        example_frame = transforms_data['frames'][0]
        print(f"file_path: {example_frame.get('file_path')}")
        print(f"mask_path: {example_frame.get('mask_path')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transforms.jsonファイルにマスク画像のパス情報を追加します")
    parser.add_argument("--input", type=str, required=True, help="入力のtransforms.jsonファイルパス")
    parser.add_argument("--output", type=str, help="出力するtransforms.jsonファイルパス（未指定の場合は入力ファイルを上書き）")
    parser.add_argument("--mask-dir", type=str, default="mask", help="マスク画像のディレクトリ名（デフォルト: mask）")
    parser.add_argument("--mask-suffix", type=str, default="_mask", help="マスク画像のファイル名の接尾辞（デフォルト: _mask）")
    
    args = parser.parse_args()
    
    # 出力ファイルが指定されていない場合は入力ファイルを上書き
    output_file = args.output if args.output else args.input
    
    add_mask_paths_to_transforms(args.input, output_file, args.mask_dir, args.mask_suffix)
