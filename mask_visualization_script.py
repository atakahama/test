#!/usr/bin/env python
"""
マスク画像が正しく読み込まれているか確認するためのスクリプト
Nerfstudioのデータローディングパイプラインを使用して、マスク画像を可視化します
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Nerfstudioのインポート - 修正バージョン
try:
    # 新しいバージョンのインポートを試みる
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio
    from nerfstudio.data.dataparsers.colmap_dataparser import Colmap
    NerfstudioParser = Nerfstudio
    ColmapParser = Colmap
except ImportError:
    try:
        # 別の可能性のあるインポートパス
        from nerfstudio.data.dataparsers.nerfstudio_dataparser import NSDataParser
        from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser
        NerfstudioParser = NSDataParser
        ColmapParser = ColmapDataParser
    except ImportError:
        print("警告: Nerfstudioのデータパーサークラスをインポートできませんでした。")
        print("代わりにより基本的なアプローチでマスクファイルを検索します。")
        NerfstudioParser = None
        ColmapParser = None


def find_mask_files(data_dir, mask_suffix="_mask"):
    """
    ディレクトリ内のマスクファイルを手動で検索します。
    Nerfstudioのパーサーが利用できない場合のフォールバック機能です。
    """
    data_dir = Path(data_dir)
    
    # 画像フォルダを探す
    images_dir = data_dir / "images"
    if not images_dir.exists():
        images_dir = data_dir  # 画像がメインディレクトリにある可能性も
    
    # マスクフォルダを探す
    masks_dir = data_dir / "masks"
    if not masks_dir.exists():
        masks_dir = images_dir  # マスクが画像と同じフォルダにある可能性も
    
    # 画像とマスクのペアを見つける
    results = []
    
    # 画像ファイルをリスト
    image_extensions = [".jpg", ".jpeg", ".png"]
    images = []
    for ext in image_extensions:
        images.extend(list(images_dir.glob(f"*{ext}")))
    
    for img_path in images:
        base_name = img_path.stem
        # マスクファイルの可能性を確認
        for ext in image_extensions:
            # 同じフォルダでのマスク
            mask_path_same_dir = images_dir / f"{base_name}{mask_suffix}{ext}"
            # マスクフォルダ内のマスク
            mask_path_mask_dir = masks_dir / f"{base_name}{mask_suffix}{ext}"
            
            # マスクファイルが存在するか確認
            if mask_path_same_dir.exists():
                results.append((str(img_path), str(mask_path_same_dir)))
                break
            elif mask_path_mask_dir.exists():
                results.append((str(img_path), str(mask_path_mask_dir)))
                break
    
    return results


def visualize_masks(image_mask_pairs, output_dir):
    """
    画像とマスクのペアを可視化します
    """
    for i, (image_path, mask_path) in enumerate(image_mask_pairs):
        print(f"処理中: {mask_path}")
        
        # マスク画像の読み込み
        if mask_path is not None and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            
            # 画像の読み込み
            image = np.array(Image.open(image_path))
            
            # プロット作成
            plt.figure(figsize=(15, 5))
            
            # 元画像表示
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title(f"元画像: {os.path.basename(image_path)}")
            plt.axis('off')
            
            # マスク画像表示
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f"マスク: {os.path.basename(mask_path)}")
            plt.axis('off')
            
            # マスク適用画像表示（元画像にマスクを適用）
            plt.subplot(1, 3, 3)
            # マスクをバイナリにする（0か1）
            binary_mask = mask > 0
            
            # マスクのチャンネル数を確認
            if len(binary_mask.shape) == 2:
                # グレースケールマスクを3チャンネルに拡張
                binary_mask = np.stack([binary_mask] * 3, axis=2)
            
            # 画像サイズとマスクサイズが一致するか確認
            if image.shape[:2] != binary_mask.shape[:2]:
                print(f"警告: 画像サイズ{image.shape[:2]}とマスクサイズ{binary_mask.shape[:2]}が一致しません")
                # リサイズして表示用に調整
                from skimage.transform import resize
                binary_mask = resize(binary_mask, image.shape[:2] + (3,), order=0, preserve_range=True)
            
            # マスクを適用した画像を表示
            masked_image = image.copy()
            if len(masked_image.shape) == 3 and masked_image.shape[2] == 3:
                masked_image = masked_image * binary_mask
            plt.imshow(masked_image)
            plt.title("マスク適用後")
            plt.axis('off')
            
            # 保存
            output_file = os.path.join(output_dir, f"mask_debug_{i:03d}.png")
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
            
            print(f"保存しました: {output_file}")
            
            # マスクの統計情報表示
            if len(mask.shape) == 2:  # グレースケールマスク
                unique_values = np.unique(mask)
                white_pixels = np.sum(mask > 127)  # 127より大きい値を白とみなす
                black_pixels = np.sum(mask <= 127)  # 127以下の値を黒とみなす
                total_pixels = mask.shape[0] * mask.shape[1]
                
                print(f"  マスクの統計情報:")
                print(f"  - サイズ: {mask.shape}")
                print(f"  - ユニークな値: {unique_values}")
                print(f"  - 白ピクセル (>127): {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
                print(f"  - 黒ピクセル (≤127): {black_pixels} ({black_pixels/total_pixels*100:.1f}%)")
            else:
                print(f"  マスクの統計情報:")
                print(f"  - サイズ: {mask.shape} (マルチチャンネルマスク)")
        else:
            print(f"警告: マスクファイルが見つかりません: {mask_path}")


def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Nerfstudioのマスク画像可視化ツール")
    parser.add_argument("--data", type=str, required=True, help="データパス（transforms.jsonのあるフォルダまたはCOLMAPのフォルダ）")
    parser.add_argument("--output", type=str, default="mask_debug", help="出力ディレクトリ")
    parser.add_argument("--parser", type=str, choices=["nerfstudio", "colmap", "simple"], default="simple", 
                        help="使用するデータパーサー (デフォルト: simple)")
    parser.add_argument("--mask-suffix", type=str, default="_mask", help="マスク画像の接尾辞")
    args = parser.parse_args()

    # 出力ディレクトリ作成
    os.makedirs(args.output, exist_ok=True)
    
    print(f"==== マスク可視化を開始: {args.data} ====")
    
    # Nerfstudioパーサーが使用可能かつリクエストされている場合
    if args.parser != "simple" and (NerfstudioParser is not None or ColmapParser is not None):
        try:
            # データパーサーの設定
            if args.parser == "nerfstudio" and NerfstudioParser is not None:
                # 新しいAPIのバージョン（Nerfstudio）
                from nerfstudio.configs.dataparser_configs import NerfstudioDataParserConfig
                parser_config = NerfstudioDataParserConfig(
                    data=Path(args.data),
                    mask_suffix=args.mask_suffix
                )
                data_parser = parser_config.setup()
            elif args.parser == "colmap" and ColmapParser is not None:
                # 新しいAPIのバージョン（Colmap）
                from nerfstudio.configs.dataparser_configs import ColmapDataParserConfig
                parser_config = ColmapDataParserConfig(
                    data=Path(args.data),
                    mask_suffix=args.mask_suffix
                )
                data_parser = parser_config.setup()
            else:
                raise ImportError("リクエストされたパーサーを使用できません")
                
            # データパーサーからデータを取得
            data = data_parser.get_dataparser_outputs("train")
            
            # マスクの有無をチェック
            has_mask = data.metadata is not None and "mask_filenames" in data.metadata
            
            if has_mask:
                mask_filenames = data.metadata["mask_filenames"]
                image_filenames = data.image_filenames
                print(f"マスクファイルが見つかりました: {len(mask_filenames)} 個")
                
                # 画像とマスクのペアを作成
                image_mask_pairs = []
                for img_path, mask_path in zip(image_filenames, mask_filenames):
                    if mask_path is not None:
                        image_mask_pairs.append((img_path, mask_path))
                
                # 可視化
                visualize_masks(image_mask_pairs, args.output)
            else:
                print("Nerfstudioパーサーではマスクが見つかりませんでした。シンプルなファイル検索を試みます。")
                # シンプルな方法でマスクを検索
                image_mask_pairs = find_mask_files(args.data, args.mask_suffix)
                if image_mask_pairs:
                    print(f"シンプル検索でマスクファイルが見つかりました: {len(image_mask_pairs)} 個")
                    visualize_masks(image_mask_pairs, args.output)
                else:
                    print("マスクファイルが見つかりませんでした。")
        except Exception as e:
            print(f"Nerfstudioパーサーでエラーが発生しました: {e}")
            print("シンプルなファイル検索にフォールバックします...")
            # シンプルな方法でマスクを検索
            image_mask_pairs = find_mask_files(args.data, args.mask_suffix)
            if image_mask_pairs:
                print(f"シンプル検索でマスクファイルが見つかりました: {len(image_mask_pairs)} 個")
                visualize_masks(image_mask_pairs, args.output)
            else:
                print("マスクファイルが見つかりませんでした。")
    else:
        # シンプルな方法でマスクを検索
        image_mask_pairs = find_mask_files(args.data, args.mask_suffix)
        if image_mask_pairs:
            print(f"マスクファイルが見つかりました: {len(image_mask_pairs)} 個")
            visualize_masks(image_mask_pairs, args.output)
        else:
            print("マスクファイルが見つかりませんでした。")
            print("以下のことを確認してください:")
            print("1. マスク画像が正しい命名規則に従っているか (_mask接尾辞など)")
            print("2. マスク画像が画像と同じフォルダか 'masks' サブフォルダにあるか")
            print("3. マスク画像の形式が適切か (PNG, JPG など)")


if __name__ == "__main__":
    main()
