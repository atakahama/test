import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

def pil_to_numpy(im):
    """NeRFStudioのpil_to_numpy関数の再実装"""
    # Load in image completely (PIL defaults to lazy loading)
    im.load()

    # Unpack data
    e = Image._getencoder(im.mode, "raw", im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        _, s, d = e.encode(bufsize)
        mem[offset : offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)

    return data

def test_mask_conversion(mask_file_path):
    """マスク画像の変換処理をテストする関数"""
    # 画像の読み込み
    pil_mask = Image.open(mask_file_path)
    print(f"PIL画像情報:")
    print(f"  - モード: {pil_mask.mode}")
    print(f"  - サイズ: {pil_mask.size}")
    
    # 標準のNumPy変換
    std_numpy = np.array(pil_mask)
    print(f"\n標準np.array変換:")
    print(f"  - データ型: {std_numpy.dtype}")
    print(f"  - 形状: {std_numpy.shape}")
    print(f"  - ユニーク値: {np.unique(std_numpy)}")
    print(f"  - 最小値: {np.min(std_numpy)}, 最大値: {np.max(std_numpy)}")
    
    # カスタムpil_to_numpy関数での変換
    custom_numpy = pil_to_numpy(pil_mask)
    print(f"\npil_to_numpy変換:")
    print(f"  - データ型: {custom_numpy.dtype}")
    print(f"  - 形状: {custom_numpy.shape}")
    print(f"  - ユニーク値: {np.unique(custom_numpy)}")
    print(f"  - 最小値: {np.min(custom_numpy)}, 最大値: {np.max(custom_numpy)}")
    
    # 差異の解析
    if std_numpy.shape == custom_numpy.shape:
        diff = std_numpy != custom_numpy
        diff_count = np.sum(diff)
        diff_percent = diff_count / std_numpy.size * 100
        print(f"\n変換値の差異:")
        print(f"  - 差異ピクセル数: {diff_count} ({diff_percent:.2f}%)")
        
        if diff_count > 0:
            diff_indices = np.where(diff)
            print(f"  - 差異サンプル (最大10個):")
            for i in range(min(10, diff_count)):
                idx = (diff_indices[0][i], diff_indices[1][i] if len(diff_indices) > 1 else None)
                idx_str = f"[{idx[0]}, {idx[1]}]" if idx[1] is not None else f"[{idx[0]}]"
                print(f"    位置 {idx_str}: std={std_numpy[idx]}, custom={custom_numpy[idx]}")
    
    # ブール変換テスト
    std_bool = torch.from_numpy(std_numpy).bool()
    custom_bool = torch.from_numpy(custom_numpy).bool()
    
    print(f"\nブール変換後:")
    print(f"  - 標準np.array→bool ユニーク値: {torch.unique(std_bool).tolist()}")
    print(f"  - pil_to_numpy→bool ユニーク値: {torch.unique(custom_bool).tolist()}")
    
    bool_diff = std_bool != custom_bool
    bool_diff_count = torch.sum(bool_diff).item()
    bool_diff_percent = bool_diff_count / std_bool.numel() * 100
    print(f"  - ブール変換後の差異: {bool_diff_count} ({bool_diff_percent:.2f}%)")
    
    # 閾値テスト
    print(f"\n各閾値での二値化結果の差異:")
    for threshold in [0, 1, 127, 128, 254, 255]:
        std_thresh = (std_numpy > threshold).astype(np.uint8)
        custom_thresh = (custom_numpy > threshold).astype(np.uint8)
        thresh_diff = std_thresh != custom_thresh
        thresh_diff_count = np.sum(thresh_diff)
        thresh_diff_percent = thresh_diff_count / std_numpy.size * 100
        print(f"  - 閾値 {threshold}: 差異 {thresh_diff_count} ({thresh_diff_percent:.2f}%)")
    
    # 可視化
    if diff_count > 0:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(std_numpy, cmap='gray')
        plt.title('標準 np.array')
        plt.colorbar()
        
        plt.subplot(2, 3, 2)
        plt.imshow(custom_numpy, cmap='gray')
        plt.title('pil_to_numpy')
        plt.colorbar()
        
        plt.subplot(2, 3, 3)
        plt.imshow(diff, cmap='hot')
        plt.title('値の差異')
        plt.colorbar()
        
        plt.subplot(2, 3, 4)
        plt.imshow(std_bool.numpy(), cmap='gray')
        plt.title('標準→bool')
        
        plt.subplot(2, 3, 5)
        plt.imshow(custom_bool.numpy(), cmap='gray')
        plt.title('pil_to_numpy→bool')
        
        plt.subplot(2, 3, 6)
        plt.imshow(bool_diff.numpy(), cmap='hot')
        plt.title('bool変換後の差異')
        
        plt.tight_layout()
        
        # 保存ディレクトリの作成
        output_dir = "mask_debug_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "mask_conversion_analysis.png")
        plt.savefig(output_path)
        print(f"\n可視化画像を保存しました: {output_path}")
        
        # 元の画像と標準/カスタム変換結果を保存
        std_plot_path = os.path.join(output_dir, "std_numpy.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(std_numpy, cmap='gray')
        plt.colorbar()
        plt.title('標準 np.array変換')
        plt.savefig(std_plot_path)
        
        custom_plot_path = os.path.join(output_dir, "custom_numpy.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(custom_numpy, cmap='gray')
        plt.colorbar()
        plt.title('pil_to_numpy変換')
        plt.savefig(custom_plot_path)
        
        print(f"詳細画像も保存しました: {std_plot_path}, {custom_plot_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python test_mask_conversion.py [マスク画像のパス]")
        sys.exit(1)
    
    mask_file_path = sys.argv[1]
    if not os.path.exists(mask_file_path):
        print(f"エラー: ファイル '{mask_file_path}' が見つかりません")
        sys.exit(1)
    
    test_mask_conversion(mask_file_path)