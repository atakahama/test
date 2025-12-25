#!/usr/bin/env python3
"""
gsplat インポート診断スクリプト
エラーの原因を特定します
"""

import sys
import os

print("=" * 60)
print("gsplat Import Diagnostics")
print("=" * 60)

# Step 1: gsplatがインストールされているか
print("\n[Step 1] Checking gsplat installation...")
try:
    import gsplat
    print(f"✅ gsplat imported successfully")
    print(f"   Version: {gsplat.__version__ if hasattr(gsplat, '__version__') else 'Unknown'}")
    print(f"   Location: {gsplat.__file__}")
except ImportError as e:
    print(f"❌ Failed to import gsplat: {e}")
    sys.exit(1)

# Step 2: gsplatの構造を確認
print("\n[Step 2] Checking gsplat structure...")
print(f"   Available attributes in gsplat:")
attrs = [attr for attr in dir(gsplat) if not attr.startswith('_')]
for attr in attrs[:10]:  # 最初の10個を表示
    print(f"      - {attr}")
if len(attrs) > 10:
    print(f"      ... and {len(attrs) - 10} more")

# Step 3: _Cモジュールの確認
print("\n[Step 3] Checking for _C module...")

# 試行1: gsplat._C
try:
    from gsplat import _C
    print(f"✅ gsplat._C imported successfully")
    print(f"   Type: {type(_C)}")
    print(f"   Functions: {[f for f in dir(_C) if not f.startswith('_')][:5]}")
except ImportError as e:
    print(f"❌ Cannot import _C from gsplat: {e}")
    
    # 試行2: gsplat.cuda._C
    try:
        from gsplat.cuda import _C
        print(f"✅ gsplat.cuda._C imported successfully (alternative location)")
        print(f"   Type: {type(_C)}")
    except ImportError as e2:
        print(f"❌ Cannot import _C from gsplat.cuda: {e2}")
        
        # 試行3: gsplat.csrc
        try:
            from gsplat import csrc
            print(f"✅ gsplat.csrc imported successfully (alternative name)")
            print(f"   Type: {type(csrc)}")
        except ImportError as e3:
            print(f"❌ Cannot import csrc from gsplat: {e3}")

# Step 4: JITコンパイルの確認
print("\n[Step 4] Checking JIT compilation...")
cache_dir = os.path.expanduser("~/.cache/torch_extensions")
if os.path.exists(cache_dir):
    print(f"✅ Cache directory exists: {cache_dir}")
    extensions = os.listdir(cache_dir)
    gsplat_exts = [e for e in extensions if 'gsplat' in e.lower()]
    if gsplat_exts:
        print(f"   Found gsplat extensions: {gsplat_exts}")
    else:
        print(f"   ⚠️  No gsplat extensions found (JIT compilation may not have run)")
else:
    print(f"❌ Cache directory does not exist: {cache_dir}")

# Step 5: CUDAの確認
print("\n[Step 5] Checking CUDA availability...")
try:
    import torch
    print(f"✅ PyTorch imported: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Device count: {torch.cuda.device_count()}")
except ImportError:
    print(f"❌ PyTorch not installed")

# Step 6: _backend.pyの確認
print("\n[Step 6] Checking _backend.py...")
try:
    from gsplat.cuda import _backend
    print(f"✅ gsplat.cuda._backend imported")
    print(f"   Location: {_backend.__file__}")
    
    # _backendの内容を確認
    if hasattr(_backend, '_C'):
        print(f"   ✅ _backend._C exists")
        print(f"      Type: {type(_backend._C)}")
    else:
        print(f"   ❌ _backend._C does not exist")
        
except ImportError as e:
    print(f"❌ Cannot import _backend: {e}")

# Step 7: 直接コンパイルを試行
print("\n[Step 7] Attempting to trigger JIT compilation...")
try:
    # rasterizationをimportするとJITが走る可能性
    from gsplat import rasterization
    print(f"✅ rasterization imported (JIT may have been triggered)")
except Exception as e:
    print(f"❌ Failed to import rasterization: {e}")

# 再度_Cを確認
print("\n[Step 8] Re-checking _C after potential JIT compilation...")
try:
    from gsplat import _C
    print(f"✅ _C now available!")
    funcs = [f for f in dir(_C) if not f.startswith('_')]
    print(f"   Available functions ({len(funcs)} total):")
    for f in funcs[:10]:
        print(f"      - {f}")
    if len(funcs) > 10:
        print(f"      ... and {len(funcs) - 10} more")
        
    # projection_ut_3dgs_fused_bwdがあるか確認
    if 'projection_ut_3dgs_fused_bwd' in funcs:
        print(f"\n   ✅✅✅ projection_ut_3dgs_fused_bwd FOUND!")
    else:
        print(f"\n   ❌ projection_ut_3dgs_fused_bwd NOT FOUND")
        print(f"   UT-related functions:")
        ut_funcs = [f for f in funcs if 'ut' in f.lower() or 'fisheye' in f.lower()]
        for f in ut_funcs:
            print(f"      - {f}")
            
except ImportError as e:
    print(f"❌ Still cannot import _C: {e}")

print("\n" + "=" * 60)
print("Diagnostics Complete")
print("=" * 60)

# 推奨アクション
print("\n[Recommended Actions]")
print("1. If _C is not found at all:")
print("   → Run: python -c 'from gsplat import rasterization'")
print("   → This should trigger JIT compilation")
print("")
print("2. If JIT compilation fails:")
print("   → Check: pip install -e . output for errors")
print("   → Ensure CUDA is properly installed")
print("")
print("3. If _C exists but projection_ut_3dgs_fused_bwd is missing:")
print("   → Check: ext.cpp has the binding")
print("   → Rebuild: pip uninstall gsplat -y && BUILD_NO_CUDA=1 pip install -e .")
