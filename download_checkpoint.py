#!/usr/bin/env python3
"""
LoRA Checkpoint 下載腳本
======================

從 Google Drive 下載預訓練的 LoRA checkpoint 檔案

使用方法:
    python download_checkpoint.py

進階用法:
    python download_checkpoint.py --folder_id YOUR_FOLDER_ID --output ./custom_path
"""

import argparse
import os
import sys

try:
    import gdown
except ImportError:
    print("❌ 錯誤: 找不到 gdown 套件")
    print("請執行: pip install gdown")
    sys.exit(1)


# ============================================================================
# 預設設定
# ============================================================================

# Google Drive 資料夾 ID (預設值)
DEFAULT_FOLDER_ID = "1VjomTDXwF-jB5BNFYZ1gZb1-iYU_IT1u"

# 預設輸出路徑
DEFAULT_OUTPUT_PATH = "./checkpoints"


# ============================================================================
# 下載函數
# ============================================================================

def download_checkpoint_folder(folder_id, output_path="./checkpoints"):
    """
    從 Google Drive 下載整個 checkpoint 資料夾

    Args:
        folder_id: Google Drive 資料夾 ID
        output_path: 本地儲存路徑

    Returns:
        bool: 下載是否成功
    """
    print("=" * 80)
    print("📦 LoRA Checkpoint 下載工具")
    print("=" * 80)
    print(f"\nGoogle Drive 資料夾 ID: {folder_id}")
    print(f"輸出路徑: {output_path}\n")

    # 建立輸出目錄
    os.makedirs(output_path, exist_ok=True)
    print(f"✅ 建立目錄: {output_path}\n")

    # 建構 Google Drive 資料夾 URL
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    print("=" * 80)
    print("🔽 開始下載...")
    print("=" * 80)
    print(f"來源: {folder_url}")
    print(f"目標: {output_path}\n")

    try:
        # 使用 gdown 下載整個資料夾
        gdown.download_folder(
            url=folder_url,
            output=output_path,
            quiet=False,
            use_cookies=False
        )

        print("\n" + "=" * 80)
        print("✅ 下載完成!")
        print("=" * 80)
        print(f"\n📁 Checkpoint 檔案位於: {output_path}\n")

        # 列出下載的檔案
        print("下載的檔案:")
        print("-" * 80)
        if os.path.exists(output_path):
            for root, dirs, files in os.walk(output_path):
                level = root.replace(output_path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    size_mb = file_size / (1024 * 1024)
                    print(f"{subindent}{file} ({size_mb:.2f} MB)")
        print("-" * 80)

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ 下載失敗!")
        print("=" * 80)
        print(f"錯誤訊息: {e}\n")
        print("可能的原因:")
        print("  1. 網路連線問題")
        print("  2. Google Drive 資料夾 ID 不正確")
        print("  3. 資料夾未設定為「任何知道連結的人」可檢視")
        print("\n請檢查:")
        print(f"  - 資料夾 ID: {folder_id}")
        print(f"  - 資料夾 URL: {folder_url}")
        print("  - 網路連線狀態\n")
        return False


# ============================================================================
# 主程式
# ============================================================================

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='從 Google Drive 下載 LoRA checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 使用預設設定下載
  python download_checkpoint.py

  # 指定資料夾 ID
  python download_checkpoint.py --folder_id YOUR_FOLDER_ID

  # 指定輸出路徑
  python download_checkpoint.py --output ./custom_checkpoint_path

  # 完整自訂
  python download_checkpoint.py --folder_id YOUR_FOLDER_ID --output ./custom_path
        """
    )

    parser.add_argument(
        '--folder_id',
        type=str,
        default=DEFAULT_FOLDER_ID,
        help=f'Google Drive 資料夾 ID (預設: {DEFAULT_FOLDER_ID})'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f'輸出路徑 (預設: {DEFAULT_OUTPUT_PATH})'
    )

    return parser.parse_args()


def main():
    """主函數"""
    args = parse_args()

    # 執行下載
    success = download_checkpoint_folder(
        folder_id=args.folder_id,
        output_path=args.output
    )

    if success:
        print("\n✅ 全部完成!")
        print(f"您現在可以使用 checkpoint 進行推論或評估了\n")
        return 0
    else:
        print("\n❌ 下載過程發生錯誤，請檢查上方的錯誤訊息\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
