# LoRA Checkpoint 說明

本目錄用於存放從 Google Drive 下載的 LoRA adapter checkpoint 檔案。

## 下載 Checkpoint

使用以下指令從 Google Drive 下載 checkpoint：

```bash
python download_checkpoint.py
```

下載完成後，本目錄將包含以下檔案：

## Checkpoint 檔案結構

標準的 LoRA adapter 包含以下檔案：

```
checkpoints/
├── adapter_config.json          # LoRA 配置檔案
├── adapter_model.safetensors    # LoRA 權重檔案
├── chat_template.jinja          # Chat template
├── special_tokens_map.json      # 特殊 token 對應
├── tokenizer.json               # Tokenizer 檔案
└── tokenizer_config.json        # Tokenizer 配置
```

### 檔案說明

| 檔案名稱 | 說明 | 必要性 |
|---------|------|--------|
| `adapter_model.safetensors` | LoRA 適配器權重 | ✅ 必要 |
| `adapter_config.json` | LoRA 配置參數 (rank, alpha 等) | ✅ 必要 |
| `tokenizer.json` | Tokenizer 配置 | ✅ 必要 |
| `tokenizer_config.json` | Tokenizer 額外配置 | ✅ 必要 |
| `special_tokens_map.json` | 特殊 token 定義 | ✅ 必要 |
| `chat_template.jinja` | 對話格式模板 | ⭕ 建議 |

## 驗證下載

### 檢查檔案是否存在

```bash
# Windows
dir checkpoints

# Linux/Mac
ls -lh checkpoints/
```

### 檢查檔案大小

LoRA adapter 通常較小，因為只儲存適配層的權重：

- `adapter_model.safetensors`: 通常 50MB - 500MB
- 其他 JSON 檔案: 通常 < 5MB

### 使用 Python 驗證

```python
import os
import json

checkpoint_dir = "./checkpoints"

# 檢查必要檔案
required_files = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json"
]

print("檢查 checkpoint 檔案...")
for file in required_files:
    file_path = os.path.join(checkpoint_dir, file)
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"✅ {file}: {size_mb:.2f} MB")
    else:
        print(f"❌ {file}: 不存在")

# 讀取 LoRA 配置
config_path = os.path.join(checkpoint_dir, "adapter_config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"\nLoRA 配置:")
    print(f"  Rank (r): {config.get('r', 'N/A')}")
    print(f"  Alpha: {config.get('lora_alpha', 'N/A')}")
    print(f"  Target modules: {config.get('target_modules', 'N/A')}")
```

## 手動下載（備用方案）

如果自動下載失敗，您可以手動下載：

### 步驟 1: 前往 Google Drive

訪問以下連結：
```
https://drive.google.com/drive/folders/1VjomTDXwF-jB5BNFYZ1gZb1-iYU_IT1u
```

### 步驟 2: 下載資料夾

1. 點擊「下載」按鈕
2. Google Drive 會將資料夾打包為 ZIP 檔案
3. 下載完成後解壓縮

### 步驟 3: 放置檔案

將解壓縮後的所有檔案放入本目錄 (`checkpoints/`)

## 常見問題

### Q: 下載很慢或失敗？

A: 可能的原因：
1. 網路連線不穩定
2. Google Drive 流量限制
3. 檔案權限設定問題

解決方案：
- 使用手動下載
- 稍後再試
- 檢查 Google Drive 分享設定

### Q: 檔案大小異常？

A: 正常的 LoRA adapter 應該：
- 比完整模型小很多
- `adapter_model.safetensors` 應該是最大的檔案
- 總大小通常在 50MB - 1GB 之間

### Q: 缺少某些檔案？

A: 最關鍵的檔案是：
- `adapter_model.safetensors` (權重)
- `adapter_config.json` (配置)

其他檔案如果缺失，可能會使用基礎模型的對應檔案。

## 使用 Checkpoint

下載完成後，您可以：

### 1. 執行評估

```bash
python evaluation.py --adapter_path ./checkpoints --test_data ./data/YourDataset.csv
```

### 2. 執行訓練

```bash
python train_gpt_oss_20b.py
```

訓練會載入此 checkpoint 作為起點（如果需要）。

## 注意事項

⚠️ **重要**:
- Checkpoint 檔案會被 `.gitignore` 排除，不會被提交至 Git
- 每次 clone repository 後需要重新下載 checkpoint
- 請勿修改 checkpoint 檔案內容
- 備份重要的 checkpoint

## 更多資訊

如需了解 LoRA 技術細節，請參考：
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
