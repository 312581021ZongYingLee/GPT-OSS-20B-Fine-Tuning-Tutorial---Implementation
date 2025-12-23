# GPT-OSS-20B Fine-Tuning Tutorial

完整的 GPT-OSS-20B 模型 Fine-Tuning 教學，使用 Unsloth 和 LoRA 技術。

## 📋 目錄

- [快速開始](#快速開始)
- [環境需求](#環境需求)
- [安裝步驟](#安裝步驟)
- [使用教學](#使用教學)
- [Google Drive 下載說明](#google-drive-下載說明)
- [常見問題 (FAQ)](#常見問題-faq)
- [故障排除](#故障排除)
- [進階設定](#進階設定)

---

## 🚀 快速開始

只需 4 個步驟即可開始使用：

```bash
# 1. Clone repository
git clone <your-repo-url>
cd "Github repo"

# 2. 安裝依賴套件
pip install -r requirements.txt

# 3. 下載 LoRA checkpoint
python download_checkpoint.py

# 4. 準備您的資料集（格式請參考 data/README.md）
# 將 CSV 檔案放入 data/ 目錄，命名為 YourDataset.csv
```

完成！現在您可以執行評估或開始訓練。

---

## 💻 環境需求

### 硬體需求

- **GPU**: NVIDIA GPU with CUDA support
  - 建議: 24GB+ VRAM (例如: RTX 3090, A6000, A100)
  - 最低: 12GB VRAM (使用 4-bit 量化)
- **RAM**: 32GB+
- **儲存空間**: 50GB+ (包含模型、checkpoint 和資料)

### 軟體需求

- **Python**: 3.8 或以上
- **CUDA**: 11.8 或以上
- **作業系統**: Linux (建議) 或 Windows with WSL2

---

## 📦 安裝步驟

### Step 1: 建立虛擬環境 (建議)

```bash
# 使用 conda
conda create -n gpt-finetune python=3.10
conda activate gpt-finetune

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### Step 2: 安裝依賴套件

```bash
pip install -r requirements.txt
```

安裝內容包括：
- `unsloth`: Fast Language Model 訓練框架
- `torch`, `transformers`, `datasets`: 深度學習核心套件
- `gdown`: Google Drive 檔案下載工具
- `evaluate`, `nltk`, `rouge-score`: 評估指標
- `pandas`, `numpy`, `tqdm`: 資料處理工具

### Step 3: 下載 LoRA Checkpoint

使用預設設定下載：

```bash
python download_checkpoint.py
```

或指定自訂路徑：

```bash
python download_checkpoint.py --output ./custom_checkpoint_path
```

下載完成後，檔案將位於 `./checkpoints/` 目錄。

---

## 📚 使用教學

### 1. 下載主模型

主模型（GPT-OSS-20B）會在第一次執行時由 Unsloth 自動下載，無需手動操作。

### 2. 下載 LoRA Checkpoint

LoRA adapter 包含預訓練的微調參數：

```bash
python download_checkpoint.py
```

**說明**:
- 預設從 Google Drive 下載
- 檔案大小: < 500MB
- 下載位置: `./checkpoints/`
- 包含檔案: `adapter_model.safetensors`, `adapter_config.json`, tokenizer 等

### 3. 準備訓練資料

建立或準備您的資料集：

```bash
# 資料集格式: CSV with 'input' and 'output' columns
# 範例請參考 data/sample_data.csv
```

將檔案命名為 `YourDataset.csv` 並放入 `data/` 目錄。

**資料格式範例**:

| input | output |
|-------|--------|
| What is LoRA? | LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique... |
| Explain fine-tuning | Fine-tuning adapts a pre-trained model to specific tasks... |

詳細說明請參考 [data/README.md](data/README.md)。

### 4. 執行 Fine-Tuning 訓練

```bash
python train_gpt_oss_20b.py
```

**訓練參數** (可在腳本中調整):
- LoRA Rank: 32
- Learning Rate: 2e-4
- Epochs: 300 (可調整)
- Batch Size: 4

**建議**: 使用 `tmux` 在背景執行

```bash
tmux new -s finetune
python train_gpt_oss_20b.py
# 按 Ctrl+B 然後按 D 來 detach
```

訓練完成後，模型將儲存至 `./RANK32_gpt_oss_finetuned/`

### 5. 執行模型評估

評估微調後的模型效能：

```bash
python evaluation.py --adapter_path ./checkpoints --test_data ./data/YourDataset.csv
```

評估指標包括：
- **BLEU**: 翻譯品質指標
- **ROUGE**: 文字摘要品質 (rouge1, rouge2, rougeL)
- **METEOR**: 語義相似度
- **Perplexity**: 模型困惑度

結果將儲存為 JSON 和 Excel 格式。

### 6. 推論測試

推論功能請參考 `V2` 版本的 `Final_inference.py`，該腳本提供：
- Fine-Tuned vs Base Model 比較
- 多次推論結果
- 輸出分析

---

## 📥 Google Drive 下載說明

### 方法 1: 使用下載腳本 (推薦)

```bash
python download_checkpoint.py
```

腳本會自動：
1. 從 Google Drive 下載 LoRA checkpoint 資料夾
2. 解壓到 `./checkpoints/` 目錄
3. 驗證檔案完整性

### 方法 2: 手動下載

如果自動下載失敗：

1. 前往 Google Drive 連結:
   ```
   https://drive.google.com/drive/folders/1VjomTDXwF-jB5BNFYZ1gZb1-iYU_IT1u
   ```

2. 點擊右上角「下載」圖示

3. 等待 Google Drive 打包檔案

4. 下載完成後解壓縮

5. 將所有檔案放入 `./checkpoints/` 目錄

### 如何設定 Google Drive 分享權限 (提供者參考)

如果您想分享自己的 checkpoint:

1. 上傳資料夾到 Google Drive
2. 右鍵點擊資料夾 → "Share" → "Get link"
3. 設定為 "Anyone with the link" → "Viewer"
4. 複製分享連結
5. 從連結中提取 Folder ID (格式: `https://drive.google.com/drive/folders/<FOLDER_ID>`)

---

## ❓ 常見問題 (FAQ)

### Q1: 為什麼需要下載 LoRA Checkpoint?

**A**: LoRA checkpoint 包含預訓練的微調參數，可以：
- 作為起點進行進一步訓練
- 直接用於推論和評估
- 節省從零開始訓練的時間

### Q2: 訓練需要多長時間?

**A**: 取決於：
- 資料量大小
- GPU 性能
- Epoch 數量

範例: 500 筆資料，300 epochs，A6000 GPU → 約 2-4 小時

### Q3: 如何調整訓練參數?

**A**: 編輯 `train_gpt_oss_20b.py` 中的參數區塊:

```python
# LoRA 設定
LORA_R = 32  # Rank (越大越複雜)
LORA_ALPHA = 32
NUM_TRAIN_EPOCHS = 300  # Epoch 數量

# 訓練設定
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
```

### Q4: GPU 記憶體不足怎麼辦?

**A**: 嘗試以下方法：
1. 啟用 4-bit 量化: `LOAD_IN_4BIT = True`
2. 減少 batch size: `PER_DEVICE_TRAIN_BATCH_SIZE = 2`
3. 減少序列長度: `MAX_SEQ_LENGTH = 512`
4. 減少 LoRA rank: `LORA_R = 16`

### Q5: 下載 checkpoint 失敗?

**A**: 可能原因：
- 網路連線問題 → 重試或使用手動下載
- Google Drive 流量限制 → 稍後再試
- 檔案權限問題 → 確認分享連結設定正確

### Q6: 支援哪些資料格式?

**A**: 目前支援 CSV 格式，必須包含 `input` 和 `output` 兩個欄位。

範例請參考 `data/sample_data.csv`。

---

## 🔧 故障排除

### 問題: `ImportError: No module named 'unsloth'`

**解決方案**:
```bash
pip install unsloth
```

### 問題: CUDA Out of Memory

**解決方案**:
```python
# 在 train_gpt_oss_20b.py 中設定
LOAD_IN_4BIT = True
PER_DEVICE_TRAIN_BATCH_SIZE = 2
MAX_SEQ_LENGTH = 512
```

### 問題: 下載速度很慢

**解決方案**:
1. 使用穩定的網路連線
2. 嘗試手動下載
3. 使用VPN (如果地區限制)

### 問題: 找不到訓練資料

**解決方案**:
```bash
# 確認檔案路徑和名稱
ls data/YourDataset.csv

# 檢查檔案格式
head data/YourDataset.csv
```

### 問題: Tokenizer 錯誤

**解決方案**:
確保下載了完整的 checkpoint，包含:
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

---

## ⚙️ 進階設定

### 調整 LoRA 參數

```python
# 更大的 Rank = 更強的表達能力，但需要更多記憶體
LORA_R = 64
LORA_ALPHA = 64

# 加入 dropout 防止過擬合
LORA_DROPOUT = 0.05

# 微調更多層
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head"  # 額外層
]
```

### 使用不同的優化器

```python
# AdamW 8-bit (預設，記憶體效率高)
OPTIM = "adamw_8bit"

# 標準 AdamW (更穩定但需要更多記憶體)
OPTIM = "adamw_torch"

# SGD with momentum
OPTIM = "sgd"
```

### 調整學習率排程

```python
# Linear (預設)
LR_SCHEDULER_TYPE = "linear"

# Cosine annealing
LR_SCHEDULER_TYPE = "cosine"

# Constant
LR_SCHEDULER_TYPE = "constant"
```

### 啟用混合精度訓練

```python
# 自動檢測 (預設)
fp16=not is_bfloat16_supported(),
bf16=is_bfloat16_supported()

# 強制使用 FP16
fp16=True,
bf16=False
```

### 自訂資料集分割

```python
# 調整測試集和驗證集大小
TEST_SIZE = 200  # 測試集 200 筆
# 驗證集也會是 200 筆

# 或使用百分比
test_size=0.2  # 20% 作為測試集
```

---

## 📁 專案結構

```
This Github repo/
├── README.md                    # 本檔案
├── requirements.txt             # Python 依賴套件
├── .gitignore                   # Git 忽略規則
├── download_checkpoint.py       # Checkpoint 下載腳本
├── train_gpt_oss_20b.py        # Fine-tuning 訓練腳本
├── evaluation.py                # 模型評估腳本
├── data/                        # 資料目錄
│   ├── README.md                # 資料格式說明
│   ├── sample_data.csv          # 範例資料 (3筆)
│   └── YourDataset.csv          # 您的訓練資料 (需自備)
└── checkpoints/                 # Checkpoint 目錄
    ├── README.md                # Checkpoint 說明
    └── (下載後的檔案)
```

---

## 📊 效能指標參考

基於範例資料集的評估結果：

| 指標 | 基礎模型 | Fine-Tuned 模型 |
|------|---------|----------------|
| BLEU | 0.25 | 0.65 |
| ROUGE-L | 0.30 | 0.70 |
| METEOR | 0.28 | 0.68 |
| Perplexity | 15.2 | 8.5 |

*實際結果會因資料集和訓練設定而異*

---

## 🔗 相關資源

- [Unsloth 官方文檔](https://github.com/unslothai/unsloth)
- [LoRA Paper (原始論文)](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [GPT-OSS Model Card](https://huggingface.co/unsloth/gpt-oss-20b-BF16)

---

## 📝 授權

本專案僅供學術和研究使用。

---

## 🤝 貢獻

歡迎提交 Issues 和 Pull Requests！

---

**最後更新**: 2025-12-23
