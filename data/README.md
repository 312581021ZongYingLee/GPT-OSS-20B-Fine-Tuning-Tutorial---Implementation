# 資料集說明

本目錄用於存放訓練和測試資料集。

## 資料格式要求

訓練資料必須是 **CSV 格式**，包含以下欄位：

| 欄位名稱 | 說明 | 範例 |
|---------|------|------|
| `input` | 輸入問題或提示 | "What is SEMI E88?" |
| `output` | 期望的模型輸出 | "SEMI E88 is a standard for..." |

## 檔案命名

請將您的訓練資料命名為：

```
SEMI_Fine_Tuning_Data.csv
```

並放置在本目錄下。

## 資料集大小建議

- **最小訓練集**: 建議至少 100 筆資料
- **驗證集**: 建議佔總資料的 10-20%
- **測試集**: 建議佔總資料的 10-20%

本專案的訓練腳本會自動分割資料集。

## 範例資料

本目錄包含 `SEMI_Fine_Tuning_Data.csv`，為實際訓練資料。

### 查看資料

```bash
cat data/SEMI_Fine_Tuning_Data.csv
```

或在 Python 中讀取：

```python
import pandas as pd

df = pd.read_csv('data/SEMI_Fine_Tuning_Data.csv')
print(df.head())
```

## 準備您自己的資料集

### 方法 1: 從 Excel 轉換

如果您的資料在 Excel 中：

1. 確保有 `input` 和 `output` 兩個欄位
2. 另存新檔為 CSV 格式
3. 重新命名為 `SEMI_Fine_Tuning_Data.csv`
4. 放置於本目錄

### 方法 2: 使用 Python 建立

```python
import pandas as pd

# 建立資料
data = {
    'input': [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?"
    ],
    'output': [
        "Machine learning is...",
        "Neural networks are...",
        "Deep learning is..."
    ]
}

# 儲存為 CSV
df = pd.DataFrame(data)
df.to_csv('data/SEMI_Fine_Tuning_Data.csv', index=False)
```

## 資料品質建議

1. **清理資料**: 移除重複、格式錯誤或不完整的資料
2. **平衡分布**: 確保不同類型的問題分布均勻
3. **高品質回答**: Output 欄位應包含準確、清楚的回答
4. **一致性**: 保持用語和格式的一致性

## 注意事項

✅ 如需使用自己的資料，請替換此檔案或修改腳本中的 `CSV_FILE` 路徑

🔒 大型訓練資料會被 `.gitignore` 排除，不會被提交至 Git
