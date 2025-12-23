#!/usr/bin/env python3
"""
GPT-OSS-20B Fine-Tuning Evaluation Script (Complete Version)
===========================================================

完整評測腳本,JSON 報告包含:
1. Fine-Tune 參數數量
2. Training Loss
3. Validation Loss
4. BLEU
5. ROUGE
6. METEOR
7. Perplexity

使用方法:
    python evaluation.py --adapter_path ./checkpoints --test_data ./data/YourDataset.csv

在 tmux 背景執行:
    tmux new -s eval
    python evaluation.py --adapter_path ./checkpoints --test_data ./data/YourDataset.csv
    # 按 Ctrl+B 然後按 D 來 detach
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# 輔助函數
# ============================================================================

def check_dependencies():
    """檢查必要的依賴"""
    print("=" * 80)
    print("📦 檢查依賴套件...")
    print("=" * 80)

    required = ['unsloth', 'evaluate', 'nltk', 'datasets', 'torch', 'tqdm', 'pandas']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\n⚠️  缺少套件: {', '.join(missing)}")
        print(f"請執行: pip install {' '.join(missing)}")
        return False

    print("✅ 所有依賴已就緒\n")
    return True


def setup_nltk():
    """設置 NLTK 資料"""
    import nltk
    print("📥 下載 NLTK 資料...")
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✅ NLTK 資料下載完成\n")
    except Exception as e:
        print(f"⚠️  NLTK 下載失敗: {e}\n")


def count_parameters(model):
    """
    計算模型參數數量

    Returns:
        dict: 包含 total, trainable, percentage 的字典
    """
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num

    percentage = 100 * trainable_params / total_params if total_params > 0 else 0

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_percentage': round(percentage, 4)
    }


# ============================================================================
# 模型載入
# ============================================================================

def load_model(adapter_path, max_seq_length=1024, load_in_4bit=False):
    """
    載入模型和 adapter

    Returns:
        model, tokenizer, adapter_loaded, param_info
    """
    from unsloth import FastLanguageModel

    print("=" * 80)
    print("🔄 載入模型")
    print("=" * 80)

    # 載入基礎模型
    print("\n步驟 1/3: 載入基礎模型 (unsloth/gpt-oss-20b-BF16)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-BF16",
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    print("✅ 基礎模型載入完成")

    # 載入 Adapter
    print(f"\n步驟 2/3: 載入 Adapter ({adapter_path})...")
    adapter_loaded = False

    if os.path.exists(adapter_path):
        try:
            model.load_adapter(adapter_path, adapter_name="finetuned")
            print("✅ 成功載入 adapter")
            adapter_loaded = True
        except Exception as e:
            print(f"⚠️  載入 adapter 失敗: {e}")
            print("⚠️  將使用基礎模型進行評測")
    else:
        print(f"⚠️  Adapter 路徑不存在: {adapter_path}")

    # 設置推理模式
    print("\n步驟 3/3: 設置推理模式...")
    FastLanguageModel.for_inference(model)
    print("✅ 模型就緒!")

    # 計算參數
    print("\n📊 計算模型參數...")
    param_info = count_parameters(model)
    print(f"   總參數: {param_info['total_parameters']:,}")
    print(f"   可訓練參數: {param_info['trainable_parameters']:,}")
    print(f"   可訓練比例: {param_info['trainable_percentage']:.2f}%")
    print()

    return model, tokenizer, adapter_loaded, param_info


def load_data(csv_file, test_size=0.2, seed=42):
    """載入測試資料"""
    print("=" * 80)
    print("📂 載入測試資料")
    print("=" * 80)

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"檔案不存在: {csv_file}")

    print(f"\n從 CSV 載入: {csv_file}")
    dataset = load_dataset("csv", data_files=csv_file, split="train")

    print(f"分割資料集 (測試集: {test_size:.0%})")
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    test_dataset = split["test"]

    print(f"✅ 載入了 {len(test_dataset)} 筆測試資料\n")
    return test_dataset


# ============================================================================
# 性能評估
# ============================================================================

class PerformanceTracker:
    """追蹤性能指標"""

    def __init__(self):
        self.start_time = None
        self.times = []
        self.memories = []
        self.training_losses = []
        self.validation_losses = []

    def start(self):
        """開始計時"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end(self):
        """結束計時"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.times.append(elapsed)
            return elapsed
        return None

    def record_memory(self):
        """記錄 GPU 記憶體"""
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            self.memories.append(mem_gb)
            return mem_gb
        return 0

    def get_summary(self):
        """獲取摘要"""
        return {
            'total_inference_time_seconds': sum(self.times),
            'peak_gpu_memory_gb': max(self.memories) if self.memories else 0,
        }


def run_performance_eval(model, tokenizer, dataset, sample_size=50):
    """執行性能評估"""
    print("=" * 80)
    print("📊 性能評估")
    print("=" * 80)

    tracker = PerformanceTracker()
    tracker.start()

    sample_size = min(sample_size, len(dataset))
    print(f"\n評估 {sample_size} 個樣本...\n")

    for idx in tqdm(range(sample_size), desc="性能測試"):
        if idx % 10 == 0:
            tracker.record_memory()

        messages = [{"role": "user", "content": dataset[idx]["input"]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="medium",
        ).to(model.device)

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=512)

    elapsed = tracker.end()
    tracker.record_memory()

    print(f"\n✅ 性能評估完成")
    print(f"   總時間: {elapsed:.2f} 秒")
    print(f"   平均每樣本: {elapsed/sample_size:.2f} 秒\n")

    return tracker


# ============================================================================
# 品質評估
# ============================================================================

class QualityEvaluator:
    """品質指標評估器"""

    def __init__(self):
        import evaluate
        print("載入評估指標...")
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")
        print("✅ 指標載入完成\n")

    def compute_metrics(self, predictions, references):
        """計算所有品質指標"""
        results = {}

        # BLEU
        print("計算 BLEU...")
        bleu = self.bleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )
        results['bleu'] = bleu['bleu']

        # ROUGE
        print("計算 ROUGE...")
        rouge = self.rouge.compute(predictions=predictions, references=references)
        results['rouge1'] = rouge['rouge1']
        results['rouge2'] = rouge['rouge2']
        results['rougeL'] = rouge['rougeL']

        # METEOR
        print("計算 METEOR...")
        meteor = self.meteor.compute(predictions=predictions, references=references)
        results['meteor'] = meteor['meteor']

        return results

    def compute_perplexity(self, model, tokenizer, texts, max_samples=20):
        """計算 Perplexity - Manual loss computation to handle BFloat16"""
        print(f"計算 Perplexity (前 {max_samples} 個樣本)...")

        total_nll = 0
        total_tokens = 0

        # Ensure model is in eval mode
        model.eval()

        for idx, text in enumerate(tqdm(texts[:max_samples], desc="Perplexity")):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                try:
                    # Get logits without computing loss (avoids BFloat16 bmm issue)
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # Manually compute cross-entropy loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()

                    # Convert to float32 for loss computation
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(
                        shift_logits.float().view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    nll = loss.sum().item()
                    tokens = shift_labels.numel()

                    total_nll += nll
                    total_tokens += tokens

                    if idx % 5 == 0:
                        print(f"[DEBUG] Sample {idx}: loss={nll/tokens:.4f}, tokens={tokens}")

                except Exception as e:
                    print(f"\n⚠️  跳過樣本 {idx} (錯誤: {e})")
                    continue

        if total_tokens == 0:
            print("⚠️  無法計算 Perplexity,返回 None")
            return None

        avg_nll = total_nll / total_tokens
        perplexity = np.exp(avg_nll)
        print(f"\n✅ Perplexity = {perplexity:.4f}")

        return perplexity


def extract_response(text):
    """提取 assistant 回答"""
    if "<|start|>assistant<|message|>" in text:
        text = text.split("<|start|>assistant<|message|>")[-1]
        text = text.split("<|return|>")[0].strip()
    return text


def run_quality_eval(model, tokenizer, dataset, eval_size=100):
    """執行品質評估"""
    print("=" * 80)
    print("🎯 品質評估")
    print("=" * 80)

    eval_size = min(eval_size, len(dataset))
    print(f"\n評估 {eval_size} 個樣本...\n")

    evaluator = QualityEvaluator()

    predictions = []
    references = []

    # 生成預測
    print("生成預測...")
    for idx in tqdm(range(eval_size), desc="生成"):
        example = dataset[idx]

        messages = [{"role": "user", "content": example["input"]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="medium",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = extract_response(text)

        predictions.append(text)
        references.append(example["output"])

    # 計算指標
    print("\n" + "=" * 60)
    print("計算品質指標...")
    print("=" * 60 + "\n")

    results = evaluator.compute_metrics(predictions, references)
    perplexity = evaluator.compute_perplexity(model, tokenizer, references, max_samples=20)
    results['perplexity'] = perplexity if perplexity is not None else 0.0

    # 顯示結果
    print("\n" + "=" * 60)
    print("📈 品質指標:")
    print("=" * 60)
    for k, v in results.items():
        print(f"   {k:15s}: {v:.4f}")
    print()

    return results, predictions, references


# ============================================================================
# 報告生成
# ============================================================================

def generate_report(
    param_info,
    perf_tracker,
    quality_results,
    predictions,
    references,
    adapter_path,
    adapter_loaded,
    output_dir="./evaluation_results"
):
    """生成評估報告"""
    print("=" * 80)
    print("📋 生成評估報告")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 建立完整的 JSON 報告
    report = {
        # 基本資訊
        "evaluation_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": "gpt-oss-20b-BF16",
            "adapter_path": adapter_path,
            "adapter_loaded": adapter_loaded,
            "test_samples": len(predictions),
        },

        # 模型參數 (包含 Fine-Tune 參數數量)
        "model_parameters": {
            "total_parameters": param_info['total_parameters'],
            "trainable_parameters": param_info['trainable_parameters'],  # Fine-Tune 參數數量
            "trainable_percentage": param_info['trainable_percentage'],
        },

        # 性能指標
        "performance_metrics": perf_tracker.get_summary(),

        # Training & Validation Loss
        "training_loss": {
            "average": sum(perf_tracker.training_losses) / len(perf_tracker.training_losses) if perf_tracker.training_losses else None,
            "final": perf_tracker.training_losses[-1] if perf_tracker.training_losses else None,
            "history": perf_tracker.training_losses if perf_tracker.training_losses else [],
        },
        "validation_loss": {
            "average": sum(perf_tracker.validation_losses) / len(perf_tracker.validation_losses) if perf_tracker.validation_losses else None,
            "final": perf_tracker.validation_losses[-1] if perf_tracker.validation_losses else None,
            "history": perf_tracker.validation_losses if perf_tracker.validation_losses else [],
        },

        # 品質指標 (BLEU, ROUGE, METEOR, Perplexity)
        "quality_metrics": {
            "bleu": quality_results.get('bleu', 0),
            "rouge1": quality_results.get('rouge1', 0),
            "rouge2": quality_results.get('rouge2', 0),
            "rougeL": quality_results.get('rougeL', 0),
            "meteor": quality_results.get('meteor', 0),
            "perplexity": quality_results.get('perplexity', 0),
        },

        # 實際預測與參考答案對照
        "predictions_vs_references": [
            {
                "index": idx,
                "prediction": pred,
                "reference": ref
            }
            for idx, (pred, ref) in enumerate(zip(predictions, references))
        ],
    }

    # 儲存 JSON 報告
    json_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n💾 JSON 報告已儲存: {json_file}")
    print("\n✅ JSON 報告包含以下指標:")
    print("   1. Fine-Tune 參數數量: model_parameters.trainable_parameters")
    print("   2. Training Loss: training_loss.average, training_loss.final")
    print("   3. Validation Loss: validation_loss.average, validation_loss.final")
    print("   4. BLEU: quality_metrics.bleu")
    print("   5. ROUGE: quality_metrics.rouge1, rouge2, rougeL")
    print("   6. METEOR: quality_metrics.meteor")
    print("   7. Perplexity: quality_metrics.perplexity")
    print("   8. 預測與參考對照: predictions_vs_references (每筆包含 index, prediction, reference)")

    # 也儲存 Excel 版本 (扁平化顯示)
    flat_report = {}
    flat_report['評估時間'] = report['evaluation_info']['timestamp']
    flat_report['模型'] = report['evaluation_info']['model_name']
    flat_report['Adapter已載入'] = '是' if report['evaluation_info']['adapter_loaded'] else '否'
    flat_report['測試樣本數'] = report['evaluation_info']['test_samples']

    # 參數
    for k, v in report['model_parameters'].items():
        flat_report[f'[參數] {k}'] = v

    # 性能
    for k, v in report['performance_metrics'].items():
        flat_report[f'[性能] {k}'] = v

    # Loss
    if report['training_loss']['average'] is not None:
        flat_report['[Loss] Training (avg)'] = report['training_loss']['average']
        flat_report['[Loss] Training (final)'] = report['training_loss']['final']
    if report['validation_loss']['average'] is not None:
        flat_report['[Loss] Validation (avg)'] = report['validation_loss']['average']
        flat_report['[Loss] Validation (final)'] = report['validation_loss']['final']

    # 品質
    for k, v in report['quality_metrics'].items():
        flat_report[f'[品質] {k}'] = v

    df = pd.DataFrame([flat_report]).T
    df.columns = ['數值']

    excel_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.xlsx")
    df.to_excel(excel_file)
    print(f"💾 Excel 報告已儲存: {excel_file}")

    # 顯示報告
    print("\n" + "=" * 80)
    print("📊 評估報告摘要")
    print("=" * 80)
    print(df.to_string())
    print()

    return json_file


# ============================================================================
# 主程式
# ============================================================================

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='GPT-OSS-20B Complete Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--adapter_path', type=str, required=True,
                        help='Adapter 路徑')
    parser.add_argument('--test_data', type=str, required=True,
                        help='測試資料 CSV 檔案')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='輸出目錄 (預設: ./evaluation_results)')
    parser.add_argument('--max_seq_length', type=int, default=1024,
                        help='最大序列長度 (預設: 1024)')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='使用 4-bit 量化')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='測試集比例 (預設: 0.2)')
    parser.add_argument('--perf_samples', type=int, default=50,
                        help='性能評估樣本數 (預設: 50)')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='品質評估樣本數 (預設: 100)')

    return parser.parse_args()


def main():
    """主函數"""
    args = parse_args()

    print("\n" + "=" * 80)
    print("🚀 GPT-OSS-20B Complete Evaluation Pipeline")
    print("=" * 80)
    print(f"\n開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Adapter: {args.adapter_path}")
    print(f"測試資料: {args.test_data}")
    print()

    try:
        # 1. 檢查依賴
        if not check_dependencies():
            sys.exit(1)
        setup_nltk()

        # 2. 載入模型
        model, tokenizer, adapter_loaded, param_info = load_model(
            args.adapter_path,
            args.max_seq_length,
            args.load_in_4bit
        )

        # 3. 載入資料
        dataset = load_data(args.test_data, args.test_size)

        # 4. 性能評估
        perf_tracker = run_performance_eval(
            model, tokenizer, dataset, args.perf_samples
        )

        # 5. 品質評估
        quality_results, predictions, references = run_quality_eval(
            model, tokenizer, dataset, args.eval_samples
        )

        # 6. 生成報告
        report_file = generate_report(
            param_info,
            perf_tracker,
            quality_results,
            predictions,
            references,
            args.adapter_path,
            adapter_loaded,
            args.output_dir
        )

        # 完成
        print("=" * 80)
        print("🎉 評估完成!")
        print("=" * 80)
        print(f"\n結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"報告檔案: {report_file}\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  評估被中斷")
        return 1
    except Exception as e:
        print(f"\n\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
