from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"  # 指定使用空闲的GPU 1和3
import matplotlib
matplotlib.use('Agg')
from torch.cuda.amp import autocast, GradScaler 
import re
import chardet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import hashlib  
from collections import defaultdict 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.cuda.amp import autocast, GradScaler
import sys
sys.path.append('./')  
from dataEnrich import LineBasedAugmentor  
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import torch.nn.functional as F

RobertaTokenizer.from_pretrained("roberta-base", cache_dir="./hf_cache/roberta-base")
RobertaModel.from_pretrained("roberta-base", cache_dir="./hf_cache/roberta-base")

print(f"PyTorch版本: {torch.__version__}")  # 需要≥1.10
assert torch.__version__ >= "1.10.0", "请升级PyTorch版本"

# 辅助函数定义
def load_dataset(file_path):
    """加载数据集并自动检测编码"""
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read()
            encoding = chardet.detect(rawdata)['encoding']

        texts, labels = [], []
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for line in f:
                line = line.strip()
                if len(line) < 2:
                    continue
                if line[0] in ('Y', 'N') and line[1] == ' ':
                    texts.append(line[2:].strip())
                    labels.append(1 if line[0] == 'Y' else 0)

        print(f"成功加载 {len(texts)} 个样本")
        return texts, np.array(labels)
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        return [], []

def check_data_leak(train_texts, val_texts):
    """检查训练集和验证集之间的数据泄漏"""
    train_hashes = {hashlib.md5(t.encode()).hexdigest() for t in train_texts}
    val_hashes = {hashlib.md5(t.encode()).hexdigest() for t in val_texts}
    overlap = train_hashes & val_hashes
    print(f"数据泄漏检查: 发现{len(overlap)}个重复样本")
    return list(overlap)

# 数据预处理类
class CodePreprocessor:
    def __init__(self, model_type='word2vec', cache_dir="./hf_models"):
        self.model_type = model_type
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        if model_type == 'bert':
            # # 增加重试机制
            # self._init_bert_with_retry()
            self._init_bert_local()  # 修改初始化方法名

    def _init_bert_with_retry(self, max_retries=3):
        """带重试的 RoBERTa 初始化（自动联网下载）"""
        for i in range(max_retries):
            try:
                self.tokenizer = RobertaTokenizer.from_pretrained(
                    "roberta-base",
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
                self.bert = RobertaModel.from_pretrained(
                    "roberta-base",
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
                print("成功加载 RoBERTa tokenizer 和 model（网络模式）")
                return
            except Exception as e:
                print(f"第{i + 1}次初始化 RoBERTa 失败: {str(e)}")
                if i == max_retries - 1:
                    raise RuntimeError("无法联网加载 RoBERTa 模型，请检查网络或手动下载")

    def _init_bert_local(self):
        """强制从本地加载 RoBERTa"""
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                "./hf_cache/roberta-base",
                local_files_only=True,
                use_fast=True
            )
            self.bert = RobertaModel.from_pretrained(
                "./hf_cache/roberta-base",
                local_files_only=True
            )
            print("成功从本地加载 RoBERTa tokenizer 和 model")
            return
        except Exception as e:
            print(f"本地加载失败: {str(e)}")
            print("请检查以下文件是否存在：")
            print("1. ./hf_cache/roberta-base/config.json")
            print("2. ./hf_cache/roberta-base/pytorch_model.bin")
            raise RuntimeError("模型文件不完整，请重新下载")

    def tokenize(self, code):
        """优化的代码分词方法"""
        tokens = re.findall(
            r'\$?\w+|[^\w\s]',  # 匹配变量、单词和符号
            code
        )
        return [t.lower() for t in tokens if len(t) > 1]

    def encode(self, code):
        """根据模型类型返回编码"""
        if self.model_type == 'word2vec':
            return self._word2vec_encode(code)
        elif self.model_type == 'bert':
            return self._bert_encode(code)
        else:
            raise ValueError("不支持的模型类型")

    def _word2vec_encode(self, code):
        """Word2Vec编码方法"""
        tokens = self.tokenize(code)
        if not hasattr(self, 'w2v_model'):
            raise RuntimeError("Word2Vec模型未训练")

        vectors = []
        for token in tokens:
            if token in self.w2v_model.wv:
                vectors.append(self.w2v_model.wv[token])
        return np.mean(vectors, axis=0) if vectors else np.zeros(128)

    # 修改编码方法返回attention_mask
    def _bert_encode(self, code):
        """BERT编码方法优化"""
        inputs = self.tokenizer(
            code,
            max_length=128,  # 原256
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return inputs


# 数据集类
class SecurityDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # 修复点
        self.preprocessor = preprocessor
        self.valid_indices = self._validate_samples()

    def _validate_samples(self):
        """验证样本有效性"""
        valid = []
        for i in range(len(self.texts)):
            if self.preprocessor.model_type == 'bert':
                encoded = self.preprocessor.encode(self.texts[i])
                if encoded is not None:
                    valid.append(i)
            else:
                valid.append(i)
        print(f"有效样本率: {len(valid)}/{len(self.texts)}")
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        text = self.texts[real_idx]
        label = self.labels[real_idx]
        features = self.preprocessor.encode(text)

        # 修复维度问题：从 [1, seq_len] -> [seq_len]
        if 'input_ids' in features:
            features['input_ids'] = features['input_ids'].squeeze(0)
        if 'attention_mask' in features:
            features['attention_mask'] = features['attention_mask'].squeeze(0)

        return features, label


# 简化结构 + CLS向量 + dropout + 两层分类器
class RoBERTaModel(nn.Module):
    def __init__(self, local_model_path="./hf_cache/roberta-base", dropout_rate=0.3, use_amp=True):
        super(RoBERTaModel, self).__init__()
        self.local_model_path = local_model_path
        # 只从本地加载 RoBERTa 模型
        self._init_bert_local()
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.use_amp = use_amp

    def _init_bert_local(self):
        """强制从本地加载 RoBERTa"""
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                "./hf_cache/roberta-base",
                local_files_only=True,
                use_fast=True
            )
            self.bert = RobertaModel.from_pretrained(
                "./hf_cache/roberta-base",
                local_files_only=True
            )
            print("成功从本地加载 RoBERTa tokenizer 和 model")
            return
        except Exception as e:
            print(f"本地加载失败: {str(e)}")
            print("请检查以下文件是否存在：")
            print("1. ./hf_cache/roberta-base/config.json")
            print("2. ./hf_cache/roberta-base/pytorch_model.bin")
            raise RuntimeError("模型文件不完整，请重新下载")

    def forward(self, input_ids, attention_mask):
        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
            pooled_output = self.layer_norm(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        return logits.squeeze(-1)


# 使用 Focal Loss:强化对 Hard Negative 的学习，减少 FP
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 改用 FocalLoss，cosine scheduler，accumulation=8
class RoBERTaModel_Trainer:
    def __init__(self, model, use_amp=True):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.use_amp = use_amp  # 通过参数控制是否启用AMP混合精度

        self.scaler = GradScaler(enabled=self.use_amp)
        self.criterion = FocalLoss(gamma=2, alpha=0.25)
        self.loss_history = defaultdict(list)
        self.acc_history = defaultdict(list)
        self.best_f1 = 0.0
        self.early_stop_counter = 0
        self.gradient_accumulation_steps = 8

    def train(self, train_loader, val_loader, epochs=15):
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total_samples = 0, 0, 0

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                labels = labels.to(self.device, non_blocking=True).float()
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size * self.gradient_accumulation_steps
                preds = (torch.sigmoid(outputs) >= 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total_samples += batch_size

                progress_bar.set_postfix({
                    'Loss': f"{total_loss / total_samples:.4f}",
                    'Acc': f"{correct / total_samples:.2%}"
                })

            val_loss, val_acc, val_f1 = self.evaluate(val_loader)

            self.loss_history['train'].append(total_loss / total_samples)
            self.acc_history['train'].append(correct / total_samples)
            self.loss_history['val'].append(val_loss)
            self.acc_history['val'].append(val_acc)

            # 早停判断用f1分数
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.early_stop_counter = 0
                self._save_checkpoint(optimizer, epoch)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= 3:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {self.loss_history['train'][-1]:.4f} | Acc: {self.acc_history['train'][-1]:.2%}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f}")
            print("-" * 60)

    def evaluate(self, loader):
        from sklearn.metrics import f1_score

        self.model.eval()
        total_loss, correct, total_samples = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                labels = labels.to(self.device, non_blocking=True).float()
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)

                preds = (torch.sigmoid(outputs) >= 0.5).long()
                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size
                correct += (preds == labels.long()).sum().item()
                total_samples += batch_size

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        f1 = f1_score(all_labels, all_preds, average='macro')
        return total_loss / total_samples, correct / total_samples, f1

    def _save_checkpoint(self, optimizer, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': {
                'train_loss': self.loss_history['train'],
                'val_f1': self.best_f1
            }
        }
        torch.save(checkpoint, 'best_bert_v2.pth')


# 性能对比可视化
def plot_comparison(base_trainer, enhanced_trainer):
    plt.figure(figsize=(12, 5))

    # 损失对比
    plt.subplot(1, 2, 1)
    plt.plot(base_trainer.loss_history['val'], label='Base Model')
    plt.plot(enhanced_trainer.loss_history['val'], label='Enhanced Model')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率对比
    plt.subplot(1, 2, 2)
    plt.plot(base_trainer.acc_history['val'], label='Base Model')
    plt.plot(enhanced_trainer.acc_history['val'], label='Enhanced Model')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png')  # 直接保存
    plt.close('all')  # 关闭图形释放内存

# === 新增函数 ===
def save_dataset(texts, labels, file_path):
    """保存数据集到文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for text, label in zip(texts, labels):
                prefix = 'Y ' if label == 1 else 'N '
                f.write(f"{prefix}{text}\n")
        print(f"成功保存数据集到 {file_path} ({len(texts)} 条样本)")
    except Exception as e:
        print(f"保存失败: {str(e)}")
        raise


def augment_dataset(texts, labels, output_path):
    """执行数据增强"""
    # 临时保存原始数据
    temp_path = "temp.txt"
    save_dataset(texts, labels, temp_path)

    # 执行增强
    augmentor = LineBasedAugmentor(
        input_path=temp_path,
        output_path=output_path,
        augment_factor=10
    )
    augmentor.process()

    # 加载增强数据
    aug_texts, aug_labels = [], []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) >= 2 and line[1] == ' ':
                aug_texts.append(line[2:].strip())
                aug_labels.append(1 if line[0] == 'Y' else 0)

    # 合并数据
    combined_texts = texts + aug_texts
    combined_labels = np.concatenate([labels, aug_labels])
    os.remove(temp_path)
    return combined_texts, combined_labels


# 主执行流程
if __name__ == "__main__":
    # 参数配置
    DATA_PATH = "/home/louchenhua/pyProject/SSRFinder/file/datasets/total/totalDataset_processed"
    CACHE_DIR = "./hf_cache"  # 模型缓存目录
    SAVE_DIR = "/home/louchenhua/pyProject/SSRFinder/file/datasets/total/bert"  # 保存路径
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_PARAMS = {
        'base': {'batch_size': 64, 'hidden_size': 256},
        'bert': {'batch_size': 8, 'max_length': 128}
    }

    # 预加载BERT模型（带重试）
    print("正在初始化BERT模型...")
    try:
        CodePreprocessor(model_type='bert', cache_dir=CACHE_DIR)
        print("BERT模型初始化成功")
    except Exception as e:
        print(f"BERT模型初始化失败: {str(e)}")
        print("请尝试以下解决方案：")
        print("1. 检查网络连接是否正常")
        print("2. 手动下载模型：")
        print("   git lfs install")
        print("   git clone https://hf-mirror.com/microsoft/codebert-base ./hf_cache")
        exit()

    # 加载原始数据集
    print("\n=== 数据加载 ===")
    texts, labels = load_dataset(DATA_PATH)

    # 划分数据集（训练/验证/测试）
    print("\n=== 数据划分 ===")
    # 首次划分：分离测试集
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    # 二次划分：训练集/验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.25, stratify=train_val_labels, random_state=42
    )

    # 执行数据增强（仅训练集）
    print("\n=== 数据增强 ===")
    aug_train_path = os.path.join(SAVE_DIR, "augmented_train.txt")
    train_texts, train_labels = augment_dataset(train_texts, train_labels, aug_train_path)

    # 数据泄漏检查
    print("\n=== 泄漏检查 ===")
    train_hashes = {hashlib.sha256(t.encode()).hexdigest() for t in train_texts}
    val_hashes = {hashlib.sha256(t.encode()).hexdigest() for t in val_texts}
    duplicates = train_hashes & val_hashes
    if duplicates:
        print(f"发现{len(duplicates)}个重复样本，已从验证集移除")
        val_texts = [t for t, h in zip(val_texts, val_hashes) if h not in duplicates]
        val_labels = [l for l, h in zip(val_labels, val_hashes) if h not in duplicates]

    # 保存所有数据集
    print("\n=== 保存数据集 ===")
    save_dataset(train_texts, train_labels, os.path.join(SAVE_DIR, "train.txt"))
    save_dataset(val_texts, val_labels, os.path.join(SAVE_DIR, "val.txt"))
    save_dataset(test_texts, test_labels, os.path.join(SAVE_DIR, "test.txt"))

    # 初始化预处理器
    base_preprocessor = CodePreprocessor(model_type='word2vec')
    print("\n训练Word2Vec模型...")
    tokenized_texts = [base_preprocessor.tokenize(text) for text in train_texts]
    base_preprocessor.w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=128,
        window=5,
        min_count=1,
        workers=4
    )

    # 数据集
    enhanced_preprocessor = CodePreprocessor(model_type='bert', cache_dir=CACHE_DIR)
    enhanced_train_set = SecurityDataset(train_texts, train_labels, enhanced_preprocessor)
    enhanced_val_set = SecurityDataset(val_texts, val_labels, enhanced_preprocessor)

    # 创建数据加载器
    # 数据加载器优化
    enhanced_train_loader = DataLoader(
        enhanced_train_set,
        batch_size=TRAIN_PARAMS['bert']['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    enhanced_val_loader = DataLoader(enhanced_val_set, batch_size=16)

    # 初始化模型
    print("\n初始化模型...")
    roberta_model = RoBERTaModel()
    roberta_model.bert.config.gradient_checkpointing = True  # 确保梯度检查点启用


    # 训练模型
    # 训练流程增加早停监控
    print("\n训练增强模型：")
    # RoBERTa模型
    enhanced_trainer = RoBERTaModel_Trainer(roberta_model, use_amp=True)
    enhanced_trainer.train(enhanced_train_loader, enhanced_val_loader, epochs=15)

    # RoBERTa模型性能对比和报告
    print("\n=== 最终性能报告 ===")
    print(f"[增强模型] 最佳验证准确率: {max(enhanced_trainer.acc_history['val']):.4f}")

    # RobertaModel
    # === 保存预测为Y的样本 ===
    def save_predicted_y_samples(texts, predictions, output_path):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            count = 0
            with open(output_path, 'w', encoding='utf-8') as f:
                for text, pred in zip(texts, predictions):
                    if pred == 1:
                        f.write('Y ' + text.strip().replace('\n', ' ') + '\n')
                        count += 1
            print(f"成功保存预测为Y的样本 ({count} 条) 至: {output_path}")
        except Exception as e:
            print(f"保存预测样本失败: {str(e)}")

    # 构建测试集
    print("\n=== 在测试集上评估模型性能 ===")
    test_set = SecurityDataset(test_texts, test_labels, enhanced_preprocessor)
    test_loader = DataLoader(test_set, batch_size=16)

    roberta_model.eval()
    y_true, y_pred = [], []
    all_texts = []  # 同步收集当前batch对应的text（用于保存Y样本）

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            input_ids = inputs['input_ids'].to(enhanced_trainer.device)
            attention_mask = inputs['attention_mask'].to(enhanced_trainer.device)
            outputs = roberta_model(input_ids, attention_mask)

            batch_preds = (torch.sigmoid(outputs) >= 0.5).long().cpu().tolist()
            y_pred.extend(batch_preds)
            y_true.extend(labels.tolist())

            # 同步收集该 batch 对应的原始文本
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + len(batch_preds)
            all_texts.extend(test_texts[start_idx:end_idx])

    # 打印测试集分类报告
    print("\n=== 增强模型测试集分类性能报告 ===")
    print(classification_report(y_true, y_pred, target_names=['N', 'Y'], digits=4))

    # 保存预测为 Y 的样本
    print("\n=== 保存预测为Y的样本 ===")
    save_predicted_y_samples(
        all_texts,
        y_pred,
        "/home/louchenhua/pyProject/SSRFinder/file/output/classification_results/predicted_Y_samples.txt"
    )
