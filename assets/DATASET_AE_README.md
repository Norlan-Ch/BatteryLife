# Dataset_AE 使用文档

## 📋 概述

`Dataset_AE` 是为 Latent Diffusion Model 的 AutoEncoder 设计的数据集类，专门用于提取电池的完整 discharge capacity 序列（SOH 序列）。

### 🎯 设计特点

1. **继承复用**：继承自 `Dataset_original`，复用了数据加载逻辑
2. **代码精简**：通过继承减少了约 200+ 行重复代码
3. **专注功能**：只提取 SOH 序列，不包含其他冗余数据
4. **灵活配置**：支持可选的 padding 和两种 padding 模式

---

## 🔧 API 接口

### 初始化参数

```python
Dataset_AE(
    args,              # 包含数据集配置的参数对象
    flag='train',      # 数据集类型: 'train', 'val', 'test'
    soh_len=None,      # 目标序列长度，None 表示不 padding
    padding_mode='zero' # padding 模式: 'zero' 或 'last'
)
```

### 参数说明

- **args**: 需要包含以下属性
  - `root_path`: 数据集根目录路径
  - `dataset`: 数据集名称（如 'MATR', 'HUST', 'CALCE' 等）

- **flag**: 数据集划分
  - `'train'`: 训练集
  - `'val'`: 验证集
  - `'test'`: 测试集

- **soh_len**: 序列长度控制
  - `None`: 保持原始长度（不 padding/不截断）
  - `int`: 统一到指定长度（短的 padding，长的截断）

- **padding_mode**: Padding 方式
  - `'zero'`: 使用 0 填充
  - `'last'`: 使用序列最后一个值填充

---

## 📦 返回数据格式

### 单个样本（`__getitem__`）

```python
{
    'discharge_capacity_seq': Tensor,  # [soh_len] 或 [eol]，归一化的 SOH 序列
    'seq_length': int,                 # 有效序列长度（EOL）
    'eol': int                         # 寿命终点（与 seq_length 相同）
}
```

### 批次数据（使用 `collate_fn_AE`）

```python
{
    'discharge_capacity_seq': Tensor,  # [B, soh_len]，批次 SOH 序列
    'seq_length': Tensor,              # [B]，各样本的有效长度
    'eol': Tensor                      # [B]，各样本的 EOL
}
```

---

## 💡 使用示例

### ⭐ 推荐方式：使用 Data Provider（最简单）

```python
from data_provider.data_factory import data_provider_AE, data_provider_AE_evaluate

# 训练和验证（会自动 shuffle 和 drop_last）
train_set, train_loader = data_provider_AE(args, 'train', soh_len=2000, padding_mode='zero')
val_set, val_loader = data_provider_AE(args, 'val', soh_len=2000, padding_mode='zero')

# 测试评估（不会 shuffle 和 drop_last）
test_set, test_loader = data_provider_AE_evaluate(args, 'test', soh_len=2000, padding_mode='zero')

# 直接使用 DataLoader，无需手动配置
for batch in train_loader:
    seqs = batch['discharge_capacity_seq']  # [B, 2000]
    seq_lengths = batch['seq_length']       # [B]
    # 训练代码...
```

### 1. 不使用 Padding（变长序列）

```python
from data_provider.data_loader import Dataset_AE

# 创建数据集
dataset = Dataset_AE(
    args=args,
    flag='train',
    soh_len=None,      # 不 padding
    padding_mode='zero'
)

# 获取样本
sample = dataset[0]
print(sample['discharge_capacity_seq'].shape)  # [eol]
print(sample['seq_length'])                     # 实际长度
```

### 2. 使用零填充（固定长度）

```python
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_AE, collate_fn_AE

# 创建数据集
dataset = Dataset_AE(
    args=args,
    flag='train',
    soh_len=2000,       # 统一到 2000 个循环
    padding_mode='zero'  # 零填充
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn_AE
)

# 迭代批次
for batch in dataloader:
    seqs = batch['discharge_capacity_seq']  # [32, 2000]
    seq_lengths = batch['seq_length']       # [32]
    # 训练代码...
```

### 3. 使用最后值填充

```python
dataset = Dataset_AE(
    args=args,
    flag='train',
    soh_len=2000,
    padding_mode='last'  # 使用最后值填充
)
```

### 4. 完整训练循环

```python
# 创建所有数据集
train_dataset = Dataset_AE(args, 'train', soh_len=2000, padding_mode='zero')
val_dataset = Dataset_AE(args, 'val', soh_len=2000, padding_mode='zero')
test_dataset = Dataset_AE(args, 'test', soh_len=2000, padding_mode='zero')

# 创建 DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_AE)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_AE)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        discharge_seqs = batch['discharge_capacity_seq']  # [B, soh_len]
        seq_lengths = batch['seq_length']                 # [B]
        
        # 前向传播
        reconstructed = autoencoder(discharge_seqs)
        
        # 计算损失（需要 mask 掉 padding 部分）
        mask = torch.arange(soh_len).expand(B, -1) < seq_lengths.unsqueeze(1)
        loss = criterion(reconstructed * mask, discharge_seqs * mask)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🗂️ 支持的数据集

完全兼容 `Dataset_original` 支持的所有数据集：

- **基础数据集**: CALCE, HNEI, HUST, MATR, RWTH, SNL
- **MICH 系列**: MICH, MICH_EXP
- **其他数据集**: UL_PUR, Tongji, Stanford, ISU_ILCC, XJTU
- **特殊电池**: ZN-coin, CALB, NAion（及其变体）
- **混合数据集**: MIX_large

---

## ⚠️ 注意事项

### 1. 数据归一化
- SOH 值已归一化：`discharge_capacity / nominal_capacity`
- 取值范围通常在 0-1 之间

### 2. Padding 处理
- **训练时**：必须在损失计算中使用 `seq_length` 创建 mask
- **推理时**：注意只使用有效部分的输出

### 3. 缺失数据
- 自动跳过未达到 EOL 的电池
- 自动处理周期内缺失数据（使用前一个值或默认值）

### 4. 内存考虑
- 无 padding 时内存效率更高，但无法批处理
- 使用 padding 时选择合适的 `soh_len` 以平衡内存和性能

---

## 🔍 与 Dataset_original 的关系

### 复用的方法
- ✅ `read_cell_data_according_to_prefix()`: 读取电池数据和 EOL
- ✅ `merge_MICH()`: 合并 MICH 数据集（通过继承自动获得）

### 新增/重写的方法
- 🆕 `extract_discharge_capacity_sequence()`: 提取 SOH 序列
- 🆕 `read_data_AE()`: AE 专用数据读取逻辑
- 🔄 `__getitem__()`: 返回 AE 专用数据格式
- 🔄 `__len__()`: 返回 SOH 序列数量

### 代码复用收益
- 减少重复代码：~200+ 行
- 维护成本降低：数据加载逻辑统一
- 兼容性保证：与原始数据集保持一致

---

## 📚 示例代码

完整的使用示例请参考：`example_dataset_ae_usage.py`

该文件包含以下示例：
1. 不使用 padding 的示例
2. 使用零填充的示例
3. 使用最后值填充的示例
4. 完整训练循环结构
5. 加载多个数据集的示例

---

## 🐛 常见问题

### Q1: 为什么返回的数据中有 NaN？
A: 检查数据预处理是否正确，或者某些电池数据本身存在问题。Dataset_AE 会在初始化时检测并抛出异常。

### Q2: 如何处理不同长度的序列？
A: 使用 `soh_len` 参数统一长度，或使用 `seq_length` 字段在损失计算时创建 mask。

### Q3: padding_mode 选择 'zero' 还是 'last'？
A: 
- `'zero'`: 更明确标识填充区域，适合大多数情况
- `'last'`: 保持序列连续性，可能对某些模型更友好

### Q4: 为什么要继承 Dataset_original？
A: 复用数据加载逻辑，减少代码重复，便于维护。

### Q5: data_provider_AE 和 data_provider_AE_evaluate 有什么区别？
A: 
- `data_provider_AE`: 根据 flag 自动配置 shuffle 和 drop_last，适合训练和验证
- `data_provider_AE_evaluate`: 固定使用 shuffle=False 和 drop_last=False，专门用于评估

---

## 🚀 Data Provider 函数

### data_provider_AE()

**用途**: 创建训练/验证数据集和 DataLoader

**参数**:
```python
data_provider_AE(
    args,              # 包含数据集配置的参数对象
    flag,              # 'train', 'val', 'test'
    soh_len=None,      # 目标序列长度
    padding_mode='zero' # padding 模式
)
```

**返回**: `(data_set, data_loader)`

**特点**:
- 训练时: `shuffle=True`, `drop_last=True`
- 验证/测试时: `shuffle=False`, `drop_last=False`
- 自动使用 `collate_fn_AE`

**示例**:
```python
from data_provider.data_factory import data_provider_AE

train_set, train_loader = data_provider_AE(args, 'train', soh_len=2000, padding_mode='zero')
val_set, val_loader = data_provider_AE(args, 'val', soh_len=2000, padding_mode='zero')

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        seqs = batch['discharge_capacity_seq']
        seq_lengths = batch['seq_length']
        # 训练代码...
```

### data_provider_AE_evaluate()

**用途**: 创建评估数据集和 DataLoader（专用于模型评估）

**参数**:
```python
data_provider_AE_evaluate(
    args,              # 包含数据集配置的参数对象
    flag,              # 'train', 'val', 'test'
    soh_len=None,      # 目标序列长度
    padding_mode='zero' # padding 模式
)
```

**返回**: `(data_set, data_loader)`

**特点**:
- 固定使用 `shuffle=False`, `drop_last=False`
- 确保所有样本都被评估到
- 自动使用 `collate_fn_AE`

**示例**:
```python
from data_provider.data_factory import data_provider_AE_evaluate

test_set, test_loader = data_provider_AE_evaluate(args, 'test', soh_len=2000, padding_mode='zero')

# 评估循环
model.eval()
with torch.no_grad():
    for batch in test_loader:
        seqs = batch['discharge_capacity_seq']
        seq_lengths = batch['seq_length']
        eols = batch['eol']
        
        # 前向传播
        reconstructed = model(seqs)
        
        # 计算指标...
```

### 与直接使用 Dataset_AE 的对比

| 方式 | 代码量 | 灵活性 | 推荐场景 |
|------|--------|--------|----------|
| data_provider_AE | 少 | 中 | ✅ 标准训练/评估 |
| Dataset_AE + DataLoader | 多 | 高 | 自定义配置需求 |

**使用 data_provider 的优势**:
1. ✅ 一行代码获取 dataset 和 dataloader
2. ✅ 自动配置合适的参数（shuffle, drop_last 等）
3. ✅ 自动使用正确的 collate_fn
4. ✅ 代码更简洁，减少出错

---

## � 示例代码

完整的使用示例请参考：
- **`example_dataset_ae_usage.py`**: Dataset_AE 类的详细使用示例
- **`example_ae_data_provider_usage.py`**: Data Provider 函数的完整示例（推荐）

`example_ae_data_provider_usage.py` 包含以下示例：
1. 基本使用方法
2. 评估模式使用
3. 不同 padding 模式对比
4. 无 padding 的使用
5. 完整的 AE 训练结构（含代码）
6. 多数据集加载示例

---

## �📝 更新日志

### v1.2 (2025-11-04)
- 🎉 新增 `data_provider_AE` 和 `data_provider_AE_evaluate` 函数
- 📝 添加 Data Provider 使用文档和示例
- ✨ 提供更简洁的数据加载方式

### v1.1 (2025-11-04)
- ✨ 重构为继承 `Dataset_original`
- 🔥 减少约 200+ 行重复代码
- ✅ 保持所有原有功能

### v1.0 (2025-11-04)
- 🎉 初始版本
- ✅ 支持完整 SOH 序列提取
- ✅ 支持可选 padding 和两种 padding 模式
