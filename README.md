<div align="center">


# Household Electricity Consumption Forecasting

### 基于 LSTM、Transformer 与 DE-iTransformer 的家庭电力多变量时间序列预测

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-6C63FF)](#)
[![Forecast](https://img.shields.io/badge/Horizon-90%20%7C%20365%20Days-success)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ll1317/Machine-Learning?style=social)](https://github.com/ll1317/Machine-Learning/stargazers)
[![Issues](https://img.shields.io/github/issues/ll1317/Machine-Learning)](https://github.com/ll1317/Machine-Learning/issues)

基于过去 **90 天**的多变量日级数据，分别预测未来 **90 天**和 **365 天**的家庭总有功功率变化曲线。

[项目简介](#-项目简介) ·
[模型结构](#-模型结构) ·
[实验结果](#-实验结果) ·
[快速开始](#-快速开始) ·
[项目结构](#-项目结构) ·
[参考文献](#-参考文献)

</div>


## 📌 项目简介

本项目研究家庭总有功功率的多变量时间序列预测问题。模型使用过去 90 天的家庭电力、气象和日历特征，分别预测未来：

* **90 天：短期预测**
* **365 天：长期预测**

两种预测任务分别训练，长期预测模型参数不会用于短期预测。

本项目实现并比较以下三种方法：

1. **LSTM**
2. **Transformer**
3. **DE-iTransformer：分解增强倒置 Transformer**

其中，DE-iTransformer 面向家庭电力预测任务进行结构化设计：

* 将 18 种异构变量转换为变量 token；
* 在变量维度上学习电力、天气与日历信息之间的关联；
* 引入具有周周期解释的 7 天趋势—残差分解支路；
* 使用可学习门控融合非线性变量交互预测和线性趋势预测。

---

## 📊 任务定义

设第 $t$ 天的多变量观测为：

```math
\mathbf{x}_t \in \mathbb{R}^{18}
```

给定过去 90 天的输入序列：

```math
\mathbf{X}
=
[\mathbf{x}_{t-89},\ldots,\mathbf{x}_t]
\in \mathbb{R}^{90\times 18}
```

模型需要一次性预测未来 $H$ 天的总有功功率：

```math
\widehat{\mathbf{Y}}
=
[\widehat{y}_{t+1},\ldots,\widehat{y}_{t+H}],
\qquad
H\in\{90,365\}
```

滑动窗口数据形状为：

```math
\mathbf{X}\in\mathbb{R}^{N\times 90\times 18},
\qquad
\mathbf{Y}\in\mathbb{R}^{N\times H}
```

---

## 🗂️ 数据来源

### 家庭电力数据

UCI Individual Household Electric Power Consumption：

* https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

原始数据为分钟级家庭电力测量数据，时间范围为 2006 年 12 月至 2010 年 11 月。

### 气象数据

Météo-France 月度基础气候数据：

* https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles

本项目选择巴黎蒙苏里气象站对应时间范围内的月度记录。

---

## 🧹 数据预处理

### 电力数据日级聚合

| 字段                    | 日级处理方式 |
| ----------------------- | ------------ |
| `global_active_power`   | 按天求和     |
| `global_reactive_power` | 按天求和     |
| `sub_metering_1`        | 按天求和     |
| `sub_metering_2`        | 按天求和     |
| `sub_metering_3`        | 按天求和     |
| `voltage`               | 按天取平均值 |
| `global_intensity`      | 按天取平均值 |

未被三个分表覆盖的剩余用电量计算为：
```math
$$
E_{\mathrm{remainder}}
=
\frac{P_{\mathrm{global}}\times 1000}{60}
-
(E_1+E_2+E_3)
$$
```
### 气象特征

使用以下月度气象字段：

* `RR`
* `NBJRR1`
* `NBJRR5`
* `NBJRR10`
* `NBJBROU`

气象数据依据年月与每日用电记录匹配，同一月份内的每日记录共享对应月份的气象统计值。

### 日历特征

项目额外构造：

* 星期
* 是否周末
* 月份
* 星期正余弦编码
* 月份正余弦编码

### 最终输入特征

```python
FEATURE_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
]
```

---

## ✂️ 数据划分

处理后共得到 1442 天连续日级数据。

| 数据集 | 日期范围                 |  样本数 |
| ------ | ------------------------ | ------: |
| 训练集 | 2006-12-16 至 2009-11-26 | 1077 天 |
| 测试集 | 2009-11-27 至 2010-11-26 |  365 天 |

### 90 天预测任务

* 输入长度：90 天
* 输出长度：90 天
* 训练窗口数：808
* 测试输入：2009-08-29 至 2009-11-26
* 测试目标：2009-11-27 至 2010-02-24

### 365 天预测任务

* 输入长度：90 天
* 输出长度：365 天
* 训练窗口数：258
* 测试输入：2009-08-29 至 2009-11-26
* 测试目标：2009-11-27 至 2010-11-26

> 所有 `StandardScaler` 均仅使用对应任务的训练历史拟合，验证目标和测试数据不参与标准化器拟合。

---


## ⚙️ 实验设置

| 设置                    | 数值               |
| ----------------------- | ------------------ |
| 输入长度                | 90                 |
| 预测长度                | 90 / 365           |
| 输入特征数              | 18                 |
| Batch Size              | 32                 |
| 最大训练轮数            | 200                |
| Early Stopping Patience | 20                 |
| Gradient Clipping       | 1.0                |
| Optimizer               | AdamW              |
| Random Seeds            | 42、52、62、72、82 |
| Evaluation Runs         | 5                  |

模型训练使用标准化尺度上的 MSE 损失：

```math
$$
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{NH}
\sum_{i=1}^{N}
\sum_{j=1}^{H}
\left(
Y_{i,j}-\widehat{Y}_{i,j}
\right)^2
$$
```
测试阶段将预测值反标准化，并转换为每日用电量 kWh。

---

## 📏 评价指标

### 均方误差
```math
$$
\mathrm{MSE}
=
\frac{1}{H}
\sum_{j=1}^{H}
\left(
y_j-\widehat{y}_j
\right)^2
$$
```

MSE 对较大的预测误差更加敏感。

### 平均绝对误差
```math
$$
\mathrm{MAE}
=
\frac{1}{H}
\sum_{j=1}^{H}
\left|
y_j-\widehat{y}_j
\right|
$$
```

MAE 表示平均每天预测值与真实值之间的绝对偏差。

所有结果均报告 5 次独立实验的：

$$
\mathrm{mean}\pm\mathrm{std}
$$

---

## 🏆 实验结果

### 真实日用电量尺度

| 模型            | 90 天 MSE（kWh²） | 90 天 MAE（kWh） | 365 天 MSE（kWh²） | 365 天 MAE（kWh） |
| --------------- | ----------------: | ---------------: | -----------------: | ----------------: |
| LSTM            |  **46.05 ± 3.97** |  **5.33 ± 0.23** |       50.12 ± 2.70 |       5.52 ± 0.20 |
| Transformer     |      53.74 ± 5.17 |      5.82 ± 0.34 |       49.59 ± 1.98 |       5.42 ± 0.11 |
| DE-iTransformer |      50.06 ± 6.93 |      5.58 ± 0.37 |   **47.60 ± 1.21** |   **5.36 ± 0.09** |

### 标准化尺度

| 模型            |           90 天 MSE |           90 天 MAE |          365 天 MSE |          365 天 MAE |
| --------------- | ------------------: | ------------------: | ------------------: | ------------------: |
| LSTM            | **0.3932 ± 0.0339** | **0.4928 ± 0.0211** |     0.3883 ± 0.0209 |     0.4860 ± 0.0172 |
| Transformer     |     0.4589 ± 0.0441 |     0.5383 ± 0.0311 |     0.3843 ± 0.0154 |     0.4775 ± 0.0097 |
| DE-iTransformer |     0.4274 ± 0.0592 |     0.5156 ± 0.0345 | **0.3688 ± 0.0094** | **0.4716 ± 0.0080** |

### 结果分析

在 90 天任务中：

* LSTM 取得最低 MSE 和 MAE；
* DE-iTransformer 相较普通 Transformer：

  * MSE 降低约 6.86%；
  * MAE 降低约 4.21%；
* 说明变量 token 建模与趋势分解能够改善普通 Transformer 的短期预测能力。

在 365 天任务中：

* DE-iTransformer 获得三种模型中的最佳结果；
* 相较 LSTM：

  * MSE 降低约 5.02%；
  * MAE 降低约 2.97%；
* 相较 Transformer：

  * MSE 降低约 4.02%；
  * MAE 降低约 1.23%；
* DE-iTransformer 的标准差最小，表现出更好的长期预测稳定性。


---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/ll1317/Machine-Learning.git
cd Machine-Learning
```

### 2. 创建环境

使用 Conda：

```bash
conda create -n power_forecasting python=3.10 -y
conda activate power_forecasting
```

安装依赖：

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### 3. 准备数据

将处理后的文件放入：

```text
data/train.csv
data/test.csv
```

### 4. 检查数据集

```bash
python dataset.py \
  --data-dir ./data \
  --horizon both
```

---

## 🏃 模型训练

### LSTM

快速测试：

```bash
python train_lstm.py \
  --data-dir ./data \
  --output-dir ./outputs/lstm \
  --horizon 90 \
  --seeds 42 \
  --epochs 5 \
  --patience 3 \
  --device auto
```

完整实验：

```bash
python train_lstm.py \
  --data-dir ./data \
  --output-dir ./outputs/lstm \
  --horizon both \
  --seeds 42 52 62 72 82 \
  --epochs 200 \
  --patience 20 \
  --device auto
```

### Transformer

```bash
python train_transformer.py \
  --data-dir ./data \
  --output-dir ./outputs/transformer \
  --horizon both \
  --seeds 42 52 62 72 82 \
  --epochs 200 \
  --patience 20 \
  --device auto
```

### DE-iTransformer

```bash
python train_de_itransformer.py \
  --data-dir ./data \
  --output-dir ./outputs/deitransformer \
  --model deitransformer \
  --horizon both \
  --seeds 42 52 62 72 82 \
  --epochs 200 \
  --patience 20 \
  --device auto
```



---

## 📦 输出文件

每个模型的输出目录包含：

```text
outputs/model_name/
├── checkpoints/
│   └── 最佳模型权重
├── history/
│   └── 每轮训练与验证损失
├── predictions/
│   └── 每轮及五轮平均预测结果
├── figures/
│   └── 损失曲线与预测曲线
├── results/
│   └── MSE、MAE 与 mean ± std
└── run_config.json
```

DE-iTransformer 的汇总结果位于：

```text
outputs/deitransformer/results/deitransformer_summary.csv
```

---

## 🔁 可复现性

项目使用以下随机种子进行 5 次独立实验：

```python
SEEDS = [42, 52, 62, 72, 82]
```

为保证可复现性，代码设置：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

不同硬件、CUDA 和 PyTorch 版本仍可能造成微小数值差异。

---

## 💡 主要结论

* LSTM 在有限数据下具有较强的短期序列归纳能力；
* 普通 Transformer 能建模时间位置间依赖，但在 90 天任务中稳定性较弱；
* 变量 token 有助于建模异构变量之间的交互；
* 7 天趋势—残差分解有助于提取周周期和长期趋势；
* DE-iTransformer 在 90 天任务中优于普通 Transformer；
* DE-iTransformer 在 365 天长期预测中取得最佳平均性能和稳定性；
* 直接多步预测能够避免递归预测中的误差累积，但预测曲线仍可能表现出平滑化倾向。

---

## ⚠️ 局限性

* 数据仅来自单户家庭，泛化能力仍需在更多家庭数据上验证；
* 月度气象数据在同一月份内被重复匹配到每日记录，无法表示日级天气波动；
* 验证集只包含一段完整预测区间，可能对特定季节较敏感；
* 365 天预测跨度较长，未来住户行为和异常事件难以从过去 90 天完全推断；
* 当前门控使用全局标量，后续可扩展为样本级或预测位置级动态门控。

---

## 🔮 后续工作

后续可从以下方向继续改进：

* 使用日级气象数据代替月度统计特征；
* 引入卷积或 patch 模块增强局部变化建模；
* 使用多尺度趋势分解；
* 引入动态门控或逐时间位置融合权重；
* 增加滚动时间验证；
* 在多家庭用电数据集上验证模型泛化能力；
* 研究概率预测与预测区间估计。

---

## 📚 参考文献

1. Hochreiter, S., & Schmidhuber, J.
   **Long Short-Term Memory.**
   *Neural Computation*, 1997.

2. Vaswani, A., et al.
   **Attention Is All You Need.**
   *NeurIPS*, 2017.

3. Liu, Y., et al.
   **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.**
   *ICLR*, 2024.
   Paper: https://openreview.net/forum?id=JePfAI8fah
   Code: https://github.com/thuml/iTransformer

4. Zeng, A., et al.
   **Are Transformers Effective for Time Series Forecasting?**
   *AAAI*, 2023.
   Code: https://github.com/cure-lab/LTSF-Linear

5. Nie, Y., et al.
   **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.**
   *ICLR*, 2023.
   Code: https://github.com/yuqinie98/PatchTST

6. UCI Machine Learning Repository.
   **Individual Household Electric Power Consumption.**
   https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

7. Météo-France.
   **Données climatologiques de base mensuelles.**
   https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles

---


## 👤 Author

**刘丽**

* Major: Artificial Intelligence
* Project: Household Electricity Consumption Forecasting
* GitHub: https://github.com/ll1317/Machine-Learning

---

<div align="center">



</div>
