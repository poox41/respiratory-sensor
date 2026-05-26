# 睡眠分期模型 - 部署说明文档

## 一、模型概述

本模型用于基于心电图（ECG）和呼吸信号进行睡眠分期，判断用户处于**清醒**或**睡眠**状态。

### 性能指标

| 指标 | 数值 |
|------|------|
| 准确率 (Accuracy) | 86.77% |
| Cohen's Kappa | 0.625 |
| Macro F1 | 0.812 |
| 清醒 F1 (Wake F1) | 0.716 |
| 睡眠 F1 (Sleep F1) | 0.914 |

### 模型配置

| 参数 | 值 |
|------|-----|
| 输入特征数 | 57 |
| 序列长度 | 11 个Epoch（约5.5分钟） |
| Epoch长度 | 30秒 |
| 模型类型 | 因果TCN (Causal Temporal Convolutional Network) |
| 决策阈值 | 0.205 |
| 模型版本 | sleep_bcg_v1.0 |

---

## 二、文件说明

### 2.1 模型文件

#### PyTorch格式：`complete_model.pt`
包含：
- 模型结构定义
- 模型权重参数
- 归一化参数（均值、标准差）
- 57个特征名称列表
- 决策阈值（0.205）

**格式**：PyTorch 模型文件（`.pt`）

#### ONNX格式：`sleep_bcg_v1.onnx`
包含：
- 模型结构和权重
- 可跨平台部署（支持Python、C++、移动端等）

**格式**：ONNX 模型文件（`.onnx`）

#### 配置文件：`sleep_bcg_v1.json`
包含：
- 归一化参数（均值、标准差）
- 特征名称列表
- 决策阈值
- 模型配置参数

**格式**：JSON 文件（`.json`）

### 2.2 推理代码：`inference_app.py`

包含：
- 模型结构定义（`CausalTCN` 类）
- `SleepStagingPredictor` 推理类
- 预处理和后处理逻辑

### 2.3 ONNX导出脚本：`export_onnx.py`

用于将PyTorch模型导出为ONNX格式。

### 2.4 特征提取参考：`extract_features_basic_v5_resp_enhanced.py`

57维特征提取的Python实现，作为Android端移植参考。

---

## 三、模型导出（模型端）

### 3.1 导出Android可用的模型文件

```bash
cd work/feature_tcn_simulated_w11
python export_onnx.py --model-dir . --output sleep_bcg_v1.onnx --test
```

### 3.2 输出文件

| 文件 | 说明 |
|------|------|
| `sleep_bcg_v1.onnx` | 模型推理核心 |
| `sleep_bcg_v1.json` | BCG专属的归一化参数和决策阈值 |

### 3.3 交付给Android端的文件

| 文件 | 用途 |
|------|------|
| `sleep_bcg_v1.onnx` | ONNX模型 |
| `sleep_bcg_v1.json` | 配置文件 |
| `extract_features_basic_v5_resp_enhanced.py` | 特征提取参考代码 |

---

## 四、Android端对接指南

### 4.1 核心差异与对齐方案

| 关键维度 | 模型要求 | Android现状 | 必须执行的操作 | 核心影响 |
|----------|----------|-------------|----------------|----------|
| 数据窗口 | 5.5分钟（11个连续30秒片段） | 仅缓存30秒 | 扩展缓存至330秒（16500个采样点） | 内存增加≈200KB |
| 采样率 | 心跳100Hz + 呼吸25Hz | 统一50Hz | 特征提取前重采样 | 计算量增加≈1ms/30秒 |
| 输入信号 | 心跳波形 + 呼吸波形 | 已分离heartWave/respBuf | 直接使用现有波形 | 完全匹配 |
| 特征维度 | 57维增强版 | 可通过波形提取 | 严格移植Python版特征提取 | 直接决定准确率 |
| 部署方式 | 本地ONNX推理 | 支持 | 集成ONNX Runtime Mobile | 单次推理<100ms |
| 输出格式 | awake/sleep/unknown + 置信度 | 要求一致 | 按对接文档返回结果 | 完全匹配 |

### 4.2 Android端对接四步走

#### 步骤1：扩展数据缓存
- 修改`Processor`类的缓存容量，从30秒改为330秒
- 保留原有的循环缓存逻辑，自动覆盖最旧数据
- 新增方法：获取最近30秒的心跳波形和呼吸波形

#### 步骤2：解决采样率不一致问题
- 实现线性插值重采样函数
- 特征提取前重采样：
  - 心跳波形：50Hz → 100Hz（上采样）
  - 呼吸波形：50Hz → 25Hz（下采样）
- 所有时间相关计算使用重采样后的采样率

#### 步骤3：移植57维特征提取代码（最关键）
**核心要求**：逐行对齐Python版，所有参数必须完全一致：
- 滤波系数（心跳5-20Hz带通，呼吸1Hz低通）
- 峰值检测阈值（心跳最小距离0.45秒，呼吸最小距离1.0秒）
- 特征计算顺序（严格对应`feature_cols`顺序）
- NaN值处理（全部替换为0）

**验证标准**：相同输入波形，Android端与Python输出的相对误差<1e-5。

#### 步骤4：集成ONNX模型并实现业务逻辑
1. 添加ONNX Runtime Mobile依赖
2. 加载`assets`目录下的模型和配置文件
3. 实现推理逻辑：归一化 → 构造输入Tensor → 推理 → 计算概率 → 决策
4. 维护长度为11的特征缓冲区，每30秒更新一次
5. 实现按钮点击触发预测和自动更新逻辑
6. 处理异常场景（数据不足、信号质量差、模型错误）

---

## 五、联调验证必做项

### 5.1 计算一致性验证
| 验证项 | 标准 |
|--------|------|
| 重采样验证 | Android端与Python `scipy.signal.resample` 误差<1e-5 |
| 特征一致性验证 | 57个特征逐维对比，相对误差<1e-5 |
| 模型推理一致性验证 | 相同特征输入，与PyTorch绝对误差<1e-4 |

### 5.2 功能验证
| 验证项 | 标准 |
|--------|------|
| 缓冲区测试 | 连续采集6分钟数据，正确存储11个30秒片段 |
| 清醒状态测试 | 采集10分钟清醒数据，输出"清醒"比例≥80% |
| 睡眠状态测试 | 采集10分钟睡眠数据，输出"睡眠"比例≥90% |

### 5.3 异常与性能验证
| 验证项 | 标准 |
|--------|------|
| 传感器脱落 | 返回"信号质量差"，清空缓冲区 |
| 采集中断 | 恢复后继续累加缓冲区，不重置 |
| 数据不足 | 点击按钮返回"数据不足"提示 |
| 性能 | 单次特征提取+推理耗时<100ms |
| 峰值内存 | <100MB |
| 稳定性 | 连续运行24小时无崩溃 |

---

## 六、开发者使用指南

### 6.1 快速开始（PyTorch方式）

```python
from inference_app import SleepStagingPredictor

predictor = SleepStagingPredictor(
    model_dir="work/feature_tcn_simulated_w11",
    device="cuda"
)

features = ...  # 形状: (11, 57)
state, prob = predictor.predict_one(features)

print(f"状态: {'睡眠' if state == 1 else '清醒'}")
print(f"睡眠概率: {prob:.2%}")
```

### 6.2 ONNX方式（Android端参考）

```python
import onnxruntime as ort
import numpy as np
import json

# 加载模型和配置
session = ort.InferenceSession("sleep_bcg_v1.onnx")
with open("sleep_bcg_v1.json", "r") as f:
    config = json.load(f)

mean = np.array(config["mean"], dtype=np.float32)
std = np.array(config["std"], dtype=np.float32)
threshold = config["threshold"]

# 准备输入（形状: (1, 11, 57)）
features = np.random.randn(1, 11, 57).astype(np.float32)
features_norm = (features - mean) / std

# 推理
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
logits = session.run([output_name], {input_name: features_norm})[0]

# 计算概率和决策
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
prob_sleep = prob[0, 1]
state = 1 if prob_sleep >= threshold else 0
```

---

## 七、特征列表（57个）

模型需要57个特征，输入顺序必须与以下列表一致：

| 序号 | 特征名称 | 说明 |
|------|---------|------|
| 0 | ecg_signal_mean | ECG信号均值 |
| 1 | ecg_signal_std | ECG信号标准差 |
| 2 | ecg_signal_range | ECG信号范围 |
| 3 | ecg_flat_ratio | ECG平坦比例 |
| 4 | ecg_zero_crossing_rate | ECG零交叉率 |
| 5 | ecg_peak_count | ECG峰值数量 |
| 6 | ecg_peak_mean | ECG峰值均值 |
| 7 | ecg_peak_std | ECG峰值标准差 |
| 8 | ecg_rr_interval_mean | RR间期均值 |
| 9 | ecg_rr_interval_std | RR间期标准差 |
| 10 | ecg_rr_interval_range | RR间期范围 |
| 11 | hr_mean | 心率均值 |
| 12 | hr_std | 心率标准差 |
| 13 | hr_min | 心率最小值 |
| 14 | hr_max | 心率最大值 |
| 15 | hrv_rmssd | HRV RMSSD |
| 16 | hrv_sdnn | HRV SDNN |
| 17 | hrv_pnn50 | HRV pNN50 |
| 18 | hrv_triangular_index | HRV三角指数 |
| 19 | hrv_tinn | HRV TINN |
| 20 | hrv_lf | HRV低频功率 |
| 21 | hrv_hf | HRV高频功率 |
| 22 | hrv_lf_hf_ratio | LF/HF比率 |
| 23 | hrv_total_power | HRV总功率 |
| 24 | qrs_complex_width_mean | QRS波宽度均值 |
| 25 | qrs_complex_width_std | QRS波宽度标准差 |
| 26 | qrs_complex_amplitude_mean | QRS波幅值均值 |
| 27 | qrs_complex_amplitude_std | QRS波幅值标准差 |
| 28 | st_segment_deviation_mean | ST段偏移均值 |
| 29 | st_segment_deviation_std | ST段偏移标准差 |
| 30 | resp_signal_mean | 呼吸信号均值 |
| 31 | resp_signal_std | 呼吸信号标准差 |
| 32 | resp_signal_range | 呼吸信号范围 |
| 33 | resp_signal_skewness | 呼吸信号偏度 |
| 34 | resp_signal_kurtosis | 呼吸信号峰度 |
| 35 | resp_amplitude_mean | 呼吸幅度均值 |
| 36 | resp_amplitude_std | 呼吸幅度标准差 |
| 37 | resp_period_mean | 呼吸周期均值 |
| 38 | resp_period_std | 呼吸周期标准差 |
| 39 | resp_rate | 呼吸率 |
| 40 | resp_regularity | 呼吸规律性 |
| 41 | resp_rr_interval_mean | 呼吸RR间期均值 |
| 42 | resp_rr_interval_std | 呼吸RR间期标准差 |
| 43 | resp_rr_interval_skewness | 呼吸RR间期偏度 |
| 44 | resp_rr_interval_kurtosis | 呼吸RR间期峰度 |
| 45 | resp_stability | 呼吸稳定性 |
| 46 | resp_asymmetry | 呼吸不对称性 |
| 47 | breath_hold_ratio | 屏息比例 |
| 48 | sigh_count | 叹息次数 |
| 49 | sigh_ratio | 叹息比例 |
| 50 | apnea_likelihood | 呼吸暂停可能性 |
| 51 | baseline_drift | 基线漂移 |
| 52 | resp_signal_quality | 呼吸信号质量 |
| 53 | resp_snr | 呼吸信噪比 |
| 54 | resp_peak_count | 呼吸峰值数量 |
| 55 | resp_peak_regularity | 呼吸峰值规律性 |
| 56 | combined_metric | 组合指标 |

---

## 八、关键注意事项

1. **绝对不能直接用50Hz波形计算特征**：所有时间相关参数都是基于100Hz/25Hz设计的，直接使用会导致准确率暴跌至50%以下
2. **特征提取是准确率的核心**：任何参数的微小改动都会导致模型输出偏差，必须严格对齐Python版
3. **不要修改模型输入形状**：模型固定输入为(1, 11, 57)，任何形状不匹配都会导致推理失败
4. **使用BCG专属的配置参数**：不要使用原ECG模型的mean/std和阈值，否则归一化会完全错误
5. **缓冲区管理**：必须维护至少11个30秒片段，不足时返回"数据不足"

---

## 九、版本信息

- 模型版本：sleep_bcg_v1.0
- 训练日期：2026-05-25
- 基础模型：因果TCN
- 训练数据集：SHHS (Sleep Heart Health Study)
- 支持格式：PyTorch (`.pt`)、ONNX (`.onnx`)
- 适用平台：Python后端、Android移动端
