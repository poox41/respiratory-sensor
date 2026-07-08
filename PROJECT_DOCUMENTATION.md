# Respiratory Sensor - 项目完整文档

## 整体架构

项目包名 `com.example.breathheartdemo`，是一个 Android 应用，通过 **BLE 蓝牙** 接收 12 位压力传感器的 ADC 数据，实时分离呼吸和心率信号，在界面上显示波形和 BPM/RPM 数值。

### 关键文件与职责

| 文件 | 职责 |
|---|---|
| `MainActivity.kt` | Compose UI、数据源切换、波形显示 |
| `BleClient.kt` | BLE 扫描连接、Little Endian 解析、采样时钟 |
| `Processor.kt` | **核心处理链**：异常过滤→去基线→预滤波→分离→BPM/RPM |
| `RateEstimator.kt` | DFT 频域估算 + 平滑器 + 呼吸率门控 |
| `SimpleFilters.kt` | 滑动平均、DualSmoother、IIR 高通 |
| `Waveform.kt` | Canvas 波形绘制、网格、心搏周期标记 |
| `RingBuffer.kt` | 循环缓冲区 |
| `MockDataSource.kt` | 模拟数据源 |
| `SleepStateService.kt` | 睡眠状态判断 |
| `SensorDataLogger.kt` | CSV 日志记录 |

---

## 完整处理流程

```text
BLE → 字节流 → Little Endian → u16 ADC → 异常过滤 → 去基线(EMA 40s)
Raw Buffer (原始波形) ← rawBuf
    ↓
8 点 MA 预滤 → Pre Buffer
    ↓
┌─ 100 点 MA → Resp Buffer (呼吸波形) → DFT(30s) → RPM
└─ IIR HPF(0.3Hz) → HR →
   ├─ 10 点 MA → DFT(20s) → RateSmoother → BPM
   └─ DualSmoother → 5s min-max 归一化 → HR Buffer (心率波形)
       → 峰值检测(>0.3, 350ms 不应期) → 心搏周期标记
```

---

## 步骤 1：蓝牙数据接收与解析

**文件**: `BleClient.kt` | 采样率: **50 Hz**

- UUID: `0000abf0-...` | 特征: `0000abf2-...`
- Little Endian → 12-bit 无符号整数 (0~4095)
- 异常过滤: 原始数据中滤除 ADC=0/4095（`BleClient` 层已做初步过滤）

---

## 步骤 2：数据合法性检查

**Processor.kt:133** — ADC=0 或 4095 时用上一有效值替代

```kotlin
val validatedRawX = if (rawX <= 0.5f || rawX >= 4094.5f)
    lastValidAdc else rawX.also { lastValidAdc = it }
```

---

## 步骤 3：极慢基线漂移校正 (EMA)

**Processor.kt:137-139** — 时间常数 **~40 秒**

| 参数 | 值 | 含义 |
|---|---|---|
| `dcAlpha` | 0.9995 | EMA 系数 |
| 时间常数 | 1/(1-0.9995)/50 ≈ 40s | 50 Hz 下 |

---

## 步骤 4：预滤波 (8 点 MA)

**Processor.kt:140** — `preLP.next(x)`

| 参数 | 值 |
|---|---|
| 窗口 | 8 点 ≈ 160ms |
| 截止 | ≈ 6 Hz |
| 存储 | `preBuf.add(t, xF)` |

---

## 步骤 5：呼吸提取 (100 点 MA)

**Processor.kt:175** — `respLP.next(xF)`

| 参数 | 值 | 含义 |
|---|---|---|
| 窗口 | 100 (2s) | 2 秒滑动平均 |
| -3dB 截止 | ≈ 0.22 Hz | 低于呼吸频率 |
| 增益 @0.25Hz | 0.637 | 呼吸通过 63.7% |
| 增益 @1.25Hz | 0.127 | 心率衰减至 12.7% |
| 存储 | `respBuf.add(t, resp)` | |

---

## 步骤 6：心率分离 (IIR 高通)

**Processor.kt:183** — `hrHPF.next(xF)`

| 参数 | 值 | 含义 |
|---|---|---|
| `alpha` | 0.964 | 截止 0.3 Hz |
| 增益 @0.25Hz | 0.633 | 呼吸衰减 36.7% |
| 增益 @1.25Hz | 0.958 | 心率仅衰减 4.2% |

```kotlin
class HighPassFilter(alpha: Float) {
    // y[n] = alpha * (y[n-1] + x[n] - x[n-1])
}
```

---

## 步骤 7：双支路处理

| 通道 | 滤波器 | 窗口 | 用途 |
|---|---|---|---|
| 计算 (`hr`) | MovingAverage | 10 点 (200ms) | DFT + 峰值检测 |
| 显示 (`hrDisplay`) | DualSmoother | 12+12 (480ms) | 波形绘制 |

---

## 步骤 8：动态归一化

| 参数 | 值 |
|---|---|
| 窗口 | 5 秒 (250 样本) |
| 输出 | -1 ~ 1 |
| 钳位 | `coerceIn(-1f, 1f)` |

> 仅用于显示，不参与估算

---

## 步骤 9：峰值检测

| 参数 | 值 |
|---|---|
| 上升阈值 | > 0.3 |
| 回落阈值 | < -0.1 |
| 不应期 | **350ms** |
| BPM 公式 | 60000 / interval(ms) |

---

## 步骤 10：DFT 频率估算

**文件**: `RateEstimator.kt`

### 心率

| 参数 | 值 |
|---|---|
| 窗口 | **20 秒** (1000 样本) |
| 分辨率 | 0.05 Hz (3 BPM) |
| 75 BPM | k=25 (精确波箱中心) |
| 扫描范围 | 40~180 BPM |
| 窗函数 | Hann (旁瓣 -32dB) |
| 估算间隔 | 1 秒 |

### 呼吸

| 参数 | 值 |
|---|---|
| 窗口 | 30 秒 (1500 样本) |
| 扫描 | 6~30 RPM |
| 估算间隔 | 2 秒 |

---

## 步骤 11：平滑与验证

| 参数 | 心率 | 呼吸 |
|---|---|---|
| historySize | 3 | 5 |
| alpha | 0.45 | 0.25 |
| maxStep/update | 3 BPM | 0.8 RPM |
| 有效范围 | 40~180 BPM | 8~30 RPM |
| 可靠性 | >= 3 次有效 | >= 3 次有效 |

---

## 步骤 12：呼吸率门控

| 参数 | 值 |
|---|---|
| minAmplitude | 0.12 |
| maxJumpRpm | 4 |
| maxHoldMs | 12000ms |

---

## 步骤 13：信号质量检测

| 参数 | 值 |
|---|---|
| 运动窗口 | 120 样本 (2.4s) |
| 运动阈值 | 80 ADC counts |
| 判定比例 | 30% |
| 恢复延迟 | 3000ms |

---

## 步骤 14：缓冲区与波形

| 缓冲区 | 数据 | 显示 |
|---|---|---|
| `rawBuf` | 原始 ADC | 原始压力波形 |
| `preBuf` | 预滤信号 | 调试 |
| `respBuf` | 呼吸信号 | 呼吸波形 |
| `hrBuf` | -1~1 归一化 | 心率波形 |

| 波形 | y 范围 | 增益 | 周期标记 |
|---|---|---|---|
| 原始 | -32768~32767 | 自动 | 无 |
| 呼吸 | -2~2 | 1.0 | 无 |
| 心率 | -1~1 | 1.0 (可自动) | 矩形虚线 |

---

## 步骤 15：CSV 日志

字段: `time_ms, time_str, rawX, x, xF`
路径: `Documents/sensor_logs/sensor_YYYYMMDD_HHmmss.csv`

---

## 模拟数据

```kotlin
val resp = 1.0f * sin(2π x 0.25 x sec)    // 15 RPM
val hr   = 0.6f * sin(2π x 1.25 x sec)    // 75 BPM
```

| 参数 | 默认 |
|---|---|
| fsHz | 50 Hz |
| BPM | 75 |
| RPM | 15 |

---

## 修复记录

| 问题 | 修复 |
|---|---|
| sensorLogger 类外声明 | 移入类体 |
| import 位置错误 | 删除同包 import |
| 波形削顶 | `coerceIn(-1f, 1f)` |
| BPM/RPM 不显示 | 持久化估算值 |
| RPM 时序错误 | 移到验证前 |
| 基线太快 | dcAlpha 0.995→0.9995 |
| DFT 泄漏→60BPM | Hann 窗 |
| 呼吸 4.4x 强于心率 | IIR 高通 0.3Hz |
| 分辨率 0.1Hz | windowSec 10→20 |
| 异常数值 0/4095 | 替换为上一值 |
| Buffer 存错数据 | rawBuf 存 ADC, 新增 preBuf |
