# App 75%峰谷锐化实现记录

## 实现范围

本次只修改“周期增强心搏展示信号”，不修改正式 `cleanHeart`、峰值检测、BPM、RPM、人体在位判断或睡眠模型输入。

展示链如下：

1. `xF` 进入独立的 1–10 Hz 展示细节带通；
2. 心搏时刻继续来自原正式 `cleanHeart` 检测器；
3. 相邻真实时刻之间的细节信号被重采样、去基线、归一化及极性/相位对齐；
4. 在最近最多 8 个保留周期中计算逐点中位稳健模板；
5. 选择与其余周期平均相似度最高的真实周期（medoid）；
6. 使用 `75% medoid + 25% 稳健模板` 生成展示形态；
7. 模板连续拼接，仅写入 `enhancedHeartBuf` 和 `heartTemplateEnhanced`。

## 日志标签

新增：

- `heartTemplateInput`：独立 1–10 Hz 展示细节通道；

保留：

- `cleanHeart`：正式连续心搏波形；
- `heartTemplateEnhanced`：75%保形周期增强输出；
- `heartTemplateQuality`：最终模板与保留真实周期的一致性；
- `heartTemplateCycles`：参与当前模板的周期数；
- `heartTemplateReady`：模板是否已准备完成；
- `bpm`：原正式 BPM，未由增强波形计算。

## 界面标注

卡片标题和说明已明确标注“75%代表周期保形、非原始波形、不参与 BPM”。

## 验证

- `HeartbeatTemplateBlendTest`：验证 75%模板确实更接近所选代表性真实周期，且模板一致性不低于 0.80；
- `HeartbeatTemplateContinuityTest`：验证周期相位复位时单采样跳变仍不超过 0.281；
- 其余原有单元测试全部通过；
- `assembleDebug` 构建通过。

## 下一次采集检查

建议仍采用坐姿椅背方式，保持安静 60 秒以上。采集后同时检查 `heartTemplateInput`、`heartTemplateEnhanced`、`heartTemplateQuality` 和 `bpm`。增强图建立需要至少 3 个有效周期；论文截图建议等待 `heartTemplateCycles` 达到 6–8 且一致性稳定后再截取。
