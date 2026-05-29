# EEG Motor Imagery Decoding with EEGNet

这是一个用 PyTorch + Braindecode 在 PhysioNet motor imagery 数据集上做运动想象解码的项目。目标是探索 cross-subject BCI 的真实表现以及 fine-tuning 的有效性。

## 实验结果

四方案对比：

| 方案 | 准确率 |
|------|--------|
| 方案 2：cross-subject 直接用 | 52.0% |
| 方案 3：全模型 fine-tune（5 被试平均） | 52.0% ± 4.9% |
| 方案 4：只调最后一层（5 被试平均） | 45.7% ± 4.8% |
| 随机猜测 baseline | 50.0% |

## 主要发现

- Cross-subject variability 真实存在；
- Naive fine-tune 在 10-trial calibration 下因 few-shot overfitting 失败；
- 只调最后一层（参数从 2930 减到 802）仍然不够，因为参数数仍远大于样本数。

## 环境

- Python 3.12
- PyTorch 2.12
- Braindecode 1.5.1
- MNE 1.12.1

## 运行方式

```bash
source venv/bin/activate
jupyter notebook eeg_motor_imagery_v1.ipynb
```

## 作者

李一恒（yihengli23@gmail.com）
