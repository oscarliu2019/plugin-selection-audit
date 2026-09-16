# 第三方代码与数据说明

本仓库包含或依赖以下第三方成果，其著作权与许可归原作者所有。

## 代码

### Time-Series-Library (TSLib)

- 路径：`third_party/tslib/`
- 来源：清华大学 THUML 团队开源的 Time-Series-Library
- 许可：MIT License，完整许可文本见 `third_party/tslib/LICENSE`
- 使用方式：**未修改源码**。本项目所有插件（RevIN / SAN-lite / FreDF）与
  逐窗口误差落盘均通过外层包装实现（见 `src/runner.py`、`src/plugins/`），
  以及运行期的 FFT dtype 兼容包装（`src/fft_compat.py`）。
  这样做的目的是保证"骨干实现"与公开基线严格一致，插件收益的差值可归因。

本项目使用了 TSLib 的骨干模型实现（DLinear、PatchTST、TimesNet、iTransformer）
与其 `data_provider` 数据管线。

## 数据集

本项目使用长期时间序列预测领域的公开标准数据集：
ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Exchange、ILI、Weather。

这些数据集**不包含在本仓库中**（`.gitignore` 已排除 `data/`）。
请按 TSLib / Autoformer 仓库公布的地址自行下载，并遵守各数据集原始的许可与引用要求。
数据集的原始出处与引用见论文 `paper/main.tex` 的参考文献部分。

## 论文中被比较/复现的方法

本项目实现的插件是对以下工作的**轻量复现**，用于插件选择的审计实验，
并非原作者的官方实现；结论中的任何不足不应归因于原方法：

- RevIN（Reversible Instance Normalization）
- SAN（Slice Adaptive Normalization）—— 本项目实现的是简化版 `san_lite`
- FreDF（Frequency-domain 预测损失）

具体引用条目见 `paper/main.tex`。
