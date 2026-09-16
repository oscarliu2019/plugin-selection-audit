"""PlugGate: 「插件何时有效」的复杂度条件化选择器研究代码库。

模块地图
--------
config     : 实验矩阵配置载入 + Cell（单格实验）定义与枚举
features   : ⭐ 从原始序列廉价计算复杂度/可预测性统计量（核心科学贡献之一）
plugins    : 零侵入插件包装层（none / revin / san_lite / fredf）
runner     : 单格实验执行（调用 TSLib，不修改其源码）
scheduler  : 测速→排产→断点续跑→原子落盘
selector   : 复杂度特征 → 插件选择的轻量可解释模型 + leave-one-dataset-out
stats      : Wilcoxon / Friedman+Nemenyi / CD 图 / win-tie-loss / Holm
horizon    : horizon-aware 早停与 checkpoint 选择准则（零算力增量的附加贡献）
report     : 结果 CSV → 投稿用 LaTeX 表格与图
synth      : 合成结果生成器（无 GPU 环境下端到端验证分析流水线）
"""

__version__ = "0.1.0"
